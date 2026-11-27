#!/usr/bin/env python3
"""
Pre-processing / cleaning of raw Livox Mid-360S point clouds.

Filter chain (cheap stages first, so the expensive ones see fewer points):

    raw cloud
      -> 1 invalid       NaN / Inf points
      -> 2 range         keep 0.03 m <= range <= 40 m   (also drops the (0,0,0)
                         "no return" points the Livox driver emits)
      -> 3 livox tag     points the sensor itself flags as noise (only if the
                         cloud has a 'tag' field, i.e. livox_ros_driver2)
      -> 4 body box      points on the robot's own body (off by default)
      -> 5 mixed pixel   "veil" / edge artefacts: points floating between a
                         foreground and a background surface
      -> 6 SOR           statistical outlier removal (range normalised so that
                         sparse far-away points are not punished)
      -> 7 DROR          dynamic radius outlier removal (search radius grows
                         with range, removes rain / dust / isolated specks)
      -> 8 voxel         optional down-sampling (off by default)
      -> clean cloud

Two ways to run it:

  ROS 2 node (default):
      python3 livox_preprocess.py
      Subscribes /livox/lidar, publishes
        /filter/output   cleaned cloud (x, y, z, intensity)
        /filter/removed  rejected points; field 'stage' = stage number 1..8
      Change any setting while it runs:
        ros2 param set /livox_preprocess range_max 30.0
        ros2 run rqt_reconfigure rqt_reconfigure

  Offline, on a file (.pcd / .ply / .xyz / .npy):
      python3 livox_preprocess.py --in raw.pcd --out clean.pcd
"""
import argparse
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

# ============================ DEFAULT SETTINGS ============================
# (name, default, min, max, description). min/max become rqt sliders.
PARAMS = [
    ('input_topic', '/livox/lidar', None, None, 'Raw cloud topic (startup only).'),
    ('log_every', 10, 1, 1000, 'Print a stats line every N frames.'),

    # 2 ---- range
    ('range_min', 0.03, 0.0, 5.0, 'Drop points closer than this (m).'),
    ('range_max', 40.0, 1.0, 100.0, 'Drop points farther than this (m).'),

    # 3 ---- Livox tag (bits 0-1 spatial noise, bits 2-3 intensity noise)
    ('use_tag', True, None, None, 'Stage 3 on/off (needs a tag field).'),
    ('tag_level', 1, 1, 3,
     'Livox noise confidence to reject: 1 = only high-confidence noise, '
     '2 = high + medium, 3 = high + medium + low.'),

    # 4 ---- robot body box (centred on the sensor, half sizes in m)
    ('use_body', False, None, None, 'Stage 4 on/off.'),
    ('body_x', 0.45, 0.0, 3.0, 'Half length of body box (m).'),
    ('body_y', 0.30, 0.0, 3.0, 'Half width of body box (m).'),
    ('body_z', 0.30, 0.0, 3.0, 'Half height of body box (m).'),

    # 5 ---- mixed pixel / veil
    ('use_veil', True, None, None, 'Stage 5 on/off.'),
    ('veil_k', 12, 2, 30, 'Angular neighbours examined.'),
    ('veil_angle_deg', 1.5, 0.2, 5.0, 'Max angular distance to a neighbour (deg).'),
    ('veil_jump_abs', 0.20, 0.02, 3.0, 'Min range jump on each side (m).'),
    ('veil_jump_rel', 0.05, 0.0, 0.5, 'Extra range jump, fraction of range.'),

    # 6 ---- statistical outlier removal
    ('use_sor', True, None, None, 'Stage 6 on/off.'),
    ('sor_k', 10, 2, 50, 'Neighbours used to measure isolation.'),
    ('sor_std_ratio', 2.5, 0.1, 10.0, 'Smaller = removes more.'),

    # 7 ---- dynamic radius outlier removal
    ('use_dror', True, None, None, 'Stage 7 on/off.'),
    ('dror_angular_res_deg', 1.0, 0.05, 5.0, 'Typical spacing between points (deg).'),
    ('dror_multiplier', 3.0, 0.5, 20.0, 'Search radius = multiplier * range * res.'),
    ('dror_min_radius', 0.05, 0.005, 1.0, 'Smallest search radius (m).'),
    ('dror_min_neighbors', 2, 1, 20, 'Neighbours needed inside the radius.'),

    # 8 ---- voxel down-sampling
    ('voxel_size', 0.0, 0.0, 1.0, 'Voxel edge (m). 0 = off.'),
]
DEFAULTS = {name: default for name, default, *_ in PARAMS}
STAGE_NAMES = ('invalid', 'range', 'tag', 'body', 'veil', 'sor', 'dror', 'voxel')
# ==========================================================================


# ------------------------------- the filters ------------------------------
def stage_range(xyz, r, p):
    return (r >= p['range_min']) & (r <= p['range_max'])


def stage_tag(tag, p):
    """Livox tag: bits[1:0] spatial noise, bits[3:2] intensity noise.
    value 1 = high, 2 = medium, 3 = low confidence that the point is noise."""
    tag = tag.astype(np.uint8)
    bad_levels = np.arange(1, int(p['tag_level']) + 1)
    spatial = tag & 0x03
    intens = (tag >> 2) & 0x03
    return ~(np.isin(spatial, bad_levels) | np.isin(intens, bad_levels))


def stage_body(xyz, p):
    inside = ((np.abs(xyz[:, 0]) < p['body_x']) &
              (np.abs(xyz[:, 1]) < p['body_y']) &
              (np.abs(xyz[:, 2]) < p['body_z']))
    return ~inside


def stage_veil(xyz, r, p):
    """Mixed-pixel filter. A laser spot that hits an edge returns a range
    somewhere between the foreground and the background. Such a point has,
    among its angular neighbours, one clearly nearer AND one clearly farther."""
    n = len(xyz)
    k = int(p['veil_k'])
    if n <= k:
        return np.ones(n, bool)
    dirs = xyz / r[:, None]
    chord = 2.0 * np.sin(np.deg2rad(p['veil_angle_deg']) / 2.0)
    _, idx = cKDTree(dirs).query(dirs, k=k + 1, distance_upper_bound=chord,
                                 workers=-1)
    idx = idx[:, 1:]                       # drop the point itself
    valid = idx < n                        # missing neighbours have idx == n
    rr = np.append(r, 0.0)[idx]
    near = np.where(valid, rr, np.inf).min(axis=1)
    far = np.where(valid, rr, -np.inf).max(axis=1)
    # On a smooth surface seen at a grazing angle (e.g. the ground) range also
    # changes from neighbour to neighbour, but some neighbour is always at a
    # similar range. A mixed pixel has none: it is isolated in range.
    closest = np.where(valid, np.abs(rr - r[:, None]), np.inf).min(axis=1)
    jump = p['veil_jump_abs'] + p['veil_jump_rel'] * r
    return ~((near < r - jump) & (far > r + jump) & (closest > jump))


def stage_sor(xyz, r, p):
    """Statistical outlier removal on range-normalised neighbour distance.
    LiDAR point spacing grows linearly with range, so dividing by range lets
    one global threshold work from 3 cm to 40 m."""
    n = len(xyz)
    k = int(p['sor_k'])
    if n <= k:
        return np.ones(n, bool)
    d, _ = cKDTree(xyz).query(xyz, k=k + 1, workers=-1)
    score = d[:, 1:].mean(axis=1) / np.maximum(r, 0.5)
    return score <= score.mean() + p['sor_std_ratio'] * score.std()


def stage_dror(xyz, r, p):
    """Dynamic radius outlier removal (Charron et al. 2018)."""
    n = len(xyz)
    if n == 0:
        return np.ones(0, bool)
    radius = np.maximum(p['dror_min_radius'],
                        p['dror_multiplier'] * r *
                        np.deg2rad(p['dror_angular_res_deg']))
    counts = cKDTree(xyz).query_ball_point(xyz, radius, return_length=True,
                                           workers=-1)
    return (counts - 1) >= int(p['dror_min_neighbors'])


def stage_voxel(xyz, p):
    """Keep the first point in every occupied voxel."""
    keys = np.floor(xyz / p['voxel_size']).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    keep = np.zeros(len(xyz), bool)
    keep[first] = True
    return keep


def run_filters(xyz, p, tag=None):
    """Run the whole chain.

    xyz : (N, 3) float array, sensor frame
    p   : dict of settings (see PARAMS)
    tag : optional (N,) uint8 Livox tag
    Returns (keep_mask, stage_code, stats):
      keep_mask  (N,) bool, True = point survived
      stage_code (N,) uint8, 0 = kept, otherwise number of the stage that removed it
      stats      list of (stage_name, removed_count, milliseconds)
    """
    n = len(xyz)
    code = np.zeros(n, np.uint8)
    alive = np.arange(n)                   # indices of surviving points
    stats = []
    xyz = np.asarray(xyz, np.float64)

    def apply(stage_no, keep_local, t0):
        nonlocal alive
        code[alive[~keep_local]] = stage_no
        stats.append((STAGE_NAMES[stage_no - 1], int((~keep_local).sum()),
                      (time.perf_counter() - t0) * 1e3))
        alive = alive[keep_local]

    t = time.perf_counter()
    apply(1, np.isfinite(xyz).all(axis=1), t)

    r_all = np.zeros(n)
    r_all[alive] = np.linalg.norm(xyz[alive], axis=1)

    t = time.perf_counter()
    apply(2, stage_range(xyz[alive], r_all[alive], p), t)

    if p['use_tag'] and tag is not None:
        t = time.perf_counter()
        apply(3, stage_tag(tag[alive], p), t)
    if p['use_body']:
        t = time.perf_counter()
        apply(4, stage_body(xyz[alive], p), t)
    if p['use_veil']:
        t = time.perf_counter()
        apply(5, stage_veil(xyz[alive], r_all[alive], p), t)
    if p['use_sor']:
        t = time.perf_counter()
        apply(6, stage_sor(xyz[alive], r_all[alive], p), t)
    if p['use_dror']:
        t = time.perf_counter()
        apply(7, stage_dror(xyz[alive], r_all[alive], p), t)
    if p['voxel_size'] > 0 and len(alive):
        t = time.perf_counter()
        apply(8, stage_voxel(xyz[alive], p), t)

    keep = np.zeros(n, bool)
    keep[alive] = True
    return keep, code, stats


def format_stats(n_in, n_out, stats):
    total = sum(s[2] for s in stats)
    parts = ' '.join(f'{name}:-{cnt}' for name, cnt, _ in stats)
    return f'{n_in} -> {n_out} pts ({total:.1f} ms) | {parts}'


# ------------------------------- ROS 2 node -------------------------------
def ros_main():
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rcl_interfaces.msg import (FloatingPointRange, IntegerRange,
                                    ParameterDescriptor, SetParametersResult)
    from sensor_msgs.msg import PointCloud2, PointField

    np_types = {PointField.INT8: 'i1', PointField.UINT8: 'u1',
                PointField.INT16: 'i2', PointField.UINT16: 'u2',
                PointField.INT32: 'i4', PointField.UINT32: 'u4',
                PointField.FLOAT32: 'f4', PointField.FLOAT64: 'f8'}

    def read_cloud(msg):
        """PointCloud2 -> xyz (N,3) float32, intensity (N,) or None, tag (N,) or None."""
        fields = {f.name: f for f in msg.fields}
        if not all(k in fields for k in 'xyz'):
            return None, None, None
        n = msg.width * msg.height
        endian = '>' if msg.is_bigendian else '<'
        names = [k for k in ('x', 'y', 'z', 'intensity', 'reflectivity', 'tag')
                 if k in fields]
        dt = np.dtype({'names': names,
                       'formats': [endian + np_types[fields[k].datatype] for k in names],
                       'offsets': [fields[k].offset for k in names],
                       'itemsize': msg.point_step})
        arr = np.frombuffer(msg.data, dtype=dt, count=n)
        xyz = np.stack([arr['x'], arr['y'], arr['z']], axis=1).astype(np.float32)
        inten_name = 'intensity' if 'intensity' in names else (
            'reflectivity' if 'reflectivity' in names else None)
        inten = arr[inten_name].astype(np.float32) if inten_name else None
        tag = arr['tag'].astype(np.uint8) if 'tag' in names else None
        return xyz, inten, tag

    def make_cloud(header, xyz, extra_name, extra):
        out = np.empty(len(xyz), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                        (extra_name, '<f4')])
        out['x'], out['y'], out['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        out[extra_name] = extra
        m = PointCloud2()
        m.header = header
        m.height, m.width = 1, len(xyz)
        m.fields = [PointField(name=nm, offset=4 * i, datatype=PointField.FLOAT32,
                               count=1)
                    for i, nm in enumerate(('x', 'y', 'z', extra_name))]
        m.is_bigendian = False
        m.point_step = 16
        m.row_step = 16 * len(xyz)
        m.is_dense = True
        m.data = out.tobytes()
        return m

    class LivoxPreprocess(Node):
        def __init__(self):
            super().__init__('livox_preprocess')
            for name, default, lo, hi, desc in PARAMS:
                d = ParameterDescriptor(description=desc)
                if isinstance(default, float) and lo is not None:
                    d.floating_point_range = [FloatingPointRange(
                        from_value=float(lo), to_value=float(hi), step=0.0)]
                elif isinstance(default, int) and not isinstance(default, bool) \
                        and lo is not None:
                    d.integer_range = [IntegerRange(
                        from_value=int(lo), to_value=int(hi), step=1)]
                self.declare_parameter(name, default, d)
            self.p = {name: self.get_parameter(name).value for name in DEFAULTS}
            self.add_on_set_parameters_callback(self.on_params)

            self.pub_out = self.create_publisher(PointCloud2, '/filter/output',
                                                 qos_profile_sensor_data)
            self.pub_rej = self.create_publisher(PointCloud2, '/filter/removed',
                                                 qos_profile_sensor_data)
            topic = self.p['input_topic']
            self.create_subscription(PointCloud2, topic, self.on_cloud,
                                     qos_profile_sensor_data)
            self.frames = 0
            self.warned_tag = False
            self.get_logger().info(f'Filtering {topic} -> /filter/output')

        def on_params(self, params):
            new = dict(self.p)
            for prm in params:
                new[prm.name] = prm.value
            if new['range_min'] >= new['range_max']:
                return SetParametersResult(successful=False,
                                           reason='range_min must be < range_max')
            self.p = new
            return SetParametersResult(successful=True)

        def on_cloud(self, msg):
            xyz, inten, tag = read_cloud(msg)
            if xyz is None:
                self.get_logger().error('Cloud has no x/y/z fields', once=True)
                return
            if tag is None and self.p['use_tag'] and not self.warned_tag:
                self.get_logger().info('No "tag" field in cloud: Livox noise '
                                       'tag stage skipped.')
                self.warned_tag = True
            if inten is None:
                inten = np.zeros(len(xyz), np.float32)

            keep, code, stats = run_filters(xyz, self.p, tag)

            self.pub_out.publish(make_cloud(msg.header, xyz[keep], 'intensity',
                                            inten[keep]))
            if self.pub_rej.get_subscription_count() > 0:
                rej = ~keep & (code != 1)            # can't publish NaN points
                self.pub_rej.publish(make_cloud(msg.header, xyz[rej], 'stage',
                                                code[rej].astype(np.float32)))

            self.frames += 1
            if self.frames % int(self.p['log_every']) == 0:
                self.get_logger().info(format_stats(len(xyz), int(keep.sum()), stats))

    rclpy.init()
    node = LivoxPreprocess()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


# ------------------------------ offline mode ------------------------------
def load_file(path):
    """Returns xyz (N,3), intensity or None."""
    if path.endswith('.npy'):
        a = np.load(path)
        return a[:, :3], (a[:, 3] if a.shape[1] > 3 else None)
    import open3d as o3d
    pc = o3d.io.read_point_cloud(path, remove_nan_points=False,
                                 remove_infinite_points=False)
    return np.asarray(pc.points), None


def save_file(path, xyz, inten):
    if path.endswith('.npy'):
        np.save(path, xyz if inten is None else np.column_stack([xyz, inten]))
        return
    import open3d as o3d
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz.astype(np.float64)))
    o3d.io.write_point_cloud(path, pc)


def file_main(args):
    p = dict(DEFAULTS)
    p['range_min'], p['range_max'] = args.range_min, args.range_max
    p['voxel_size'] = args.voxel
    xyz, inten = load_file(args.inp)
    keep, _, stats = run_filters(xyz, p)
    print(format_stats(len(xyz), int(keep.sum()), stats))
    save_file(args.out, np.asarray(xyz)[keep], None if inten is None else inten[keep])
    print(f'saved {args.out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in', dest='inp', help='Input file. Omit to run as ROS 2 node.')
    ap.add_argument('--out', help='Output file (offline mode).')
    ap.add_argument('--range-min', type=float, default=DEFAULTS['range_min'])
    ap.add_argument('--range-max', type=float, default=DEFAULTS['range_max'])
    ap.add_argument('--voxel', type=float, default=DEFAULTS['voxel_size'])
    args, _ = ap.parse_known_args()        # ignore ROS --ros-args
    if args.inp:
        if not args.out:
            sys.exit('--out is required with --in')
        file_main(args)
    else:
        ros_main()
