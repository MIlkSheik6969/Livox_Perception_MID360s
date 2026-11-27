    #!/usr/bin/env python3
"""
Mid-360S point cloud filter chain. Every parameter can be changed while it runs.

    raw cloud
      -> 1 range filter
      -> 2 intensity filter
      -> 3 statistical outlier removal
      -> 4 dynamic radius outlier removal
      -> 5 edge filter
      -> filtered cloud

Publishes:
  /filter/output   the points that survived all stages (x, y, z, intensity)
  /filter/removed  the points that were thrown away. Its "intensity" field holds
                   the number of the stage that removed each point (1 to 5).

Change a setting while it runs:   ros2 param set /lidar_filter range_max 10.0
List all settings:                ros2 param list /lidar_filter
Or use sliders:                   ros2 run rqt_reconfigure rqt_reconfigure
"""
import time

import numpy as np
from scipy.spatial import cKDTree

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.msg import (FloatingPointRange, IntegerRange,
                                ParameterDescriptor, SetParametersResult)
from sensor_msgs.msg import PointCloud2, PointField

EDGE_MODES = ('mixed', 'both', 'behind')

# (name, default, min, max, description). min/max give sliders in rqt_reconfigure.
# Float settings need a decimal point when you set them: 10.0, not 10.
PARAMS = [
    ('input_topic', '/livox/lidar', None, None,'Cloud to filter. Only read at startup.'),
    ('log_every', 10, 1, 1000, 'Print one statistics line every N frames.'),

    # 1 ---- range
    ('use_range', True, None, None, 'Stage 1 on/off.'),
    ('range_min', 0.3, 0.0, 50.0, 'Remove points closer than this (m).'),
    ('range_max', 15.0, 0.5, 100.0, 'Remove points farther than this (m).'),

    # 2 ---- intensity
    ('use_intensity', True, None, None, 'Stage 2 on/off.'),
    ('intensity_min', 0.0, 0.0, 255.0, 'Remove points weaker than this.'),
    ('intensity_max', 255.0, 0.0, 255.0, 'Remove points brighter than this.'),

    # 3 ---- statistical outlier removal
    ('use_sor', True, None, None, 'Stage 3 on/off.'),
    ('sor_k', 10, 1, 50, 'Neighbours used to measure how isolated a point is.'),
    ('sor_std_ratio', 2.0, 0.1, 5.0,
     'Smaller = removes more. Cutoff is mean + ratio * std of neighbour distances.'),

    # 4 ---- dynamic radius outlier removal
    ('use_dror', True, None, None, 'Stage 4 on/off.'),
    ('dror_angular_res_deg', 1.0, 0.05, 5.0,'Assumed spacing between neighbouring points, in degrees.'),
    ('dror_multiplier', 6.0, 0.5, 20.0,'Search radius = multiplier * range * angular resolution. Too small deletes ''the floor, because floor points are spread out along the ground.'),
    ('dror_min_radius', 0.08, 0.01, 1.0, 'Smallest allowed search radius (m).'),
    ('dror_min_neighbors', 3, 1, 20,'A point needs at least this many neighbours inside its radius.'),

    # 5 ---- edge filter
    ('use_edge', True, None, None, 'Stage 5 on/off.'),
    ('edge_mode', 'mixed', None, None,
     "mixed = remove points with a nearer AND a farther neighbour along their beam. "
     "both = remove either side of a depth jump. "
     "behind = remove points sitting right behind a nearer surface."),
    ('edge_neighbors', 8, 2, 30, 'Neighbours (by direction) checked for each point.'),
    ('edge_radius_deg', 1.5, 0.2, 5.0, 'How close in angle a neighbour must be (degrees).'),
    ('edge_depth_jump', 0.3, 0.02, 3.0, 'Range gap that counts as a jump (m).'),
    ('edge_angle_deg', 3.0, 0.5, 30.0,
     'Jump only counts if the gap lies within this angle of the beam (degrees). '
     'Larger removes more, including far floor.'),
]

STAGE_NAMES = ['range', 'intensity', 'sor', 'dror', 'edge']
_DTYPES = {1: 'i1', 2: 'u1', 3: '<i2', 4: '<u2', 5: '<i4', 6: '<u4', 7: '<f4', 8: '<f8'}


# ------------------------- message <-> numpy helpers -------------------------
def read_cloud(msg):
    """PointCloud2 -> (xyz float32 (N,3), intensity float32 (N,) or None)."""
    fields = {f.name: f for f in msg.fields}
    if not all(k in fields for k in 'xyz'):
        return None, None
    if any(fields[k].datatype != 7 for k in 'xyz'):
        return None, None
    n = msg.width * msg.height
    names, formats = ['x', 'y', 'z'], ['<f4'] * 3
    offsets = [fields[k].offset for k in 'xyz']
    has_i = 'intensity' in fields
    if has_i:
        names.append('i')
        formats.append(_DTYPES.get(fields['intensity'].datatype, '<f4'))
        offsets.append(fields['intensity'].offset)
    dt = np.dtype({'names': names, 'formats': formats, 'offsets': offsets,
                   'itemsize': msg.point_step})
    arr = np.frombuffer(bytes(msg.data), dtype=dt, count=n)
    xyz = np.stack([arr['x'], arr['y'], arr['z']], axis=1)
    inten = arr['i'].astype(np.float32) if has_i else None
    ok = np.isfinite(xyz).all(axis=1)
    return xyz[ok], (inten[ok] if has_i else None)


def make_cloud(header, xyz, values):
    """Build a PointCloud2 with fields x, y, z, intensity (all float32)."""
    arr = np.empty((len(xyz), 4), np.float32)
    arr[:, :3] = xyz
    arr[:, 3] = values
    m = PointCloud2()
    m.header = header
    m.height = 1
    m.width = len(xyz)
    m.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                for i, n in enumerate(('x', 'y', 'z', 'intensity'))]
    m.is_bigendian = False
    m.point_step = 16
    m.row_step = 16 * len(xyz)
    m.is_dense = True
    m.data = arr.tobytes()
    return m


# ------------------------------- the 5 stages -------------------------------
# Each stage takes the points that are still alive and returns a True/False
# array: True = keep the point.

def stage_range(xyz, p):
    """1. Keep points between range_min and range_max (distance from the sensor)."""
    r = np.linalg.norm(xyz, axis=1)
    return (r >= p['range_min']) & (r <= p['range_max'])


def stage_intensity(inten, p):
    """2. Keep points whose reflectivity is inside [intensity_min, intensity_max]."""
    return (inten >= p['intensity_min']) & (inten <= p['intensity_max'])


def stage_sor(xyz, p):
    """3. Statistical outlier removal.
    For every point, average the distance to its k nearest neighbours. Points whose
    average is much larger than the typical average (mean + ratio * std) are isolated."""
    n, k = len(xyz), int(p['sor_k'])
    if n <= k + 1:
        return np.ones(n, bool)
    dist, _ = cKDTree(xyz).query(xyz, k=k + 1, workers=-1)
    mean_d = dist[:, 1:].mean(axis=1)             # column 0 is the point itself
    cutoff = mean_d.mean() + p['sor_std_ratio'] * mean_d.std()
    return mean_d <= cutoff


def stage_dror(xyz, p):
    """4. Dynamic radius outlier removal.
    Far points are naturally sparser, so the search radius grows with range:
    radius = max(min_radius, multiplier * range * angular_resolution).
    A point is kept if it has enough neighbours inside its own radius.
    Caution: the formula assumes a surface facing the sensor. The floor is seen at a
    shallow angle, so its points are spread out more than that, and a small multiplier
    deletes it. That is why the default multiplier is fairly large."""
    if len(xyz) == 0:
        return np.ones(0, bool)
    r = np.linalg.norm(xyz, axis=1)
    radius = np.maximum(p['dror_min_radius'],
                        p['dror_multiplier'] * r * np.radians(p['dror_angular_res_deg']))
    counts = cKDTree(xyz).query_ball_point(xyz, radius, return_length=True, workers=-1)
    return (counts - 1) >= p['dror_min_neighbors']   # minus the point itself


def stage_edge(xyz, p):
    """5. Edge filter (for unorganised clouds like the Mid-360S).
    Find each point's neighbours by viewing direction. If a neighbour's range differs by
    a lot AND the line to it runs almost along the laser beam, the point sits on a depth
    jump (a 'flying' point between a near object and the wall behind it)."""
    n, k = len(xyz), int(p['edge_neighbors'])
    keep = np.ones(n, bool)
    if n <= k + 1:
        return keep
    r = np.maximum(np.linalg.norm(xyz, axis=1), 1e-6)
    dirs = xyz / r[:, None]
    chord = 2.0 * np.sin(np.radians(p['edge_radius_deg']) / 2.0)
    dist, idx = cKDTree(dirs).query(dirs, k=k + 1, distance_upper_bound=chord, workers=-1)
    idx = idx[:, 1:]                                  # drop the point itself
    valid = idx < n                                   # missing neighbours have idx == n
    idx = np.where(valid, idx, 0)

    vec = xyz[idx] - xyz[:, None, :]                  # point -> neighbour, shape (n, k, 3)
    length = np.maximum(np.linalg.norm(vec, axis=2), 1e-9)
    along_beam = np.abs(np.einsum('nkj,nj->nk', vec, dirs)) / length   # 1.0 = exactly along beam
    d_range = r[idx] - r[:, None]

    jump = (valid & (np.abs(d_range) > p['edge_depth_jump'])
            & (along_beam > np.cos(np.radians(p['edge_angle_deg']))))
    nearer = (jump & (d_range < 0)).any(axis=1)       # a nearer neighbour on my beam
    farther = (jump & (d_range > 0)).any(axis=1)      # a farther neighbour on my beam

    mode = p['edge_mode']
    if mode == 'both':
        bad = nearer | farther
    elif mode == 'behind':
        bad = nearer
    else:                                             # 'mixed'
        bad = nearer & farther
    return ~bad


def run_filters(xyz, inten, p):
    """Run all enabled stages in order.
    Returns (indices of kept points, per-point stage that removed it, report)."""
    removed_by = np.zeros(len(xyz), np.float32)
    state = {'alive': np.arange(len(xyz))}
    report = []                                       # (name, removed count or None, ms)

    def stage(code, enabled, fn):
        name = STAGE_NAMES[code - 1]
        alive = state['alive']
        if not enabled:
            report.append((name, None, 0.0))
            return
        t0 = time.perf_counter()
        keep = fn(alive) if len(alive) else np.ones(0, bool)
        removed_by[alive[~keep]] = code
        state['alive'] = alive[keep]
        report.append((name, int((~keep).sum()), (time.perf_counter() - t0) * 1000.0))

    stage(1, p['use_range'], lambda a: stage_range(xyz[a], p))
    stage(2, p['use_intensity'] and inten is not None,
          lambda a: stage_intensity(inten[a], p))
    stage(3, p['use_sor'], lambda a: stage_sor(xyz[a], p))
    stage(4, p['use_dror'], lambda a: stage_dror(xyz[a], p))
    stage(5, p['use_edge'], lambda a: stage_edge(xyz[a], p))
    return state['alive'], removed_by, report


# ------------------------------- the ROS node -------------------------------
class LidarFilter(Node):
    def __init__(self):
        super().__init__('lidar_filter')
        for name, default, lo, hi, desc in PARAMS:
            self.declare(name, default, lo, hi, desc)
        self.add_on_set_parameters_callback(self.on_params)

        topic = self.get_parameter('input_topic').value
        self.create_subscription(PointCloud2, topic, self.on_cloud, qos_profile_sensor_data)
        self.pub_out = self.create_publisher(PointCloud2, '/filter/output', 2)
        self.pub_removed = self.create_publisher(PointCloud2, '/filter/removed', 2)
        self.frames = 0
        self.warned_intensity = False
        self.get_logger().info('Filtering %s. Change settings with: '
                               'ros2 param set /lidar_filter <name> <value>' % topic)

    def declare(self, name, default, lo, hi, desc):
        floats, ints = [], []
        if lo is not None and isinstance(default, float):
            floats = [FloatingPointRange(from_value=float(lo), to_value=float(hi), step=0.0)]
        elif lo is not None and isinstance(default, int) and not isinstance(default, bool):
            ints = [IntegerRange(from_value=int(lo), to_value=int(hi), step=1)]
        self.declare_parameter(name, default, ParameterDescriptor(
            description=desc, floating_point_range=floats, integer_range=ints))

    def on_params(self, params):
        """Reject settings that make no sense (runs on every ros2 param set)."""
        new = {p.name: p.value for p in params}

        def val(name):
            return new[name] if name in new else self.get_parameter(name).value

        if val('range_min') >= val('range_max'):
            return SetParametersResult(successful=False,
                                       reason='range_min must be smaller than range_max')
        if val('intensity_min') > val('intensity_max'):
            return SetParametersResult(successful=False,
                                       reason='intensity_min must not exceed intensity_max')
        if val('edge_mode') not in EDGE_MODES:
            return SetParametersResult(successful=False,
                                       reason='edge_mode must be one of %s' % (EDGE_MODES,))
        return SetParametersResult(successful=True)

    def on_cloud(self, msg):
        t0 = time.perf_counter()
        xyz, inten = read_cloud(msg)
        if xyz is None:
            self.get_logger().error('Cloud needs float32 x, y, z fields', once=True)
            return
        # Read the settings fresh every frame, so changes apply immediately.
        p = {name: self.get_parameter(name).value for name, *_ in PARAMS}
        if inten is None and p['use_intensity'] and not self.warned_intensity:
            self.get_logger().warn('Cloud has no intensity field: stage 2 is skipped.')
            self.warned_intensity = True

        kept, removed_by, report = run_filters(xyz, inten, p)
        mask = np.zeros(len(xyz), bool)
        mask[kept] = True
        self.pub_out.publish(make_cloud(
            msg.header, xyz[mask], inten[mask] if inten is not None else 0.0))
        self.pub_removed.publish(make_cloud(msg.header, xyz[~mask], removed_by[~mask]))

        self.frames += 1
        if self.frames % int(p['log_every']) == 1:
            parts = ['%s %s' % (n, 'off' if c is None else '-%d (%.0fms)' % (c, ms))
                     for n, c, ms in report]
            self.get_logger().info('raw %d | %s | kept %d | %.0f ms total' % (
                len(xyz), ' | '.join(parts), len(kept), (time.perf_counter() - t0) * 1000))


def main():
    rclpy.init()
    node = LidarFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
