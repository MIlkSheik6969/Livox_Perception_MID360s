#!/usr/bin/env python3
"""
Classical obstacle detection for a Livox Mid-360S (no neural network).

Pipeline, every frame:
  1. crop by range, remove points on the robot's own body, downsample
  2. remove the ground (RANSAC plane fit)
  3. cluster what is left (points close together = one object)
  4. fit an oriented box around each cluster

Each stage is published as its own topic so you can look at it in RViz:
  /obstacles/filtered    cloud after step 1
  /obstacles/ground      points treated as ground (step 2)
  /obstacles/non_ground  everything else (step 2)
  /obstacles/boxes       green wireframe boxes (steps 3 + 4)
"""
import time
from collections import deque

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray

# ======================= SETTINGS YOU CAN CHANGE =======================
IN_TOPIC = '/livox/lidar'

MIN_RANGE = 0.3          # metres: ignore points closer than this
MAX_RANGE = 15.0         # metres: ignore points further than this
# Half-size (x, y, z) in metres of a box around the sensor that covers the
# robot's own body. Points inside it are thrown away. Tune this to your robot.
BODY_BOX = (0.45, 0.30, 0.30)
VOXEL = 0.08             # metres: downsample grid size
ACCUM_FRAMES = 2         # frames merged together (use 1 while the robot walks)

GROUND_DIST = 0.08       # metres: points this close to the ground plane = ground
MAX_TILT_DEG = 25.0      # the ground plane may tilt this much from level

CLUSTER_EPS = 0.25       # metres: points closer than this join the same object
MIN_POINTS = 8           # clusters with fewer points are ignored (noise)
MAX_SIZE = 8.0           # metres: clusters bigger than this (walls) are ignored
# =======================================================================


# ---------------------- message <-> numpy helpers ----------------------
def cloud_to_xyz(msg):
    """PointCloud2 -> (N, 3) float32 array. Reads x, y, z by field name."""
    offs = {f.name: f.offset for f in msg.fields}
    if not all(k in offs for k in ('x', 'y', 'z')):
        return None
    n = msg.width * msg.height
    if n == 0:
        return np.zeros((0, 3), np.float32)
    dt = np.dtype({'names': ['x', 'y', 'z'],
                   'formats': ['<f4', '<f4', '<f4'],
                   'offsets': [offs['x'], offs['y'], offs['z']],
                   'itemsize': msg.point_step})
    arr = np.frombuffer(bytes(msg.data), dtype=dt, count=n)
    xyz = np.stack([arr['x'], arr['y'], arr['z']], axis=1)
    return xyz[np.isfinite(xyz).all(axis=1)]


def xyz_to_cloud(xyz, header):
    m = PointCloud2()
    m.header = header
    m.height = 1
    m.width = len(xyz)
    m.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                for i, n in enumerate('xyz')]
    m.is_bigendian = False
    m.point_step = 12
    m.row_step = 12 * len(xyz)
    m.is_dense = True
    m.data = np.ascontiguousarray(xyz, dtype=np.float32).tobytes()
    return m


# ------------------------- step 1: preprocessing -------------------------
def voxel_downsample(xyz, voxel):
    """Keep one point per voxel-sized cube."""
    if len(xyz) == 0:
        return xyz
    idx = np.floor(xyz / voxel).astype(np.int64)
    idx -= idx.min(axis=0)
    dims = idx.max(axis=0) + 1
    key = (idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]
    _, first = np.unique(key, return_index=True)
    return xyz[first]


def preprocess(xyz):
    r = np.linalg.norm(xyz, axis=1)
    keep = (r > MIN_RANGE) & (r < MAX_RANGE)
    bx, by, bz = BODY_BOX
    on_body = ((np.abs(xyz[:, 0]) < bx) & (np.abs(xyz[:, 1]) < by)
               & (np.abs(xyz[:, 2]) < bz))
    return voxel_downsample(xyz[keep & ~on_body], VOXEL)


# ------------------------- step 2: ground removal -------------------------
def remove_ground(xyz, rng):
    """Fit one plane to the lowest points. Returns (ground, non_ground)."""
    empty = xyz[:0]
    if len(xyz) < 50:
        return empty, xyz
    z_low = np.percentile(xyz[:, 2], 2)
    cand = xyz[xyz[:, 2] < z_low + 0.4]      # only the lowest points can be floor
    if len(cand) < 30:
        return empty, xyz

    cos_tilt = np.cos(np.radians(MAX_TILT_DEG))
    best_count, best = 0, None
    for _ in range(100):
        p = cand[rng.choice(len(cand), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        norm = np.linalg.norm(n)
        if norm < 1e-6:
            continue
        n = n / norm
        if abs(n[2]) < cos_tilt:             # reject walls and steep planes
            continue
        if n[2] < 0:
            n = -n
        d = -float(n @ p[0])
        count = int(np.sum(np.abs(cand @ n + d) < GROUND_DIST))
        if count > best_count:
            best_count, best = count, (n, d)

    if best is None:
        return empty, xyz
    n, d = best
    is_ground = (xyz @ n + d) < GROUND_DIST
    return xyz[is_ground], xyz[~is_ground]


# ---------------------------- step 3: clustering ----------------------------
def cluster(xyz):
    """Group points that are within CLUSTER_EPS of each other."""
    n = len(xyz)
    if n < MIN_POINTS:
        return []
    pairs = cKDTree(xyz).query_pairs(CLUSTER_EPS, output_type='ndarray')
    if len(pairs) == 0:
        return []
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    order = np.argsort(labels)
    labels_sorted = labels[order]
    splits = np.flatnonzero(np.diff(labels_sorted)) + 1
    groups = np.split(order, splits)
    return [xyz[g] for g in groups if len(g) >= MIN_POINTS]


# ---------------------------- step 4: box fitting ----------------------------
def fit_box(pts):
    """Oriented box: returns (center xyz, size xyz, yaw in radians)."""
    xy = pts[:, :2]
    c = xy.mean(axis=0)
    cov = np.cov((xy - c).T)
    w, v = np.linalg.eigh(cov)
    major = v[:, np.argmax(w)]
    yaw = float(np.arctan2(major[1], major[0]))
    u = np.array([np.cos(yaw), np.sin(yaw)])
    s = np.array([-np.sin(yaw), np.cos(yaw)])
    lu, ls = (xy - c) @ u, (xy - c) @ s
    umin, umax, smin, smax = lu.min(), lu.max(), ls.min(), ls.max()
    center_xy = c + u * (umin + umax) / 2 + s * (smin + smax) / 2
    zmin, zmax = pts[:, 2].min(), pts[:, 2].max()
    size = np.array([max(umax - umin, 0.1), max(smax - smin, 0.1), max(zmax - zmin, 0.1)])
    center = np.array([center_xy[0], center_xy[1], (zmin + zmax) / 2])
    return center, size, yaw


def box_corners(center, size, yaw):
    sx, sy, sz = size / 2
    corners = np.array([[sx, sy, sz], [sx, -sy, sz], [-sx, -sy, sz], [-sx, sy, sz],
                        [sx, sy, -sz], [sx, -sy, -sz], [-sx, -sy, -sz], [-sx, sy, -sz]])
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return corners @ rot.T + center


EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]


def detect(xyz_new, buffer, rng):
    """Run the whole pipeline on one frame. Returns everything for publishing."""
    filtered_frame = preprocess(xyz_new)
    buffer.append(filtered_frame)
    filtered = voxel_downsample(np.vstack(list(buffer)), VOXEL)
    ground, non_ground = remove_ground(filtered, rng)
    boxes = []
    for pts in cluster(non_ground):
        center, size, yaw = fit_box(pts)
        if size[:2].max() <= MAX_SIZE:
            boxes.append((center, size, yaw, len(pts)))
    return filtered, ground, non_ground, boxes


# ------------------------------- the ROS node -------------------------------
class ObstacleDetector(Node):
    def __init__(self):
        super().__init__('obstacle_detector')
        self.create_subscription(PointCloud2, IN_TOPIC, self.on_cloud,
                                 qos_profile_sensor_data)
        self.pub_filtered = self.create_publisher(PointCloud2, '/obstacles/filtered', 2)
        self.pub_ground = self.create_publisher(PointCloud2, '/obstacles/ground', 2)
        self.pub_non_ground = self.create_publisher(PointCloud2, '/obstacles/non_ground', 2)
        self.pub_boxes = self.create_publisher(MarkerArray, '/obstacles/boxes', 2)
        self.buffer = deque(maxlen=ACCUM_FRAMES)
        self.rng = np.random.default_rng(0)
        self.frames = 0
        self.get_logger().info('Waiting for point clouds on %s' % IN_TOPIC)

    def on_cloud(self, msg):
        t0 = time.time()
        xyz = cloud_to_xyz(msg)
        if xyz is None:
            self.get_logger().error('Cloud has no x/y/z fields')
            return
        filtered, ground, non_ground, boxes = detect(xyz, self.buffer, self.rng)

        self.pub_filtered.publish(xyz_to_cloud(filtered, msg.header))
        self.pub_ground.publish(xyz_to_cloud(ground, msg.header))
        self.pub_non_ground.publish(xyz_to_cloud(non_ground, msg.header))
        self.pub_boxes.publish(self.make_markers(boxes, msg.header))

        self.frames += 1
        if self.frames % 30 == 1:
            self.get_logger().info(
                'in=%d  filtered=%d  ground=%d  other=%d  boxes=%d  (%.0f ms)' % (
                    len(xyz), len(filtered), len(ground), len(non_ground),
                    len(boxes), (time.time() - t0) * 1000))

    @staticmethod
    def make_markers(boxes, header):
        m = Marker()
        m.header = header
        m.ns = 'obstacle_boxes'
        m.id = 0
        m.type = Marker.LINE_LIST
        m.action = Marker.ADD
        m.scale.x = 0.03
        m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 0.0, 1.0
        m.pose.orientation.w = 1.0
        m.lifetime.sec = 0
        m.lifetime.nanosec = 300_000_000
        for center, size, yaw, _ in boxes:
            corners = box_corners(center, size, yaw)
            for a, b in EDGES:
                m.points.append(Point(x=float(corners[a][0]), y=float(corners[a][1]),
                                      z=float(corners[a][2])))
                m.points.append(Point(x=float(corners[b][0]), y=float(corners[b][1]),
                                      z=float(corners[b][2])))
        arr = MarkerArray()
        arr.markers.append(m)
        return arr


def main():
    rclpy.init()
    node = ObstacleDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
