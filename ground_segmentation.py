import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
import sensor_msgs_py.point_cloud2 as pc2
import message_filters
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


def gravity_align(pts, odom_msg):
    """Rotate body-frame points so gravity points along -z."""
    q = odom_msg.pose.pose.orientation
    R_wb = R.from_quat([q.x, q.y, q.z, q.w])        # body in camera_init
    g_body = R_wb.inv().apply([0, 0, -1])           # gravity dir in body frame
    v = np.cross(g_body, [0, 0, -1])
    s, c = np.linalg.norm(v), np.dot(g_body, [0, 0, -1])
    if s < 1e-6:
        return pts
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    R_align = np.eye(3) + K + K @ K * ((1 - c) / s**2)
    return pts @ R_align.T


class GroundSeg(Node):
    def __init__(self):
        super().__init__('ground_seg')

        # Parameters
        self.declare_parameter('sensor_h_min', 0.1)    # ground must be at least this far below sensor (m)
        self.declare_parameter('sensor_h_max', 1.0)    # ground can't be farther below than this (m)
        self.declare_parameter('max_tilt_deg', 25.0)   # max plane tilt from horizontal
        self.declare_parameter('ransac_thresh', 0.05)  # RANSAC inlier distance (m)
        self.declare_parameter('label_thresh', 0.08)   # final ground labeling distance (m)
        self.declare_parameter('ransac_iters', 200)

        # Subscribers (synchronized)
        cloud_sub = message_filters.Subscriber(self, PointCloud2, '/cloud_registered_body')
        odom_sub = message_filters.Subscriber(self, Odometry, '/Odometry')
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [cloud_sub, odom_sub], queue_size=10, slop=0.05)
        self.sync.registerCallback(self.cb)

        # Publishers
        self.pub_ground = self.create_publisher(PointCloud2, '/ground', 10)
        self.pub_obst = self.create_publisher(PointCloud2, '/obstacles', 10)

    def cb(self, cloud_msg, odom_msg):
        h_min = self.get_parameter('sensor_h_min').value
        h_max = self.get_parameter('sensor_h_max').value
        max_tilt = self.get_parameter('max_tilt_deg').value
        ransac_thresh = self.get_parameter('ransac_thresh').value
        label_thresh = self.get_parameter('label_thresh').value
        iters = self.get_parameter('ransac_iters').value

        pts = np.array([[p[0], p[1], p[2]] for p in
                        pc2.read_points(cloud_msg, field_names=('x', 'y', 'z'),
                                        skip_nans=True)], dtype=np.float32)
        if len(pts) < 100:
            return

        # 1. Gravity-align: z = 0 is sensor height, ground is at negative z
        pts = gravity_align(pts, odom_msg).astype(np.float32)

        # 2. Candidate points: below the sensor, within the height band
        cand = pts[(pts[:, 2] < -h_min) & (pts[:, 2] > -h_max)]
        if len(cand) < 100:
            self.publish(cloud_msg.header, np.empty((0, 3), np.float32), pts)
            return

        # 3. RANSAC on candidates only
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cand)
        plane, _ = pcd.segment_plane(distance_threshold=ransac_thresh,
                                     ransac_n=3, num_iterations=iters)
        a, b, c, d = plane
        n = np.array([a, b, c])
        n_norm = np.linalg.norm(n)

        # 4. Reject planes that are not roughly horizontal
        tilt = np.degrees(np.arccos(abs(n[2]) / n_norm))
        if tilt > max_tilt:
            self.publish(cloud_msg.header, np.empty((0, 3), np.float32), pts)
            return

        # 5. Label all points by distance to the fitted plane
        dist = np.abs(pts @ n + d) / n_norm
        mask = (dist < label_thresh) & (pts[:, 2] < -h_min)

        self.publish(cloud_msg.header, pts[mask], pts[~mask])

    def publish(self, header, ground, obstacles):
        self.pub_ground.publish(pc2.create_cloud_xyz32(header, ground))
        self.pub_obst.publish(pc2.create_cloud_xyz32(header, obstacles))


def main():
    rclpy.init()
    node = GroundSeg()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()