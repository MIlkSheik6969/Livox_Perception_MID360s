#!/usr/bin/env python3
"""Runtime-configurable point-cloud filters for Livox MID-360S."""

import time
import numpy as np
from scipy.spatial import cKDTree

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.msg import (
    FloatingPointRange, IntegerRange, ParameterDescriptor,
    SetParametersResult,
)
from sensor_msgs.msg import PointCloud2, PointField


# Parameter defaults and allowed ranges.
PARAMS = {
    "input_topic": ("/livox/lidar", None, None),
    "log_every": (10, 1, 1000),

    "use_range": (True, None, None),
    "range_min": (0.3, 0.0, 50.0),
    "range_max": (15.0, 0.5, 100.0),

    "use_intensity": (True, None, None),
    "intensity_min": (0.0, 0.0, 255.0),
    "intensity_max": (255.0, 0.0, 255.0),

    "use_sor": (True, None, None),
    "sor_k": (10, 1, 50),
    "sor_std_ratio": (2.0, 0.1, 5.0),

    "use_dror": (True, None, None),
    "dror_angular_res_deg": (1.0, 0.05, 5.0),
    "dror_multiplier": (6.0, 0.5, 20.0),
    "dror_min_radius": (0.08, 0.01, 1.0),
    "dror_min_neighbors": (3, 1, 20),
    
    "use_edge": (True, None, None),
    "edge_mode": ("mixed", None, None),
    "edge_neighbors": (8, 2, 30),
    "edge_radius_deg": (1.5, 0.2, 5.0),
    "edge_depth_jump": (0.3, 0.02, 3.0),
    "edge_angle_deg": (3.0, 0.5, 30.0),
}

INTENSITY_TYPES = {
    1: "i1", 2: "u1", 3: "<i2", 4: "<u2",
    5: "<i4", 6: "<u4", 7: "<f4", 8: "<f8",
}
STAGES = ("range", "intensity", "sor", "dror", "edge")
EDGE_MODES = ("mixed", "both", "behind")


def read_cloud(msg):
    """Read float32 XYZ and optional intensity from PointCloud2."""
    fields = {field.name: field for field in msg.fields}
    if any(name not in fields or fields[name].datatype != PointField.FLOAT32
           for name in ("x", "y", "z")):
        return None, None

    names = ["x", "y", "z"]
    formats = ["<f4"] * 3
    offsets = [fields[name].offset for name in names]

    if "intensity" in fields:
        names.append("intensity")
        formats.append(INTENSITY_TYPES.get(fields["intensity"].datatype, "<f4"))
        offsets.append(fields["intensity"].offset)

    dtype = np.dtype({
        "names": names, "formats": formats,
        "offsets": offsets, "itemsize": msg.point_step,
    })
    data = np.frombuffer(bytes(msg.data), dtype=dtype,
                         count=msg.width * msg.height)
    xyz = np.column_stack((data["x"], data["y"], data["z"]))
    intensity = data["intensity"].astype(np.float32) if "intensity" in names else None

    valid = np.isfinite(xyz).all(axis=1)
    return xyz[valid], intensity[valid] if intensity is not None else None


def make_cloud(header, xyz, intensity):
    """Create an XYZ-intensity PointCloud2; all output fields are float32."""
    values = np.zeros(len(xyz), dtype=np.float32) if intensity is None else intensity
    points = np.column_stack((xyz, values)).astype(np.float32, copy=False)

    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = len(points)
    msg.fields = [
        PointField(name=name, offset=4 * i,
                   datatype=PointField.FLOAT32, count=1)
        for i, name in enumerate(("x", "y", "z", "intensity"))
    ]
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = 16 * len(points)
    msg.is_dense = True
    msg.data = points.tobytes()
    return msg


def filter_range(xyz, p):
    distance = np.linalg.norm(xyz, axis=1)
    return (distance >= p["range_min"]) & (distance <= p["range_max"])


def filter_intensity(intensity, p):
    return ((intensity >= p["intensity_min"]) &
            (intensity <= p["intensity_max"]))


def filter_sor(xyz, p):
    """Statistical outlier removal using mean k-neighbour distance."""
    k = int(p["sor_k"])
    if len(xyz) <= k + 1:
        return np.ones(len(xyz), dtype=bool)

    distances, _ = cKDTree(xyz).query(xyz, k=k + 1, workers=-1)
    mean_distance = distances[:, 1:].mean(axis=1)
    limit = mean_distance.mean() + p["sor_std_ratio"] * mean_distance.std()
    return mean_distance <= limit


def filter_dror(xyz, p):
    """Dynamic radius outlier removal; search radius increases with range."""
    if len(xyz) == 0:
        return np.ones(0, dtype=bool)

    distance = np.linalg.norm(xyz, axis=1)
    radius = np.maximum(
        p["dror_min_radius"],
        p["dror_multiplier"] * distance *
        np.radians(p["dror_angular_res_deg"]),
    )
    neighbours = cKDTree(xyz).query_ball_point(
        xyz, radius, return_length=True, workers=-1
    )
    return neighbours - 1 >= p["dror_min_neighbors"]


def filter_edges(xyz, p):
    """Remove points around large depth jumps along similar viewing rays."""
    count = len(xyz)
    k = int(p["edge_neighbors"])
    if count <= k + 1:
        return np.ones(count, dtype=bool)

    distance = np.maximum(np.linalg.norm(xyz, axis=1), 1e-6)
    directions = xyz / distance[:, None]
    chord = 2 * np.sin(np.radians(p["edge_radius_deg"]) / 2)

    _, indices = cKDTree(directions).query(
        directions, k=k + 1, distance_upper_bound=chord, workers=-1
    )
    indices = indices[:, 1:]
    valid = indices < count
    indices = np.where(valid, indices, 0)

    vectors = xyz[indices] - xyz[:, None, :]
    lengths = np.maximum(np.linalg.norm(vectors, axis=2), 1e-9)
    along_ray = np.abs(np.einsum("nkj,nj->nk", vectors, directions)) / lengths
    depth_difference = distance[indices] - distance[:, None]

    jump = (
        valid &
        (np.abs(depth_difference) > p["edge_depth_jump"]) &
        (along_ray > np.cos(np.radians(p["edge_angle_deg"])))
    )
    nearer = (jump & (depth_difference < 0)).any(axis=1)
    farther = (jump & (depth_difference > 0)).any(axis=1)

    if p["edge_mode"] == "both":
        remove = nearer | farther
    elif p["edge_mode"] == "behind":
        remove = nearer
    else:  # mixed
        remove = nearer & farther

    return ~remove


def run_filters(xyz, intensity, p):
    """Run enabled filters in the original order; return kept and removed stages."""
    alive = np.arange(len(xyz))
    removed_stage = np.zeros(len(xyz), dtype=np.float32)
    report = []

    filters = [
        ("range", p["use_range"], lambda x: filter_range(x, p)),
        ("intensity", p["use_intensity"] and intensity is not None,
         lambda x: filter_intensity(intensity[x], p)),
        ("sor", p["use_sor"], lambda x: filter_sor(x, p)),
        ("dror", p["use_dror"], lambda x: filter_dror(x, p)),
        ("edge", p["use_edge"], lambda x: filter_edges(x, p)),
    ]

    for stage_number, (name, enabled, function) in enumerate(filters, start=1):
        if not enabled:
            report.append((name, None, 0.0))
            continue

        start = time.perf_counter()
        keep = function(xyz[alive]) if name != "intensity" else function(alive)
        removed_stage[alive[~keep]] = stage_number
        alive = alive[keep]
        report.append((
            name, int((~keep).sum()),
            (time.perf_counter() - start) * 1000,
        ))

    return alive, removed_stage, report


class LidarFilter(Node):
    def __init__(self):
        super().__init__("lidar_filter")

        for name, (default, low, high) in PARAMS.items():
            ranges = {}
            if low is not None:
                if isinstance(default, float):
                    ranges["floating_point_range"] = [
                        FloatingPointRange(from_value=float(low),
                                           to_value=float(high), step=0.0)
                    ]
                elif isinstance(default, int) and not isinstance(default, bool):
                    ranges["integer_range"] = [
                        IntegerRange(from_value=int(low), to_value=int(high), step=1)
                    ]
            self.declare_parameter(
                name, default,
                ParameterDescriptor(**ranges)
            )

        self.add_on_set_parameters_callback(self.validate_parameters)

        topic = self.get_parameter("input_topic").value
        self.create_subscription(
            PointCloud2, topic, self.on_cloud, qos_profile_sensor_data
        )
        self.output_pub = self.create_publisher(PointCloud2, "/filter/output", 2)
        self.removed_pub = self.create_publisher(PointCloud2, "/filter/removed", 2)

        self.frames = 0
        self.warned_no_intensity = False
        self.get_logger().info(f"Filtering {topic}")

    def validate_parameters(self, parameters):
        values = {param.name: param.value for param in parameters}

        def get(name):
            return values.get(name, self.get_parameter(name).value)

        if get("range_min") >= get("range_max"):
            return SetParametersResult(
                successful=False, reason="range_min must be smaller than range_max"
            )
        if get("intensity_min") > get("intensity_max"):
            return SetParametersResult(
                successful=False, reason="intensity_min must not exceed intensity_max"
            )
        if get("edge_mode") not in EDGE_MODES:
            return SetParametersResult(
                successful=False, reason=f"edge_mode must be one of {EDGE_MODES}"
            )
        return SetParametersResult(successful=True)

    def on_cloud(self, msg):
        start = time.perf_counter()
        xyz, intensity = read_cloud(msg)

        if xyz is None:
            self.get_logger().error("PointCloud2 must contain float32 x, y, z fields")
            return

        p = {name: self.get_parameter(name).value for name in PARAMS}

        if intensity is None and p["use_intensity"] and not self.warned_no_intensity:
            self.get_logger().warning("No intensity field; intensity filter skipped.")
            self.warned_no_intensity = True

        kept, removed_stage, report = run_filters(xyz, intensity, p)
        keep_mask = np.zeros(len(xyz), dtype=bool)
        keep_mask[kept] = True

        kept_intensity = intensity[keep_mask] if intensity is not None else None
        self.output_pub.publish(make_cloud(msg.header, xyz[keep_mask], kept_intensity))
        self.removed_pub.publish(
            make_cloud(msg.header, xyz[~keep_mask], removed_stage[~keep_mask])
        )

        self.frames += 1
        if self.frames % int(p["log_every"]) == 1:
            details = [
                f"{name} {'off' if removed is None else f'-{removed} ({ms:.0f}ms)'}"
                for name, removed, ms in report
            ]
            total_ms = (time.perf_counter() - start) * 1000
            self.get_logger().info(
                f"raw {len(xyz)} | {' | '.join(details)} | "
                f"kept {len(kept)} | {total_ms:.0f}ms total"
            )


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


if __name__ == "__main__":
    main()
