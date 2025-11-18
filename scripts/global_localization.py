#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import copy
import time
from typing import Tuple

import numpy as np
import open3d as o3d

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Point, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

# Prefer tf_transformations (pip install tf-transformations) in ROS 2
try:
    import tf_transformations as tft
except Exception:
    # Fallback small helpers
    tft = None


def pose_to_mat_from_pose_with_cov(pose_msg: PoseWithCovarianceStamped) -> np.ndarray:
    p = pose_msg.pose.pose.position
    q = pose_msg.pose.pose.orientation
    return xyz_quat_to_mat44(p.x, p.y, p.z, q.x, q.y, q.z, q.w)


def pose_to_mat_from_odom(odom_msg: Odometry) -> np.ndarray:
    p = odom_msg.pose.pose.position
    q = odom_msg.pose.pose.orientation
    return xyz_quat_to_mat44(p.x, p.y, p.z, q.x, q.y, q.z, q.w)


def xyz_quat_to_mat44(x, y, z, qx, qy, qz, qw) -> np.ndarray:
    if tft is not None:
        M = tft.quaternion_matrix([qx, qy, qz, qw])
        M[0:3, 3] = [x, y, z]
        return M
    # Minimal pure-numpy fallback
    q = np.array([qx, qy, qz, qw], dtype=float)
    q = q / np.linalg.norm(q)
    x2, y2, z2 = q[0] + q[0], q[1] + q[1], q[2] + q[2]
    xx, yy, zz = q[0] * x2, q[1] * y2, q[2] * z2
    xy, xz, yz = q[0] * y2, q[0] * z2, q[1] * z2
    wx, wy, wz = q[3] * x2, q[3] * y2, q[3] * z2
    R = np.array([
        [1.0 - (yy + zz),       xy - wz,          xz + wy],
        [      xy + wz,   1.0 - (xx + zz),        yz - wx],
        [      xz - wy,         yz + wx,    1.0 - (xx + yy)]
    ])
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = [x, y, z]
    return M


def inverse_se3(T: np.ndarray) -> np.ndarray:
    Ti = np.eye(4)
    R = T[:3, :3]
    t = T[:3, 3]
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def voxel_down_sample(pcd: o3d.geometry.PointCloud, voxel_size: float) -> o3d.geometry.PointCloud:
    return pcd.voxel_down_sample(voxel_size)


def registration_at_scale(pc_scan: o3d.geometry.PointCloud,
                          pc_map: o3d.geometry.PointCloud,
                          initial: np.ndarray,
                          scale: float) -> Tuple[np.ndarray, float]:
    # Open3D >= 0.10 API
    reg = o3d.pipelines.registration.registration_icp(
        voxel_down_sample(pc_scan, FastLioLocalizationNode.SCAN_VOXEL_SIZE * scale),
        voxel_down_sample(pc_map, FastLioLocalizationNode.MAP_VOXEL_SIZE * scale),
        max_correspondence_distance=1.0 * scale,
        init=initial,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20)
    )
    return reg.transformation, reg.fitness


def points_np_to_pointcloud2(points: np.ndarray,
                             frame_id: str,
                             stamp_sec: float = None,
                             intensities: np.ndarray = None) -> PointCloud2:
    """
    points: Nx3 (xyz) or Nx4 (xyzi). If intensities is provided, it overrides the 4th column.
    """
    from sensor_msgs_py import point_cloud2 as pc2

    assert points.ndim == 2 and points.shape[1] >= 3
    has_i = (points.shape[1] >= 4) or (intensities is not None)

    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    point_step = 12

    if has_i:
        fields.append(PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1))
        point_step = 16

    header = Header()
    header.frame_id = frame_id

    # rclpy fills stamp automatically when publishing; if you want explicit:
    # leave as zero or set via Clock; keeping zero is fine for most viz/debug uses.

    if intensities is not None:
        data = np.zeros((points.shape[0], 4), dtype=np.float32)
        data[:, :3] = points[:, :3].astype(np.float32)
        data[:, 3] = intensities.astype(np.float32)
    elif has_i:
        data = points[:, :4].astype(np.float32)
    else:
        data = points[:, :3].astype(np.float32)

    # Build generator of tuples
    if has_i:
        gen = (tuple(p) for p in data)
    else:
        gen = (tuple(p[:3]) for p in data)

    cloud = pc2.create_cloud(header, fields, gen)
    return cloud


def pointcloud2_to_xyz_numpy(msg: PointCloud2) -> np.ndarray:
    """
    Robust ROS 2 PointCloud2 -> Nx3 float32 using sensor_msgs_py
    """
    from sensor_msgs_py import point_cloud2 as pc2
    pts = []
    for p in pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True):
        pts.append([p[0], p[1], p[2]])
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


class FastLioLocalizationNode(Node):
    """
    ROS2 refactor of the original ROS1 node:
      - subscribes to /map (PointCloud2), /cloud_registered (PointCloud2), /Odometry (Odometry)
      - publishes /cur_scan_in_map, /submap, /map_to_odom
      - runs periodic scan-to-map ICP to estimate T_map_to_odom
    """

    # Tunables (class-level for easy reuse in helpers)
    MAP_VOXEL_SIZE = 0.4
    SCAN_VOXEL_SIZE = 0.1
    FREQ_LOCALIZATION = 0.5  # Hz
    LOCALIZATION_TH = 0.95
    FOV = 2 * np.pi           # radians
    FOV_FAR = 150.0     # meters

    def __init__(self):
        super().__init__('fast_lio_localization')

        # Allow overriding via params
        self.declare_parameter('map_voxel', self.MAP_VOXEL_SIZE)
        self.declare_parameter('scan_voxel', self.SCAN_VOXEL_SIZE)
        self.declare_parameter('freq', self.FREQ_LOCALIZATION)
        self.declare_parameter('fov', self.FOV)
        self.declare_parameter('fov_far', self.FOV_FAR)
        self.declare_parameter('localization_th', self.LOCALIZATION_TH)

        self.MAP_VOXEL_SIZE = float(self.get_parameter('map_voxel').value)
        self.SCAN_VOXEL_SIZE = float(self.get_parameter('scan_voxel').value)
        self.FREQ_LOCALIZATION = float(self.get_parameter('freq').value)
        self.FOV = float(self.get_parameter('fov').value)
        self.FOV_FAR = float(self.get_parameter('fov_far').value)
        self.LOCALIZATION_TH = float(self.get_parameter('localization_th').value)

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.pub_pc_in_map = self.create_publisher(PointCloud2, '/cur_scan_in_map', 10)
        self.pub_submap = self.create_publisher(PointCloud2, '/submap', 10)
        self.pub_map_to_odom = self.create_publisher(Odometry, '/map_to_odom', 10)

        self.sub_scan = self.create_subscription(PointCloud2, '/cloud_registered', self._cb_save_cur_scan, qos)
        self.sub_odom = self.create_subscription(Odometry, '/Odometry', self._cb_save_cur_odom, qos)
        self.sub_map = self.create_subscription(PointCloud2, '/map', self._cb_init_global_map, qos)

        # State
        self.global_map: o3d.geometry.PointCloud | None = None
        self.initialized = False
        self.T_map_to_odom = np.eye(4, dtype=float)
        self.cur_odom: Odometry | None = None
        self.cur_scan: o3d.geometry.PointCloud | None = None

        # Timer for periodic localization
        if self.FREQ_LOCALIZATION > 0.0:
            self.timer = self.create_timer(1.0 / self.FREQ_LOCALIZATION, self._timer_localization)
        else:
            self.timer = None

        self.get_logger().info('Localization Node Inited (ROS 2)...')
        self.get_logger().warn('Waiting for global map and first scan / odometry...')

    # --- Callbacks ---

    def _cb_init_global_map(self, pc_msg: PointCloud2):
        if self.global_map is not None:
            return
        xyz = pointcloud2_to_xyz_numpy(pc_msg)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        self.global_map = voxel_down_sample(pcd, self.MAP_VOXEL_SIZE)
        self.get_logger().info(f'Global map received with {len(self.global_map.points)} pts (downsampled).')

    def _cb_save_cur_odom(self, odom_msg: Odometry):
        self.cur_odom = odom_msg

    def _cb_save_cur_scan(self, pc_msg: PointCloud2):
        # In the original, FAST-LIO scan is already in odom frame; keep behavior.
        # Republish with 'camera_init' frame, as before.
        try:
            xyz = pointcloud2_to_xyz_numpy(pc_msg)
        except Exception as e:
            self.get_logger().warn(f'PointCloud2 conversion failed: {e}')
            return

        # Publish passthrough (optional visualization)
        passthrough = points_np_to_pointcloud2(xyz, frame_id='camera_init')
        self.pub_pc_in_map.publish(passthrough)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz[:, :3])
        self.cur_scan = pcd

    # --- Periodic localization ---

    def _timer_localization(self):
        if self.global_map is None:
            return
        if self.cur_scan is None:
            return
        if self.cur_odom is None and not self.initialized:
            # We need at least an initial pose or odometry to proceed
            return

        if not self.initialized:
            # Wait for an initial pose on /initialpose (ROS 2 topic) — we’ll opportunistically consume it once
            # Instead of blocking wait, try to check if someone published recently by creating a one-shot subscription
            # Simpler approach: just do a first run using current T_map_to_odom (identity) as initial guess.
            success = self._global_localization(self.T_map_to_odom)
            if success:
                self.initialized = True
                self.get_logger().info('Initialize successfully!')
            return

        # Regular periodic relocalization using last T_map_to_odom as initial
        self._global_localization(self.T_map_to_odom)

    # --- Core logic ---

    def _crop_global_map_in_FOV(self,
                                pose_estimation: np.ndarray,
                                cur_odom: Odometry) -> o3d.geometry.PointCloud:
        # Pose of scan origin in odom
        T_odom_to_base_link = pose_to_mat_from_odom(cur_odom)
        T_map_to_base_link = pose_estimation @ T_odom_to_base_link
        T_base_link_to_map = inverse_se3(T_map_to_base_link)

        # Transform map points into base_link
        gm_map = np.asarray(self.global_map.points)
        homog = np.hstack([gm_map, np.ones((gm_map.shape[0], 1), dtype=gm_map.dtype)])
        gm_in_base = (T_base_link_to_map @ homog.T).T  # Nx4

        # Angular filter
        ang = np.arctan2(gm_in_base[:, 1], gm_in_base[:, 0])
        if self.FOV > math.pi:
            # wraparound lidar – only distance + angle cut
            mask = (np.abs(ang) < self.FOV / 2.0) & (gm_in_base[:, 0] < self.FOV_FAR)
        else:
            # forward looking lidar – keep front sector only (x>0)
            mask = (gm_in_base[:, 0] > 0.0) & (gm_in_base[:, 0] < self.FOV_FAR) & (np.abs(ang) < self.FOV / 2.0)

        sub = o3d.geometry.PointCloud()
        sub.points = o3d.utility.Vector3dVector(gm_map[mask, :3])

        # Publish a decimated submap for visualization (in 'map' frame)
        if sub.has_points():
            decim = np.asarray(sub.points)[::10]
            if decim.size > 0:
                self.pub_submap.publish(points_np_to_pointcloud2(decim, frame_id='map'))
        return sub

    def _global_localization(self, pose_estimation: np.ndarray) -> bool:
        # Open3D ICP on FOV-cropped map
        if self.global_map is None or self.cur_scan is None or self.cur_odom is None:
            return False

        self.get_logger().info('Global localization by scan-to-map matching...')
        scan_to_map = copy.copy(self.cur_scan)
        t0 = time.time()

        submap = self._crop_global_map_in_FOV(pose_estimation, self.cur_odom)

        # Coarse
        T, _ = registration_at_scale(scan_to_map, submap, initial=pose_estimation, scale=5.0)
        # Fine
        T, fitness = registration_at_scale(scan_to_map, submap, initial=T, scale=1.0)

        dt = time.time() - t0
        self.get_logger().info(f'ICP time: {dt:.3f} s, fitness: {fitness:.4f} / {self.LOCALIZATION_TH:.4f}')
        self.get_logger().info(f'Estimated T_map_to_odom:\n{T}')

        if fitness > self.LOCALIZATION_TH:
            self.T_map_to_odom = T
            # Publish map->odom as Odometry msg for convenience (pose only)
            odom_msg = Odometry()
            xyz = self.T_map_to_odom[:3, 3]
            quat = self._quat_from_matrix(self.T_map_to_odom)
            odom_msg.pose.pose.position = Point(x=xyz[0], y=xyz[1], z=xyz[2])
            odom_msg.pose.pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
            odom_msg.header.frame_id = 'map'
            # Use current odom stamp if present
            if self.cur_odom is not None:
                odom_msg.header.stamp = self.cur_odom.header.stamp
            self.pub_map_to_odom.publish(odom_msg)
            return True
        else:
            self.get_logger().warn('ICP did not converge sufficiently (fitness below threshold).')
            return False

    @staticmethod
    def _quat_from_matrix(T: np.ndarray) -> np.ndarray:
        if tft is not None:
            return np.array(tft.quaternion_from_matrix(T), dtype=np.float64)
        # Minimal fallback via eigen decomposition (not as robust as tf)
        R = T[:3, :3]
        qw = math.sqrt(max(0.0, 1.0 + np.trace(R))) / 2.0
        qx = (R[2, 1] - R[1, 2]) / (4.0 * qw + 1e-12)
        qy = (R[0, 2] - R[2, 0]) / (4.0 * qw + 1e-12)
        qz = (R[1, 0] - R[0, 1]) / (4.0 * qw + 1e-12)
        return np.array([qx, qy, qz, qw], dtype=np.float64)


def main():
    rclpy.init()
    node = FastLioLocalizationNode()

    # Optional: subscribe once to /initialpose (like ROS1 behavior).
    # In practice, many setups feed identity or a prior and let ICP snap in.
    # If you want to honor /initialpose exactly once:
    initial_set = {'done': False}

    def _once_initialpose(msg: PoseWithCovarianceStamped):
        if initial_set['done']:
            return
        T0 = pose_to_mat_from_pose_with_cov(msg)
        node.T_map_to_odom = T0
        node.initialized = True
        node.get_logger().info('Initial pose received from /initialpose. Initialization complete.')
        initial_set['done'] = True
        # After first message, we can let GC drop this subscription
        node.destroy_subscription(sub_once)

    sub_once = node.create_subscription(PoseWithCovarianceStamped, '/initialpose', _once_initialpose, 10)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
