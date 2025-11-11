#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import math
import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose, Point, Quaternion, TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster

# pip install tf-transformations
import tf_transformations as tft


def pose_to_mat_from_odom(odom_msg: Odometry) -> np.ndarray:
    p = odom_msg.pose.pose.position
    q = odom_msg.pose.pose.orientation
    return xyz_quat_to_mat44(p.x, p.y, p.z, q.x, q.y, q.z, q.w)


def pose_to_mat_from_pose(pose_msg) -> np.ndarray:
    p = pose_msg.pose.pose.position
    q = pose_msg.pose.pose.orientation
    return xyz_quat_to_mat44(p.x, p.y, p.z, q.x, q.y, q.z, q.w)


def xyz_quat_to_mat44(x, y, z, qx, qy, qz, qw) -> np.ndarray:
    M = tft.quaternion_matrix([qx, qy, qz, qw])
    M[0:3, 3] = [x, y, z]
    return M


def mat_to_xyz_quat(T: np.ndarray):
    xyz = T[:3, 3]
    quat = tft.quaternion_from_matrix(T)
    return xyz, quat  # (3,), (4,)


class TransformFusionNode(Node):
    def __init__(self):
        super().__init__('transform_fusion')

        # Params (same defaults as ROS1 script)
        self.declare_parameter('freq_pub_localization', 50.0)
        self.freq = float(self.get_parameter('freq_pub_localization').value)
        self.period = 1.0 / max(1e-6, self.freq)

        # State
        self.cur_odom_to_baselink: Odometry | None = None
        self.cur_map_to_odom: Odometry | None = None

        # Pub/Sub
        self.sub_odom = self.create_subscription(
            Odometry, '/Odometry', self.cb_save_cur_odom, 10
        )
        self.sub_map_to_odom = self.create_subscription(
            Odometry, '/map_to_odom', self.cb_save_map_to_odom, 10
        )
        self.pub_localization = self.create_publisher(
            Odometry, '/localization', 10
        )

        # TF broadcaster
        self.br = TransformBroadcaster(self)

        # Timer (publishing loop)
        self.timer = self.create_timer(self.period, self.timer_cb)

        self.get_logger().info('Transform Fusion Node Inited (ROS 2)...')

    # --- Callbacks ---
    def cb_save_cur_odom(self, odom_msg: Odometry):
        self.cur_odom_to_baselink = odom_msg

    def cb_save_map_to_odom(self, odom_msg: Odometry):
        self.cur_map_to_odom = odom_msg

    def timer_cb(self):
        # Build T_map_to_odom
        if self.cur_map_to_odom is not None:
            T_map_to_odom = pose_to_mat_from_pose(self.cur_map_to_odom)
        else:
            T_map_to_odom = np.eye(4)

        # Broadcast TF: map -> camera_init  (kept exactly like original)
        xyz, quat = mat_to_xyz_quat(T_map_to_odom)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'camera_init'
        t.transform.translation.x = float(xyz[0])
        t.transform.translation.y = float(xyz[1])
        t.transform.translation.z = float(xyz[2])
        t.transform.rotation.x = float(quat[0])
        t.transform.rotation.y = float(quat[1])
        t.transform.rotation.z = float(quat[2])
        t.transform.rotation.w = float(quat[3])
        self.br.sendTransform(t)

        # Publish /localization if we have current odometry
        if self.cur_odom_to_baselink is not None:
            # T_odom_to_base_link from /Odometry
            T_odom_to_base_link = pose_to_mat_from_odom(self.cur_odom_to_baselink)

            # No time sync correction (as in original)
            T_map_to_base_link = T_map_to_odom @ T_odom_to_base_link
            l_xyz, l_quat = mat_to_xyz_quat(T_map_to_base_link)

            localization = Odometry()
            localization.pose.pose = Pose(
                position=Point(*l_xyz.tolist()),
                orientation=Quaternion(*l_quat.tolist())
            )
            localization.twist = copy.deepcopy(self.cur_odom_to_baselink.twist)

            # Keep original header semantics
            localization.header.stamp = self.cur_odom_to_baselink.header.stamp
            localization.header.frame_id = 'map'
            localization.child_frame_id = 'body'

            self.pub_localization.publish(localization)


def main():
    rclpy.init()
    node = TransformFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
