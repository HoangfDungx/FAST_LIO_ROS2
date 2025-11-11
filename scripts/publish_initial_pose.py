#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time

import rclpy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from rclpy.node import Node

from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Point, Quaternion

# pip install tf-transformations (ROS2-friendly)
import tf_transformations as tft


def build_initial_pose(x, y, z, yaw, pitch, roll, frame_id='map') -> PoseWithCovarianceStamped:
    # Note: original code parsed (yaw, pitch, roll) but passed to quaternion_from_euler(roll, pitch, yaw)
    qx, qy, qz, qw = tft.quaternion_from_euler(roll, pitch, yaw)
    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = frame_id
    # rclpy will stamp this when publishing, but we can leave it zero or set explicitly via node clock if desired.
    msg.pose.pose = Pose(
        position=Point(x=x, y=y, z=z),
        orientation=Quaternion(x=qx, y=qy, z=qz, w=qw)
    )
    # covariance left at zeros (same as original behavior)
    return msg


def main():
    parser = argparse.ArgumentParser(description='Publish initial pose to /initialpose (ROS 2).')
    parser.add_argument('x', type=float)
    parser.add_argument('y', type=float)
    parser.add_argument('z', type=float)
    parser.add_argument('yaw', type=float)
    parser.add_argument('pitch', type=float)
    parser.add_argument('roll', type=float)
    parser.add_argument('--frame-id', default='map', help='TF frame for the pose (default: map)')
    parser.add_argument('--repeat', type=int, default=3, help='How many times to publish (default: 3)')
    parser.add_argument('--interval', type=float, default=0.1, help='Seconds between publishes (default: 0.1)')
    parser.add_argument('--durability', choices=['volatile', 'transient_local'], default='volatile',
                        help='QoS durability for publisher (default: volatile)')
    args = parser.parse_args()

    rclpy.init()
    node = Node('publish_initial_pose')

    durability = (QoSDurabilityPolicy.TRANSIENT_LOCAL
                  if args.durability == 'transient_local'
                  else QoSDurabilityPolicy.VOLATILE)

    qos = QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=durability
    )

    pub = node.create_publisher(PoseWithCovarianceStamped, '/initialpose', qos)

    msg = build_initial_pose(args.x, args.y, args.z,
                             args.yaw, args.pitch, args.roll,
                             frame_id=args.frame_id)

    node.get_logger().info(
        f'Initial Pose -> x:{args.x:.3f} y:{args.y:.3f} z:{args.z:.3f} '
        f'yaw:{args.yaw:.3f} pitch:{args.pitch:.3f} roll:{args.roll:.3f} '
        f'frame:{args.frame_id} (QoS durability: {args.durability})'
    )

    # Publish a few times to improve chance of reception
    for i in range(max(1, args.repeat)):
        # Stamp right before publish (optional)
        msg.header.stamp = node.get_clock().now().to_msg()
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.0)
        time.sleep(args.interval)

    node.get_logger().info('Initial pose published.')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()