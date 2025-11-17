from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterFile


def generate_launch_description():
    # ---- Launch args ----
    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Start RViz2 with predefined config'
    )
    map_arg = DeclareLaunchArgument(
        'map', default_value='',
        description='map name'
    )

    # Package share
    pkg_share = FindPackageShare('fast_lio')

    # Config files (livox.yaml, rviz config)
    livox_yaml = PathJoinSubstitution([pkg_share, 'config', 'mid360_localization.yaml'])
    rviz_cfg = PathJoinSubstitution([pkg_share, 'rviz_cfg', 'localization.rviz'])
    map_path = PathJoinSubstitution([pkg_share, 'PCD', LaunchConfiguration('map')])

    # Parameter file wrapper (so missing file throws a good error)
    livox_params = ParameterFile(livox_yaml, allow_substs=True)

    # ----- Nodes -----

    # fastlio mapping (laserMapping)
    laser_mapping = Node(
        package='fast_lio',
        executable='fastlio_mapping',
        name='laserMapping',
        output='screen',
        parameters=[
            livox_params,
            {
                'feature_extract_enable': False,
                'point_filter_num': 4,
                'max_iteration': 3,
                'filter_size_surf': 0.5,
                'filter_size_map': 0.5,
                'cube_side_length': 1000.0,
                'runtime_pos_log_enable': False,
                'pcd_save_enable': False,
            }
        ],
    )

    # Global localization
    global_localization = Node(
        package='fast_lio',
        executable='global_localization.py',   # or 'global_localization' if installed as entry point
        name='global_localization',
        output='screen',
    )

    # Transform fusion
    transform_fusion = Node(
        package='fast_lio',
        executable='transform_fusion.py',      # or 'transform_fusion' if installed as entry point
        name='transform_fusion',
        output='screen',
    )

    # PCD map publisher (only if map arg is non-empty)
    # ROS 2 pcl_ros pcd_to_pointcloud typically takes: <pcd_path> <publish_rate_hz>
    # Remap 'cloud_pcd' -> '/map' and set frame_id='map'
    pcd_to_pointcloud = Node(
        package='pcl_ros',
        executable='pcd_to_pointcloud',
        name='map_publisher',
        output='screen',
        arguments=[map_path, '5'],
        remappings=[('cloud_pcd', '/map')],
        parameters=[{'frame_id': 'map'}],
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration('map'), "' != ''"])),
    )

    # RViz2 (conditional)
    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        output='screen',
        arguments=['-d', rviz_cfg],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    return LaunchDescription([
        rviz_arg,
        map_arg,
        laser_mapping,
        global_localization,
        transform_fusion,
        pcd_to_pointcloud,
        rviz2,
    ])
