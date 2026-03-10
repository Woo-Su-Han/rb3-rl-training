import os
import subprocess

from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable, OpaqueFunction, TimerAction
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def launch_setup(context, *args, **kwargs):
    pkg_description = get_package_share_directory("rbpodo_description")
    pkg_bringup     = get_package_share_directory("rbpodo_bringup")

    dual_model_path    = os.path.join(pkg_description, "robots", "rb3_dual.urdf.xacro")
    controllers_yaml   = os.path.join(pkg_bringup, "config", "controllers_dual.yaml")

    # GAZEBO_MODEL_PATH
    gazebo_model_path = os.path.dirname(pkg_description)
    existing = os.environ.get("GAZEBO_MODEL_PATH", "")
    new_model_path = gazebo_model_path + (":" + existing if existing else "")

    # 두 로봇 합친 URDF 생성 (단일 플러그인, 네임스페이스 없음)
    urdf_str = subprocess.check_output([
        "xacro", dual_model_path,
        "robot2_x:=1.5",
        "robot2_y:=0.0",
        f"params_file:={controllers_yaml}",
    ]).decode("utf-8")

    # robot_state_publisher (두 로봇 모두 포함)
    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{
            "robot_description": ParameterValue(urdf_str, value_type=str)
        }],
        output="screen",
    )

    # 단일 엔티티로 스폰
    spawn = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-topic", "/robot_description",
            "-entity", "rb3_dual",
            "-x", "0", "-y", "0", "-z", "0",
        ],
        output="screen",
    )

    # 컨트롤러 로드 (스폰 후 3초 대기)
    load_jsb = TimerAction(
        period=3.0,
        actions=[Node(
            package="controller_manager",
            executable="spawner",
            arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
            output="screen",
        )]
    )

    load_rb3_1_jtc = TimerAction(
        period=5.0,
        actions=[Node(
            package="controller_manager",
            executable="spawner",
            arguments=["rb3_1_joint_trajectory_controller", "--controller-manager", "/controller_manager"],
            output="screen",
        )]
    )

    load_rb3_2_jtc = TimerAction(
        period=5.0,
        actions=[Node(
            package="controller_manager",
            executable="spawner",
            arguments=["rb3_2_joint_trajectory_controller", "--controller-manager", "/controller_manager"],
            output="screen",
        )]
    )

    gazebo = ExecuteProcess(
        cmd=[
            "gazebo", "--verbose",
            "-s", "libgazebo_ros_init.so",      # /reset_simulation, /pause_physics 등
            "-s", "libgazebo_ros_factory.so",   # /spawn_entity, /delete_entity
            "-s", "libgazebo_ros_state.so",     # /gazebo/link_states (자기충돌 감지용)
        ],
        output="screen",
        additional_env={"GAZEBO_MODEL_PATH": new_model_path},
    )

    return [
        SetEnvironmentVariable("GAZEBO_MODEL_PATH", new_model_path),
        gazebo,
        rsp,
        spawn,
        load_jsb,
        load_rb3_1_jtc,
        load_rb3_2_jtc,
    ]


def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=launch_setup),
    ])
