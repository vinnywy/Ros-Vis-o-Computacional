from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    image_arg = DeclareLaunchArgument(
        "image_path",
        default_value="dog.png",
        description="Caminho absoluto para a imagem de entrada",
    )

    turtlesim_node = Node(
        package="turtlesim",
        executable="turtlesim_node",
        name="turtlesim",
    )

    # TimerAction aguarda 2s para o turtlesim subir antes de conectar
    controller_node = TimerAction(
        period=2.0,
        actions=[Node(
            package="turtle_draw",
            executable="turtle_controller",
            name="turtle_controller",
            parameters=[{
                "image_path":     LaunchConfiguration("image_path"),
                "blur_sigma":     1.4,
                "edge_threshold": 0.15,
                "max_points":     600,
            }],
            output="screen",
        )],
    )

    return LaunchDescription([image_arg, turtlesim_node, controller_node])