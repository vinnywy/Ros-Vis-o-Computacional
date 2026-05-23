import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/vinnywy/Documentos/Github/Ros-Vis-o-Computacional/ros2_ws/install/turtle_draw'
