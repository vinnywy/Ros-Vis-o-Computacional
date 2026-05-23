"""
No ROS 2 — TurtleController
Carrega imagem, extrai contorno e comanda o turtlesim para desenha-lo.

Servicos utilizados:
  /turtle1/teleport_absolute  — move a tartaruga para (x, y, theta)
  /turtle1/set_pen            — liga/desliga a caneta
"""

import rclpy
from rclpy.node import Node
from turtlesim.srv import TeleportAbsolute, SetPen
import numpy as np

from turtle_draw.preprocessing import run_pipeline
from turtle_draw.path_mapper import build_path

JUMP_THRESHOLD = 1.2   # distancia maxima para considerar segmento continuo


class TurtleController(Node):

    def __init__(self):
        super().__init__("turtle_controller")

        self.declare_parameter("image_path",     "dog.png")
        self.declare_parameter("blur_sigma",     1.0)
        self.declare_parameter("edge_threshold", 0.10)
        self.declare_parameter("max_points",     1200)

        self._teleport = self.create_client(TeleportAbsolute, "/turtle1/teleport_absolute")
        self._set_pen  = self.create_client(SetPen,           "/turtle1/set_pen")

        self._wait_for_services()
        self.get_logger().info("Iniciando pipeline de visao computacional...")
        self._draw()

    def _wait_for_services(self):
        for client in (self._teleport, self._set_pen):
            while not client.wait_for_service(timeout_sec=2.0):
                self.get_logger().warn(f"Aguardando servico: {client.srv_name}")

    def _call(self, client, request):
        """Chama servico de forma sincrona e aguarda resposta."""
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def _teleport_to(self, x: float, y: float, theta: float = 0.0):
        req = TeleportAbsolute.Request()
        req.x, req.y, req.theta = float(x), float(y), float(theta)
        self._call(self._teleport, req)

    def _pen(self, on: bool, r: int = 255, g: int = 255, b: int = 255, width: int = 1):
        req = SetPen.Request()
        req.r, req.g, req.b = r, g, b
        req.width = width
        req.off   = 0 if on else 1   # off=1 levanta caneta, off=0 abaixa
        self._call(self._set_pen, req)

    def _draw(self):
        image_path     = self.get_parameter("image_path").value
        blur_sigma     = self.get_parameter("blur_sigma").value
        edge_threshold = self.get_parameter("edge_threshold").value
        max_points = self.get_parameter("max_points").value

        edges, img_shape = run_pipeline(image_path, blur_sigma, edge_threshold)
        path = build_path(edges, img_shape, max_points)

        self.get_logger().info(f"max_points recebido: {max_points}") 
        self._draw_path(path)
        self.get_logger().info("Desenho concluido.")

    def _draw_path(self, path: np.ndarray):
        """
        Percorre o caminho levantando a caneta em saltos grandes
        (segmentos desconexos do contorno).
        """
        self._pen(on=False)
        self._teleport_to(*path[0])
        self._pen(on=True)

        for i in range(1, len(path)):
            dist = np.linalg.norm(path[i] - path[i - 1])

            if dist > JUMP_THRESHOLD:
                self._pen(on=False)
                self._teleport_to(*path[i])
                self._pen(on=True)
            else:
                self._teleport_to(*path[i])

            if i % 100 == 0:
                self.get_logger().info(f"  {i}/{len(path)} pontos")


def main(args=None):
    rclpy.init(args=args)
    node = TurtleController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()