"""
Modulo de pre-processamento e deteccao de bordas.
Implementado do zero com NumPy — sem algoritmos prontos de visao.
OpenCV restrito ao carregamento da imagem (cv2.imread).
"""

import numpy as np
import cv2


def load_image(path: str) -> np.ndarray:
    """Carrega imagem e converte BGR->RGB via slice (sem cv2.cvtColor)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Imagem nao encontrada: {path}")
    return img[:, :, ::-1].astype(np.float64)


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Pesos ITU-R BT.601: aproximam a sensibilidade espectral do olho humano.
    Verde recebe maior peso pois o olho e mais sensivel a essa faixa.
    """
    return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]


def gaussian_kernel(size: int = 5, sigma: float = 1.4) -> np.ndarray:
    """
    Kernel gaussiano 2D normalizado.
    Suaviza a imagem antes da derivacao para suprimir ruido de alta frequencia.
    sigma maior -> suavizacao mais agressiva -> menos bordas espurias.
    """
    k = size // 2
    y, x = np.mgrid[-k : k + 1, -k : k + 1]
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolucao 2D vetorizada via sliding_window_view.
    Evita loops Python: cria uma visao de janelas deslizantes e aplica
    o produto interno com einsum — tudo em operacoes NumPy.
    Padding 'reflect' replica pixels da borda para evitar artefatos.
    """
    kh, kw = kernel.shape
    padded = np.pad(img, ((kh // 2, kh // 2), (kw // 2, kw // 2)), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw))
    return np.einsum("ijkl,kl->ij", windows, kernel)


def sobel_edges(gray: np.ndarray, threshold: float = 0.15):
    """
    Deteccao de bordas pelo operador de Sobel.
    Kx: detecta variacao horizontal de intensidade (bordas verticais).
    Ky = Kx.T: detecta variacao vertical (bordas horizontais).
    Magnitude = sqrt(Gx^2 + Gy^2) — intensidade do gradiente em cada pixel.
    threshold [0,1]: limiar sobre magnitude normalizada.
    """
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    Ky = Kx.T

    Gx = convolve2d(gray, Kx)
    Gy = convolve2d(gray, Ky)

    magnitude = np.hypot(Gx, Gy)
    magnitude /= magnitude.max()

    edges = (magnitude > threshold).astype(np.uint8) * 255
    return edges, magnitude


def run_pipeline(image_path: str, blur_sigma: float = 1.4, edge_threshold: float = 0.15):
    """
    Executa o pipeline completo e retorna o mapa binario de bordas.
    Ponto de entrada para o no ROS 2.
    """
    img     = load_image(image_path)
    gray    = to_grayscale(img)
    blurred = convolve2d(gray, gaussian_kernel(size=5, sigma=blur_sigma))
    edges, magnitude = sobel_edges(blurred, threshold=edge_threshold)

    print(f"[preprocessing] Imagem: {img.shape[1]}x{img.shape[0]}")
    print(f"[preprocessing] Pixels de borda: {(edges > 0).sum()}")
    return edges, img.shape
