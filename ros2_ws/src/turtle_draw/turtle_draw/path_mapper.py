"""
Mapeia pixels de borda para coordenadas do espaco turtlesim.

Espaco turtlesim: x em [0, 11], y em [0, 11], origem no canto inferior esquerdo.
Espaco da imagem: row em [0, H], col em [0, W], origem no canto superior esquerdo.
Logo, o eixo Y deve ser invertido na conversao.
"""

import numpy as np

TURTLE_MIN = 0.5
TURTLE_MAX = 10.5


def extract_and_downsample(edges: np.ndarray, max_points: int = 800) -> np.ndarray:
    """
    Extrai coordenadas dos pixels de borda e reduz o volume de pontos.
    step e calculado para amostrar uniformemente ate max_points.
    Retorna array (N, 2) com [row, col].
    """
    coords = np.argwhere(edges > 0)

    if len(coords) > max_points:
        step = len(coords) // max_points
        coords = coords[::step]

    return coords


def map_to_turtlesim(coords: np.ndarray, img_shape: tuple) -> np.ndarray:
    H, W = img_shape[:2]
    rows, cols = coords[:, 0], coords[:, 1]

    # Espelha X para corrigir orientacao
    x = TURTLE_MAX - (cols / W) * (TURTLE_MAX - TURTLE_MIN)
    y = TURTLE_MAX - (rows / H) * (TURTLE_MAX - TURTLE_MIN)

    return np.column_stack([x, y])

def order_by_nearest_neighbor(points: np.ndarray) -> np.ndarray:
    """
    Ordena pontos por vizinho mais proximo a partir do centroide.
    Reduz saltos grandes, produzindo tracado mais continuo.
    Complexidade O(N^2) — adequado para N <= 1000.
    """
    centroid  = points.mean(axis=0)
    start_idx = np.argmin(np.linalg.norm(points - centroid, axis=1))

    ordered   = [points[start_idx]]
    remaining = np.delete(points, start_idx, axis=0)

    while len(remaining) > 0:
        last    = ordered[-1]
        dists   = np.linalg.norm(remaining - last, axis=1)
        nearest = np.argmin(dists)
        ordered.append(remaining[nearest])
        remaining = np.delete(remaining, nearest, axis=0)

    return np.array(ordered)


def build_path(edges: np.ndarray, img_shape: tuple, max_points: int = 1200) -> np.ndarray:
    """Pipeline completo: pixels de borda -> caminho ordenado em espaco turtlesim."""
    coords = extract_and_downsample(edges, max_points)
    turtle = map_to_turtlesim(coords, img_shape)
    path   = order_by_nearest_neighbor(turtle)

    print(f"[path_mapper] Pontos no caminho: {len(path)}")
    return path