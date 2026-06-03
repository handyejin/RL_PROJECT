"""
지리 유틸리티 모듈
- Haversine 거리 계산
- 정류소 간 거리 행렬 생성
- 거리 → 이동 step 수 변환
"""

import numpy as np

EARTH_RADIUS_KM = 6371.0


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    두 좌표 간 Haversine 거리 (km).

    Parameters
    ----------
    lat1, lon1 : 출발지 위경도 (도 단위)
    lat2, lon2 : 도착지 위경도 (도 단위)

    Returns
    -------
    float : 거리 (km)
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    정류소 좌표 배열로부터 거리 행렬 생성.

    Parameters
    ----------
    coords : np.ndarray, shape (n_stations, 2)
        각 행이 [위도, 경도].

    Returns
    -------
    np.ndarray, shape (n_stations, n_stations)
        거리 행렬 (km). 대각선은 0.
    """
    n = len(coords)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine(coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1])
            dist[i, j] = d
            dist[j, i] = d
    return dist


def distance_to_travel_steps(
    distance_matrix: np.ndarray,
    speed_kmh: float,
    step_duration_min: float,
) -> np.ndarray:
    """
    거리 행렬을 이동 step 수 행렬로 변환.

    이동 step 수 = ceil(거리 / (속도 × step당 시간))
    같은 정류소(거리=0)는 0 step.

    Parameters
    ----------
    distance_matrix : np.ndarray, shape (n, n)
    speed_kmh : float, 트럭 평균 속도 (km/h)
    step_duration_min : float, 1 step 길이 (분)

    Returns
    -------
    np.ndarray (int), shape (n, n)
    """
    speed_km_per_step = speed_kmh * (step_duration_min / 60.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        steps = np.ceil(distance_matrix / speed_km_per_step).astype(int)
    # 같은 정류소 → 0 step
    np.fill_diagonal(steps, 0)
    return steps
