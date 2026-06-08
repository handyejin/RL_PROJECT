"""후보 Top-K action wrapper.

문제:
    원본 action space는 모든 정류소 N개 중 하나를 고르는 구조다.
    N이 크면 랜덤 탐색과 policy gradient가 대부분 의미 없는 정류소에
    확률을 쓰게 되어 BC 이후 RL fine-tuning이 쉽게 무너진다.

해결:
    원본 env는 그대로 두고, agent-local wrapper에서 현재 state 기준으로
    의미 있는 후보 K개를 만든다. agent action은 0..K-1 rank이고,
    wrapper가 이를 실제 정류소 index로 변환해 원본 env.step()에 전달한다.

해석:
    action=0은 현재 후보 중 가장 강한 휴리스틱 후보,
    action=1은 두 번째 후보처럼 동작한다. 따라서 RL은 전체 N개 정류소 중
    아무 곳이나 고르는 대신, 좋은 후보들 사이의 선택을 학습한다.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CandidateTopKActionWrapper(gym.Wrapper):
    """Discrete(N) station action을 Discrete(K) candidate-rank action으로 바꾼다.

    추가 옵션:
        travel_coef:
            멀리 있는 정류소의 score를 낮춘다. 원본 reward에 이동 비용이 있으므로
            후보 선정 단계에서도 가까운 대안을 조금 선호하게 만든다.
        zone_mode="static3":
            정류소를 경도 기준 3개 권역으로 나누고, 트럭별 담당 권역 밖 후보에
            작은 penalty를 준다. hard rule이 아니라 soft penalty라서 후보가 막히지 않는다.
        feature_mode="basic":
            현재 Top-K 후보의 score, 이동거리, 재고비율 등을 observation 뒤에 붙인다.
            agent action이 rank이므로 "0번 후보가 어떤 후보인지"를 함께 알려주는 역할이다.
    """

    VALID_MODES = {"imbalance", "forecast_imbalance"}
    VALID_ZONE_MODES = {"none", "static3"}
    VALID_FEATURE_MODES = {"none", "basic"}
    FEATURE_DIM = 8

    def __init__(
        self,
        env,
        top_k: int = 12,
        mode: str = "imbalance",
        travel_coef: float = 0.0,
        zone_mode: str = "none",
        zone_count: int = 3,
        zone_penalty: float = 0.0,
        feature_mode: str = "none",
    ):
        if top_k < 2:
            raise ValueError("candidate top_k must be >= 2")
        if mode not in self.VALID_MODES:
            raise ValueError(f"unknown candidate mode: {mode}")
        if zone_mode not in self.VALID_ZONE_MODES:
            raise ValueError(f"unknown zone mode: {zone_mode}")
        if feature_mode not in self.VALID_FEATURE_MODES:
            raise ValueError(f"unknown candidate feature mode: {feature_mode}")
        super().__init__(env)
        self.top_k = int(top_k)
        self.mode = mode
        self.travel_coef = float(travel_coef)
        self.zone_mode = zone_mode
        self.zone_count = max(1, int(zone_count))
        self.zone_penalty = float(zone_penalty)
        self.feature_mode = feature_mode
        self._station_zones: np.ndarray | None = None
        self.action_space = spaces.Discrete(self.top_k)
        self.observation_space = self._build_observation_space(env.observation_space)

    def __getattr__(self, name):
        """wrapper에 없는 속성은 원본 env로 위임한다."""
        return getattr(self.env, name)

    def reset(self, *args, **kwargs):
        """원본 env reset 후, 필요하면 후보 feature를 observation에 붙인다."""
        obs, info = self.env.reset(*args, **kwargs)
        return self._augment_obs(obs), info

    def step(self, action):
        """candidate rank action을 실제 station action으로 변환해 실행한다."""
        candidates, valid = self._candidate_actions()
        rank = int(action)
        if rank < 0 or rank >= len(candidates) or not valid[rank]:
            rank = int(np.flatnonzero(valid)[0])
        station_action = int(candidates[rank])
        obs, reward, terminated, truncated, info = self.env.step(station_action)
        return self._augment_obs(obs), reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """candidate 중 실제로 채워진 rank만 action 가능하게 표시한다."""
        _, valid = self._candidate_actions()
        return valid.astype(bool)

    def teacher_action(self, bc_policy: str, horizon: int = 6) -> int:
        """원본 station teacher action을 candidate rank로 변환한다."""
        if bc_policy == "future_heuristic":
            base_action = self._future_teacher_action(horizon)
        elif bc_policy == "forecast_heuristic":
            base_action = self._forecast_teacher_action()
        else:
            base_action = self._masked_teacher_action(projected=False)
        return self.map_station_to_rank(base_action)

    def map_station_to_rank(self, station_action: int) -> int:
        """실제 station index가 candidate 안에 있으면 rank를 반환하고, 없으면 0을 반환한다."""
        candidates, valid = self._candidate_actions()
        for rank, station in enumerate(candidates):
            if valid[rank] and int(station) == int(station_action):
                return rank
        return 0

    def candidate_station_ids(self) -> np.ndarray:
        """현재 state의 candidate station index 목록을 반환한다."""
        candidates, _ = self._candidate_actions()
        return candidates.copy()

    def _candidate_actions(self) -> tuple[np.ndarray, np.ndarray]:
        """현재 state에서 점수가 높은 station top-K를 만든다."""
        scores = self._station_scores(projected=self.mode == "forecast_imbalance")
        scores = self._apply_travel_and_zone_penalty(scores)
        base_mask = self.env.action_masks()
        scores = scores.astype(np.float32, copy=True)
        scores[~base_mask] = -np.inf

        valid_stations = np.flatnonzero(np.isfinite(scores))
        if len(valid_stations) == 0:
            fallback = int(np.flatnonzero(base_mask)[0])
            candidates = np.full(self.top_k, fallback, dtype=np.int64)
            valid = np.zeros(self.top_k, dtype=bool)
            valid[0] = True
            return candidates, valid

        order = valid_stations[np.argsort(scores[valid_stations])[::-1]]
        selected = order[: self.top_k].astype(np.int64)
        valid = np.zeros(self.top_k, dtype=bool)
        valid[: len(selected)] = True

        # K개보다 후보가 적으면 첫 후보로 padding한다. padding rank는 mask=False다.
        candidates = np.full(self.top_k, int(selected[0]), dtype=np.int64)
        candidates[: len(selected)] = selected
        return candidates, valid

    def _station_scores(self, projected: bool) -> np.ndarray:
        """트럭 적재 상태에 따라 방문 가치가 큰 정류소 score를 계산한다."""
        bikes = self.env.bikes.astype(np.float32)
        if projected:
            bikes = self._projected_bikes_from_forecast(bikes)
        target = self.env.data.capacity.astype(np.float32) * self.env.target_fill_ratio
        truck = self.env.trucks[self.env.current_truck]
        if truck.load == 0:
            # 빈 트럭은 자전거가 과잉인 정류소로 가야 적재할 수 있다.
            scores = bikes - target
        elif truck.load >= self.env.truck_capacity:
            # 가득 찬 트럭은 자전거가 부족한 정류소로 가야 하차할 수 있다.
            scores = target - bikes
        else:
            # 부분 적재 상태에서는 과잉/부족이 큰 곳을 우선 후보로 둔다.
            scores = np.abs(bikes - target)
        return scores.astype(np.float32)

    def _apply_travel_and_zone_penalty(self, scores: np.ndarray) -> np.ndarray:
        """후보 score에 이동비용과 권역 penalty를 반영한다."""
        adjusted = scores.astype(np.float32, copy=True)
        if self.travel_coef != 0.0:
            loc = self.env.trucks[self.env.current_truck].location
            # travel_steps는 10분 단위 이동 시간이다. coef는 "score에서 몇 점 깎을지"를 뜻한다.
            travel_steps = self.env.data.travel_steps[loc].astype(np.float32)
            adjusted -= self.travel_coef * travel_steps

        if self.zone_mode != "none" and self.zone_penalty != 0.0:
            truck_zone = self.env.current_truck % self.zone_count
            zones = self._get_station_zones()
            adjusted[zones != truck_zone] -= self.zone_penalty
        return adjusted

    def _projected_bikes_from_forecast(self, bikes: np.ndarray) -> np.ndarray:
        """외부 forecast가 있으면 1시간 뒤 예상 재고를 만든다."""
        forecast = getattr(self.env.data, "agent_demand_forecast", None)
        if forecast is None or len(forecast) == 0:
            return bikes
        idx = min(int(self.env.t), len(forecast) - 1)
        net = forecast[idx, :, 2].astype(np.float32)
        capacity = self.env.data.capacity.astype(np.float32)
        return np.clip(bikes + net, 0.0, capacity)

    def _build_observation_space(self, base_space: spaces.Space) -> spaces.Space:
        """후보 feature를 붙일 때 observation dimension을 확장한다."""
        if self.feature_mode == "none":
            return base_space
        if not isinstance(base_space, spaces.Box):
            raise TypeError("candidate feature observation requires Box observation_space")
        base_dim = int(base_space.shape[0])
        extra_dim = self.top_k * self.FEATURE_DIM
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(base_dim + extra_dim,),
            dtype=np.float32,
        )

    def _augment_obs(self, obs: np.ndarray) -> np.ndarray:
        """후보 feature를 observation 뒤에 붙인다."""
        obs = np.asarray(obs, dtype=np.float32)
        if self.feature_mode == "none":
            return obs
        return np.concatenate([obs, self._candidate_features()], dtype=np.float32)

    def _candidate_features(self) -> np.ndarray:
        """Top-K 각 후보를 설명하는 feature를 만든다.

        feature 순서:
            valid, score_norm, travel_norm, bike_ratio,
            projected_bike_ratio, forecast_net_norm, same_zone, rank_norm
        """
        candidates, valid = self._candidate_actions()
        bikes = self.env.bikes.astype(np.float32)
        capacity = np.maximum(self.env.data.capacity.astype(np.float32), 1.0)
        projected = self._projected_bikes_from_forecast(bikes)
        scores = self._station_scores(projected=self.mode == "forecast_imbalance")
        adjusted = self._apply_travel_and_zone_penalty(scores)

        loc = self.env.trucks[self.env.current_truck].location
        max_steps = max(float(getattr(self.env, "max_travel_steps", 10)), 1.0)
        travel_norm_all = np.clip(self.env.data.travel_steps[loc].astype(np.float32) / max_steps, 0.0, 1.0)
        raw_net = projected - bikes
        net_norm = np.clip(raw_net / capacity, -1.0, 1.0)

        zones = self._get_station_zones()
        truck_zone = self.env.current_truck % self.zone_count
        denom = max(float(np.nanmax(np.abs(adjusted[np.isfinite(adjusted)]))) if np.isfinite(adjusted).any() else 1.0, 1.0)
        rows = []
        for rank, station in enumerate(candidates):
            s = int(station)
            rows.extend(
                [
                    1.0 if valid[rank] else 0.0,
                    float(np.clip(adjusted[s] / denom, -1.0, 1.0)),
                    float(travel_norm_all[s]),
                    float(np.clip(bikes[s] / capacity[s], 0.0, 1.0)),
                    float(np.clip(projected[s] / capacity[s], 0.0, 1.0)),
                    float(net_norm[s]),
                    1.0 if zones[s] == truck_zone else 0.0,
                    float(rank / max(self.top_k - 1, 1)),
                ]
            )
        return np.asarray(rows, dtype=np.float32)

    def _get_station_zones(self) -> np.ndarray:
        """정류소를 경도 기준으로 3개 권역으로 나눈다.

        마포구처럼 동서로 긴 지역에서는 경도 기준 split이 단순하면서도 설명하기 쉽다.
        좌표가 없으면 station index 순서로 deterministic fallback한다.
        """
        if self._station_zones is not None and len(self._station_zones) == self.env.N:
            return self._station_zones

        coords = getattr(self.env.data, "station_coords", None)
        if coords is not None and len(coords) == self.env.N and coords.shape[1] >= 2:
            key = coords[:, 1]  # longitude
        else:
            key = np.arange(self.env.N, dtype=np.float32)

        order = np.argsort(key)
        zones = np.zeros(self.env.N, dtype=np.int64)
        for zone, station_indexes in enumerate(np.array_split(order, self.zone_count)):
            zones[station_indexes] = zone
        self._station_zones = zones
        return zones

    def _masked_teacher_action(self, projected: bool) -> int:
        """현재 candidate scoring과 같은 방식의 station-level teacher action."""
        scores = self._station_scores(projected=projected)
        mask = self.env.action_masks()
        scores = scores.astype(np.float32, copy=True)
        scores[~mask] = -np.inf
        best = int(np.argmax(scores))
        if not np.isfinite(scores[best]):
            return int(np.flatnonzero(mask)[0])
        return best

    def _forecast_teacher_action(self) -> int:
        """forecast projected imbalance 기반 teacher station action."""
        return self._masked_teacher_action(projected=True)

    def _future_teacher_action(self, horizon: int) -> int:
        """실제 미래 수요를 쓰는 oracle teacher station action."""
        bikes = self.env.bikes.astype(np.float32)
        t_end = min(self.env.t + int(horizon), self.env.T)
        if t_end > self.env.t:
            rentals = self.env.data.rentals[self.env.t:t_end].sum(axis=0).astype(np.float32)
            returns = self.env.data.returns[self.env.t:t_end].sum(axis=0).astype(np.float32)
            capacity = self.env.data.capacity.astype(np.float32)
            bikes = np.clip(bikes + returns - rentals, 0.0, capacity)
        target = self.env.data.capacity.astype(np.float32) * self.env.target_fill_ratio
        truck = self.env.trucks[self.env.current_truck]
        if truck.load == 0:
            scores = bikes - target
        elif truck.load >= self.env.truck_capacity:
            scores = target - bikes
        else:
            scores = np.abs(bikes - target)
        mask = self.env.action_masks()
        scores = scores.astype(np.float32, copy=True)
        scores[~mask] = -np.inf
        best = int(np.argmax(scores))
        if not np.isfinite(scores[best]):
            return int(np.flatnonzero(mask)[0])
        return best


def maybe_wrap_candidate_actions(env, args):
    """CLI 옵션에 따라 env를 candidate Top-K wrapper로 감싼다."""
    top_k = int(getattr(args, "candidate_top_k", 0) or 0)
    if top_k <= 0:
        return env
    return CandidateTopKActionWrapper(
        env,
        top_k=top_k,
        mode=getattr(args, "candidate_mode", "imbalance"),
        travel_coef=getattr(args, "candidate_travel_coef", 0.0),
        zone_mode=getattr(args, "candidate_zone_mode", "none"),
        zone_count=getattr(args, "candidate_zone_count", 3),
        zone_penalty=getattr(args, "candidate_zone_penalty", 0.0),
        feature_mode=getattr(args, "candidate_feature_mode", "none"),
    )
