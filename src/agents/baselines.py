"""베이스라인 정책 (Phase 3).

RL과 비교할 비학습 정책. 환경 내부 상태(env.bikes, env.trucks 등)를 직접 보고 결정.

- NoopPolicy: 트럭이 자기 위치에 머무름 (재배치 없음)
- MostImbalancedPolicy: 트럭 적재량에 따라 가장 균형 어긋난 정류소로 이동
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.envs.rebalance_env import RebalanceEnv


class BasePolicy(ABC):
    name: str = "base"

    @abstractmethod
    def act(self, env: RebalanceEnv) -> int:
        """현재 결정 트럭(env.current_truck)이 갈 정류소 idx 반환."""
        ...


class NoopPolicy(BasePolicy):
    """현재 위치에 머무름."""

    name = "noop"

    def act(self, env: RebalanceEnv) -> int:
        return env.trucks[env.current_truck].location


class MostImbalancedPolicy(BasePolicy):
    """탐욕적 균형 정책.

    - 트럭 비어있음(load==0): 가장 잉여(bikes - target)가 큰 정류소로 → 적재
    - 트럭 가득(load==capacity): 가장 부족(target - bikes)이 큰 정류소로 → 하차
    - 부분 적재: 절대 불균형(|bikes - target|) 가장 큰 정류소로

    다른 트럭의 목적지는 제외하여 중복 이동 방지.
    """

    name = "most_imbalanced"

    def act(self, env: RebalanceEnv) -> int:
        truck = env.trucks[env.current_truck]
        bikes = env.bikes.astype(np.float32)
        target = env.data.capacity.astype(np.float32) * env.target_fill_ratio

        if truck.load == 0:
            scores = bikes - target  # 잉여 큰 곳일수록 높음
        elif truck.load >= env.truck_capacity:
            scores = target - bikes  # 부족 큰 곳일수록 높음
        else:
            scores = np.abs(bikes - target).astype(np.float32)

        # 자기 위치 + 다른 트럭 목적지 제외
        scores = scores.copy()
        scores[truck.location] = -np.inf
        for i, other in enumerate(env.trucks):
            if i == env.current_truck:
                continue
            if not other.is_idle:
                scores[other.destination] = -np.inf

        best = int(np.argmax(scores))
        # 모든 후보가 -inf면 (이론상 거의 없음) 자기 위치 머무름
        if not np.isfinite(scores[best]):
            return truck.location
        return best


class PredictiveImbalancedPolicy(BasePolicy):
    """예측형 균형 정책 — 현재가 아니라 *미래 H스텝 후* 예상 상태로 결정.

    반응형(MostImbalanced)은 현재 불균형만 보고 움직여 오전 수요 ramp에 늦는다
    (진단: 격차의 100%가 stockout/full). 예측형은 향후 H스텝 net demand를 더해
    "곧 비거나 곧 꽉 찰" 정류소를 미리 공략한다.

      predicted = bikes + Σ_{t..t+H}(returns - rentals)   # 개입 없을 때 도달할 상태
      - 빈 트럭: predicted 잉여 큰 곳(곧 넘침) → 수거
      - 가득 트럭: predicted 부족 큰 곳(곧 소진) → 배달
      - 부분:     |predicted - target| 큰 곳

    다른 트럭 목적지 제외는 반응형과 동일. horizon은 트럭 이동시간 정도(기본 6=60분).
    """

    name = "predictive_imbalanced"

    def __init__(self, horizon: int = 3):   # H=3(30분 선행)이 7일 eval 최적 (-382.79)
        self.horizon = int(horizon)

    def act(self, env: RebalanceEnv) -> int:
        truck = env.trucks[env.current_truck]
        target = env.data.capacity.astype(np.float32) * env.target_fill_ratio

        t = env.t
        T = env.data.rentals.shape[0]
        t_end = min(t + self.horizon, T)
        future_rent = env.data.rentals[t:t_end].sum(axis=0).astype(np.float32)
        future_ret = env.data.returns[t:t_end].sum(axis=0).astype(np.float32)
        predicted = env.bikes.astype(np.float32) + (future_ret - future_rent)

        if truck.load == 0:
            scores = predicted - target          # 곧 넘칠 곳 → 수거
        elif truck.load >= env.truck_capacity:
            scores = target - predicted          # 곧 소진될 곳 → 배달
        else:
            scores = np.abs(predicted - target).astype(np.float32)

        # 자기 위치 + 다른 트럭 목적지 제외 (반응형과 동일 조정)
        scores = scores.copy()
        scores[truck.location] = -np.inf
        for i, other in enumerate(env.trucks):
            if i == env.current_truck:
                continue
            if not other.is_idle:
                scores[other.destination] = -np.inf

        best = int(np.argmax(scores))
        if not np.isfinite(scores[best]):
            return truck.location
        return best


class ForecastErrorPolicy(BasePolicy):
    """예측오차 보정 예측형 — 서영현(2020) RTDP/ADP 논문의 핵심 아이디어 ①.

    forecast 예측형은 과거평균(gr/ge)만 믿어 *오늘의 편차*에 둔감 → -459에서 정체
    (oracle 상한 -383). 이 정책은 운영자가 실시간 관측하는 "예측오차"
    (관측수요 − forecast)에 빠르게 대응한다:

      1) 최근 W스텝 관측 net demand vs forecast net demand → 정류소별 *드리프트*(편차율)
      2) 그 드리프트를 앞으로 H스텝에 투영해 forecast를 보정 → 보정 예측 상태
      3) 보정 예측의 불균형으로 점수화 (예측형과 동일)

    관측(env.data)·forecast(self.pr/pre)만 사용 → oracle 미래 누설 없음(배포 가능).

    mode:
      "drift" — 가산 보정: future += (관측−forecast)/W × H  (정류소별 편차, 추천)
      "scale" — 승산 보정: future *= clip(관측합/forecast합)  (전역 busy-day 스케일)
    error_focus=True면 |예측오차| 큰 상위 focus_k 정류소만 후보로(논문의 탐색 절감).
    """

    name = "forecast_error"

    def __init__(self, profile_rent, profile_ret, horizon: int = 3, window: int = 6,
                 mode: str = "drift", alpha: float = 1.0,
                 error_focus: bool = False, focus_k: int | None = None):
        self.pr = profile_rent.astype(np.float32)    # forecast 대여 (T,N)
        self.pre = profile_ret.astype(np.float32)    # forecast 반납 (T,N)
        self.horizon = int(horizon)
        self.window = int(window)
        self.mode = str(mode)
        self.alpha = float(alpha)                     # 보정 신뢰도(0=forecast 그대로, 1=완전보정)
        self.error_focus = bool(error_focus)
        self.focus_k = focus_k

    def act(self, env: RebalanceEnv) -> int:
        truck = env.trucks[env.current_truck]
        target = env.data.capacity.astype(np.float32) * env.target_fill_ratio
        t, T = env.t, self.pr.shape[0]
        H, W = self.horizon, self.window
        t_end = min(t + H, T)
        steps_ahead = max(t_end - t, 0)

        # 미래 forecast 기준선
        fr = self.pr[t:t_end].sum(axis=0).astype(np.float32)
        fe = self.pre[t:t_end].sum(axis=0).astype(np.float32)

        # 최근 W스텝 관측 vs forecast → 예측오차
        w0 = max(t - W, 0)
        n_obs = t - w0
        err_mag = np.zeros(env.N, dtype=np.float32)
        if n_obs > 0 and steps_ahead > 0:
            obs_rent = env.data.rentals[w0:t].sum(axis=0).astype(np.float32)
            obs_ret = env.data.returns[w0:t].sum(axis=0).astype(np.float32)
            fc_rent = self.pr[w0:t].sum(axis=0).astype(np.float32)
            fc_ret = self.pre[w0:t].sum(axis=0).astype(np.float32)
            if self.mode == "scale":
                eps = 1e-3
                ratio_r = 1.0 + self.alpha * (np.clip(obs_rent / np.maximum(fc_rent, eps), 0.2, 5.0) - 1.0)
                ratio_e = 1.0 + self.alpha * (np.clip(obs_ret / np.maximum(fc_ret, eps), 0.2, 5.0) - 1.0)
                fr = fr * ratio_r
                fe = fe * ratio_e
                err_mag = np.abs(obs_rent - fc_rent) + np.abs(obs_ret - fc_ret)
            else:  # drift (가산)
                rent_rate = (obs_rent - fc_rent) / n_obs
                ret_rate = (obs_ret - fc_ret) / n_obs
                fr = np.maximum(fr + self.alpha * rent_rate * steps_ahead, 0.0)
                fe = np.maximum(fe + self.alpha * ret_rate * steps_ahead, 0.0)
                err_mag = np.abs(rent_rate) + np.abs(ret_rate)

        predicted = env.bikes.astype(np.float32) + (fe - fr)

        if truck.load == 0:
            scores = predicted - target
        elif truck.load >= env.truck_capacity:
            scores = target - predicted
        else:
            scores = np.abs(predicted - target).astype(np.float32)

        scores = scores.copy()
        # 예측오차 큰 정류소만 후보로(논문: 전체탐색과 성능 유사 + 계산 절감)
        if self.error_focus and self.focus_k is not None and n_obs > 0:
            keep = np.zeros(env.N, dtype=bool)
            keep[np.argsort(err_mag)[-int(self.focus_k):]] = True
            scores[~keep] = -np.inf

        scores[truck.location] = -np.inf
        for i, other in enumerate(env.trucks):
            if i == env.current_truck:
                continue
            if not other.is_idle:
                scores[other.destination] = -np.inf

        best = int(np.argmax(scores))
        if not np.isfinite(scores[best]):
            return int(truck.location)
        return best


class ConstantIntentPolicy(BasePolicy):
    """항상 같은 (추상) action index 반환 — DQfD warm-start demo용.

    추상 action 공간에서 "항상 predictive 의도(index 5)"를 모방시키면
    출발 정책 = 예측형. 상수라 BC clone이 자명(146지선다 clone 실패 회피).
    """
    name = "constant_intent"

    def __init__(self, idx: int = 5):
        self.idx = int(idx)

    def act(self, env) -> int:  # env 무시, 고정 의도
        return self.idx


POLICY_REGISTRY: dict[str, type[BasePolicy]] = {
    NoopPolicy.name: NoopPolicy,
    MostImbalancedPolicy.name: MostImbalancedPolicy,
    PredictiveImbalancedPolicy.name: PredictiveImbalancedPolicy,
}


def get_policy(name: str) -> BasePolicy:
    if name not in POLICY_REGISTRY:
        raise ValueError(f"unknown policy '{name}'. choices: {list(POLICY_REGISTRY)}")
    return POLICY_REGISTRY[name]()
