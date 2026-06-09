"""저장된 dqn_small 모델을 holdout 73일 전체로 post-hoc 재평가한다.

배경:
    기존 학습/평가는 holdout 73일 중 7일만 평가에 썼고, best 체크포인트를 그 7일로
    골라 낙관 편향이 있었다. 보고서용으로는 (1) 표본 확대(73일 전체), (2) best가 아닌
    저장된 모델 고정 평가, (3) 분포(평균±95% CI)+추월일수 제시가 더 정직하다.

    재학습 없이 저장 모델을 로드해 평가만 다시 하므로 빠르다(구당 ~1-2분).

방법:
    - 정류소 선택은 학습 로그의 "선택 정류소 id: [...]" 줄을 파싱해 그대로 재현
      (train 200일 재로딩 불필요).
    - dqn_small_core의 evaluate / evaluate_heuristic / make_env / 오버라이드 헬퍼를 재사용.
    - model·heuristic이 같은 seed(=같은 실현·트럭 시작)를 공유하는 paired 평가.

사용:
    PYTHONPATH=. python scripts/posthoc_eval_holdout.py \
        --model-tag-prefix small400 --stations-log-dir logs/runs/small400 \
        --demand-noise none --out logs/runs/posthoc_det400.csv
    PYTHONPATH=. python scripts/posthoc_eval_holdout.py \
        --model-tag-prefix small400p --stations-log-dir logs/runs/small400_poisson \
        --demand-noise poisson --eval-samples 20 --districts 강남구 마포구 도봉구 \
        --out logs/runs/posthoc_poisson400.csv
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.algorithms.dqn_small import core as C  # noqa: E402
from src.agents.masked_dqn import MaskableDQN  # noqa: E402
from src.envs.data_loader import load_episode  # noqa: E402


# 평가에 쓸 holdout 날짜: dqn_small_core와 동일 분할에서 7일이 아니라 전체 73일.
HOLDOUT_DATES = sorted(C.ALL_DATES[C.N_TRAIN:])


ALL_DISTRICTS = [
    "강남구", "강동구", "강북구", "강서구", "관악구", "광진구", "구로구", "금천구",
    "노원구", "도봉구", "동대문구", "동작구", "마포구", "서대문구", "서초구", "성동구",
    "성북구", "송파구", "양천구", "영등포구", "용산구", "은평구", "종로구", "중구", "중랑구",
]


def parse_station_ids(log_path: Path) -> list[str] | None:
    """학습 로그의 '선택 정류소 id: [...]' 줄에서 정류소 id 목록을 파싱한다."""
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"선택 정류소 id:\s*(\[[^\]]*\])", text)
    if not m:
        return None
    return ast.literal_eval(m.group(1))


def subset_by_ids(ep, ids: list[str]):
    """정류소 id 목록(학습 시 순서)대로 episode를 subset한다."""
    pos = {sid: i for i, sid in enumerate(ep.station_ids)}
    idx = [pos[s] for s in ids if s in pos]
    if len(idx) != len(ids):
        missing = [s for s in ids if s not in pos]
        raise ValueError(f"매칭 안 된 정류소: {missing}")
    return C.subset_episode(ep, np.array(idx, dtype=int))


def build_args(a: argparse.Namespace) -> SimpleNamespace:
    """dqn_small_core 평가 헬퍼가 기대하는 args 네임스페이스를 학습 설정대로 만든다."""
    return SimpleNamespace(
        n_trucks=1,
        demand_noise=a.demand_noise,
        demand_rate_scale=1.0,
        eval_samples=a.eval_samples,
        future_mode="forecast_projected_travel",
        future_horizon=6,
        candidate_top_k=0,
        candidate_mode="forecast_imbalance",
        candidate_travel_coef=0.0,
        candidate_zone_mode="none",
        candidate_zone_count=3,
        candidate_zone_penalty=0.0,
        candidate_feature_mode="none",
        agent_shaping_mode="projected_imbalance",
        agent_shaping_scale=0.0,
        agent_shaping_gamma=0.99,
    )


def ci95(values: list[float]) -> float:
    """표본 평균의 95% 신뢰구간 반폭(1.96·s/√n)."""
    n = len(values)
    if n < 2:
        return 0.0
    return 1.96 * float(np.std(values, ddof=1)) / np.sqrt(n)


def evaluate_district(gu: str, a: argparse.Namespace, args_ns: SimpleNamespace, env_kw: dict) -> dict | None:
    """한 자치구: 저장 모델을 holdout 73일로 평가해 휴리스틱 대비 Δ 통계를 낸다."""
    log_path = Path(a.stations_log_dir) / f"{gu}.log"
    ids = parse_station_ids(log_path)
    if ids is None:
        print(f"[{gu}] 정류소 id 로그 없음 → 건너뜀 ({log_path})")
        return None
    model_dir = PROJECT_ROOT / "logs" / f"dqn_{a.model_tag_prefix}_{gu}"
    model_path = model_dir / f"{a.model_kind}_model.zip"
    if not model_path.exists():
        print(f"[{gu}] 모델 없음 → 건너뜀 ({model_path})")
        return None

    eval_eps = [
        load_episode(a.processed_dir, district=gu, episode_start=f"{d} 00:00")
        for d in HOLDOUT_DATES
    ]
    eval_eps = [subset_by_ids(ep, ids) for ep in eval_eps]
    C.apply_capacity_override(eval_eps, a.capacity_path, 0.5)
    forecast_path = f"data/forecast_by_gu/demand_forecast_1h_{gu}.parquet"
    C.attach_forecast_override(eval_eps, forecast_path)

    model = MaskableDQN.load(str(model_path))
    h_mean, h_rewards = C.evaluate_heuristic(eval_eps, args_ns, env_kw, a.seed)
    m_mean, m_rewards = C.evaluate(model, eval_eps, args_ns, env_kw, a.seed)

    deltas = [m - h for m, h in zip(m_rewards, h_rewards)]
    wins = sum(1 for d in deltas if d > 0)
    return {
        "gu": gu,
        "n_days": len(deltas),
        "heur": h_mean,
        "model": m_mean,
        "delta": float(np.mean(deltas)),
        "delta_ci95": ci95(deltas),
        "win_days": wins,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="저장 모델을 holdout 73일로 post-hoc 재평가")
    p.add_argument("--model-tag-prefix", default="small400", help="모델 디렉토리 prefix → logs/dqn_<prefix>_<구>")
    p.add_argument("--stations-log-dir", default="logs/runs/small400", help="선택 정류소 id가 있는 학습 로그 디렉토리")
    p.add_argument("--model-kind", choices=["final", "best"], default="final")
    p.add_argument("--demand-noise", choices=["none", "poisson"], default="none")
    p.add_argument("--eval-samples", type=int, default=1, help="poisson일 때 날짜별 실현 수")
    p.add_argument("--processed-dir", default="data/processed_seoul_all")
    p.add_argument("--capacity-path", default="data/processed/station_capacity.csv")
    p.add_argument("--districts", nargs="*", default=ALL_DISTRICTS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="logs/runs/posthoc_eval.csv")
    a = p.parse_args()

    args_ns = build_args(a)
    env_kw = C.build_env_kw(args_ns)
    print(f"holdout 평가일 수: {len(HOLDOUT_DATES)}일, demand_noise={a.demand_noise}, "
          f"eval_samples={a.eval_samples}, model={a.model_kind}")

    rows = []
    for gu in a.districts:
        t0 = time.time()
        r = evaluate_district(gu, a, args_ns, env_kw)
        if r is None:
            continue
        rows.append(r)
        print(f"[{gu}] 휴={r['heur']:.1f}  모델={r['model']:.1f}  "
              f"Δ={r['delta']:+.1f}±{r['delta_ci95']:.1f}  추월일={r['win_days']}/{r['n_days']}  "
              f"({time.time()-t0:.0f}s)")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("gu,n_days,heuristic,model,delta,delta_ci95,win_days\n")
        for r in rows:
            f.write(f"{r['gu']},{r['n_days']},{r['heur']:.2f},{r['model']:.2f},"
                    f"{r['delta']:.2f},{r['delta_ci95']:.2f},{r['win_days']}\n")

    if rows:
        n = len(rows)
        mean_d = float(np.mean([r["delta"] for r in rows]))
        wins = sum(1 for r in rows if r["delta"] > 0)
        print(f"\n=== 요약 ({n}구) ===")
        print(f"추월 구: {wins}/{n}  |  구 평균 Δ: {mean_d:+.1f}")
        print(f"CSV 저장: {out}")


if __name__ == "__main__":
    main()
