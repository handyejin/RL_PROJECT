"""서울 25개 구별 A2C Top-K no-BC 확장 실험 runner.

목적:
    마포구에서 검증한 forecast + Top-K + 3권역 후보 action 구조를
    서울 25개 구에 독립적으로 적용해, 구별 성능 차이를 비교한다.

진행 방식:
    1. 각 구별 1시간 수요예측 parquet 생성
    2. 같은 구에서 A2C Top-K Plus no-BC full 학습
    3. history.npy와 로그를 읽어 summary csv/md 생성

진행 확인:
    subprocess stdout을 즉시 flush하므로 Codex 화면에서
    [구 번호/25], [1/5] forecast 단계, episode별 eval 로그를 볼 수 있다.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DISTRICTS = [
    "강남구",
    "강동구",
    "강북구",
    "강서구",
    "관악구",
    "광진구",
    "구로구",
    "금천구",
    "노원구",
    "도봉구",
    "동대문구",
    "동작구",
    "마포구",
    "서대문구",
    "서초구",
    "성동구",
    "성북구",
    "송파구",
    "양천구",
    "영등포구",
    "용산구",
    "은평구",
    "종로구",
    "중구",
    "중랑구",
]


def safe_name(text: str) -> str:
    """로그/파일명에 쓰기 편한 구 이름을 만든다."""
    return text.replace(" ", "_")


def run_streaming(cmd: list[str], log_path: Path, env: dict[str, str] | None = None) -> int:
    """명령을 실행하면서 stdout/stderr를 화면과 log 파일에 동시에 남긴다."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return proc.wait()


def parse_training_log(path: Path) -> dict[str, float | str]:
    """학습 로그에서 baseline/final 같은 핵심 숫자를 추출한다."""
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    out: dict[str, float | str] = {}
    m = re.search(r"heuristic mean reward:\s*([-0-9.]+)", text)
    if m:
        out["baseline_avg_reward"] = float(m.group(1))
    m = re.search(r"best reward:\s*([-0-9.]+)\s+at episode\s+([0-9]+)", text)
    if m:
        out["best_avg_reward_log"] = float(m.group(1))
        out["best_episode_log"] = float(m.group(2))
    m = re.search(r"final reward:\s*([-0-9.]+)", text)
    if m:
        out["final_avg_reward_log"] = float(m.group(1))
    return out


def read_history(log_dir: Path) -> dict[str, float]:
    """history.npy에서 best/final eval reward를 읽는다."""
    path = log_dir / "history.npy"
    if not path.exists():
        return {}
    rows = np.load(path, allow_pickle=True).tolist()
    if not isinstance(rows, list):
        rows = list(rows)
    eval_rows = [r for r in rows if isinstance(r, dict) and "eval_reward" in r]
    if not eval_rows:
        return {}
    rewards = [float(r["eval_reward"]) for r in eval_rows]
    best_idx = int(np.argmax(rewards))
    best_row = eval_rows[best_idx]
    final_row = eval_rows[-1]
    return {
        "best_avg_reward": float(rewards[best_idx]),
        "best_episode": float(best_row.get("episode", 0)),
        "final_avg_reward": float(final_row["eval_reward"]),
    }


def district_profile(processed_dir: Path) -> dict[str, dict[str, int]]:
    """구별 전체/active 정류소 수를 계산한다."""
    stations = pd.read_parquet(processed_dir / "stations.parquet")
    demand = pd.read_parquet(processed_dir / "demand_10min.parquet", columns=["station_id"])
    active = set(demand["station_id"].dropna().unique())
    profiles: dict[str, dict[str, int]] = {}
    for gu, group in stations.groupby("gu"):
        ids = set(group["station_id"].dropna())
        profiles[str(gu)] = {
            "station_count": int(len(ids)),
            "active_station_count": int(len(ids & active)),
        }
    return profiles


def write_summary(rows: list[dict], csv_path: Path, md_path: Path) -> None:
    """구별 결과 summary를 CSV와 Markdown으로 저장한다."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "district",
        "station_count",
        "active_station_count",
        "forecast_status",
        "train_status",
        "baseline_avg_reward",
        "best_avg_reward",
        "final_avg_reward",
        "best_delta_vs_baseline",
        "final_delta_vs_baseline",
        "best_episode",
        "forecast_mae_rent",
        "forecast_mae_return",
        "train_log",
        "forecast_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    lines = [
        "# 서울 25개 구별 A2C Top-K no-BC 확장 실험",
        "",
        "각 구를 독립적인 재배치 환경으로 보고, `forecast_projected_travel + Top-K 후보 action + 3개 권역 penalty`를 적용한 A2C no-BC 결과이다.",
        "Reward는 음수이므로 0에 가까울수록 좋고, Delta는 같은 구의 MostImbalanced baseline 대비 개선폭이다.",
        "",
        "| 구 | 정류소 | Active | Baseline | Best | Final | Best Delta | Final Delta | Best Ep | 상태 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        def fmt(value):
            if value == "" or value is None:
                return ""
            try:
                return f"{float(value):.1f}"
            except (TypeError, ValueError):
                return str(value)

        lines.append(
            "| {district} | {station_count} | {active_station_count} | {baseline} | {best} | {final} | {best_delta} | {final_delta} | {best_ep} | {status} |".format(
                district=row.get("district", ""),
                station_count=row.get("station_count", ""),
                active_station_count=row.get("active_station_count", ""),
                baseline=fmt(row.get("baseline_avg_reward")),
                best=fmt(row.get("best_avg_reward")),
                final=fmt(row.get("final_avg_reward")),
                best_delta=fmt(row.get("best_delta_vs_baseline")),
                final_delta=fmt(row.get("final_delta_vs_baseline")),
                best_ep=fmt(row.get("best_episode")),
                status=row.get("train_status", ""),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """구별 batch 실행 옵션을 정의한다."""
    parser = argparse.ArgumentParser(description="Run gu-level A2C Top-K no-BC full experiments.")
    parser.add_argument("--processed-dir", default="data/processed_seoul_all")
    parser.add_argument("--forecast-dir", default="data/forecast_by_gu")
    parser.add_argument("--run-tag", default=f"gu_a2c_topk_no_bc_{dt.date.today().isoformat()}")
    parser.add_argument("--districts", nargs="*", default=DEFAULT_DISTRICTS)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--n-train-dates", type=int, default=200)
    parser.add_argument("--forecast-max-train-rows", type=int, default=500_000)
    parser.add_argument("--forecast-max-eval-rows", type=int, default=200_000)
    parser.add_argument("--forecast-max-iter", type=int, default=140)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "mps"])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--summary-csv", default="")
    parser.add_argument("--summary-md", default="")
    return parser.parse_args()


def main() -> None:
    """25개 구별 forecast/A2C 학습을 순차 실행한다."""
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    forecast_dir = Path(args.forecast_dir)
    log_root = Path("logs") / args.run_tag
    forecast_dir.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.summary_csv or f"docs/{args.run_tag}_summary.csv")
    md_path = Path(args.summary_md or f"docs/{args.run_tag}_summary.md")
    profiles = district_profile(processed_dir)
    rows: list[dict] = []

    env = dict(**dict(os_environ()))
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = "."

    print(f"=== Gu-level A2C Top-K no-BC batch | districts={len(args.districts)} ===", flush=True)
    print(f"processed_dir={processed_dir}", flush=True)
    print(f"forecast_dir={forecast_dir}", flush=True)
    print(f"log_root={log_root}", flush=True)
    print(f"summary_csv={csv_path}", flush=True)
    print(f"summary_md={md_path}", flush=True)

    for idx, district in enumerate(args.districts, start=1):
        started = time.time()
        name = safe_name(district)
        forecast_path = forecast_dir / f"demand_forecast_1h_{name}.parquet"
        model_path = forecast_dir / f"demand_forecast_1h_{name}.joblib"
        metrics_path = forecast_dir / f"demand_forecast_1h_{name}_metrics.json"
        forecast_log = log_root / f"{idx:02d}_{name}_forecast.log"
        train_log = log_root / f"{idx:02d}_{name}_a2c.log"
        train_tag = f"{args.run_tag}_{name}_a2c"
        train_dir = Path("logs") / f"actor_critic_{train_tag}"

        row = {
            "district": district,
            **profiles.get(district, {}),
            "forecast_path": str(forecast_path),
            "train_log": str(train_log),
        }
        print("\n" + "=" * 80, flush=True)
        print(f"[{idx}/{len(args.districts)}] {district} 시작", flush=True)

        if not (args.skip_existing and forecast_path.exists()):
            forecast_cmd = [
                sys.executable,
                "scripts/train_demand_forecast.py",
                "--processed-dir",
                str(processed_dir),
                "--district",
                district,
                "--max-train-rows",
                str(args.forecast_max_train_rows),
                "--max-eval-rows",
                str(args.forecast_max_eval_rows),
                "--max-iter",
                str(args.forecast_max_iter),
                "--model-out",
                str(model_path),
                "--forecast-out",
                str(forecast_path),
                "--metrics-out",
                str(metrics_path),
            ]
            print(f"[{idx}/{len(args.districts)}] forecast 생성: {forecast_path}", flush=True)
            code = run_streaming(forecast_cmd, forecast_log, env)
            row["forecast_status"] = "ok" if code == 0 else f"failed:{code}"
            if code != 0:
                rows.append(row)
                write_summary(rows, csv_path, md_path)
                continue
        else:
            row["forecast_status"] = "skipped"
            print(f"[{idx}/{len(args.districts)}] forecast skip existing", flush=True)

        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            row["forecast_mae_rent"] = metrics.get("model_rent_mae", "")
            row["forecast_mae_return"] = metrics.get("model_return_mae", "")

        train_cmd = [
            sys.executable,
            "-m",
            "src.agents.ours.experiments.a2c_topk_forecast_plus",
            "--processed-dir",
            str(processed_dir),
            "--district",
            district,
            "--forecast-path",
            str(forecast_path),
            "--episodes",
            str(args.episodes),
            "--eval-every",
            str(args.eval_every),
            "--n-train-dates",
            str(args.n_train_dates),
            "--bc-epochs",
            "0",
            "--bc-val-dates",
            "0",
            "--anchor-coef",
            "0.0",
            "--device",
            args.device,
            "--tag",
            train_tag,
        ]
        print(f"[{idx}/{len(args.districts)}] A2C no-BC 학습 시작: tag={train_tag}", flush=True)
        code = run_streaming(train_cmd, train_log, env)
        row["train_status"] = "ok" if code == 0 else f"failed:{code}"

        row.update(parse_training_log(train_log))
        row.update(read_history(train_dir))
        baseline = row.get("baseline_avg_reward", "")
        best = row.get("best_avg_reward", row.get("best_avg_reward_log", ""))
        final = row.get("final_avg_reward", row.get("final_avg_reward_log", ""))
        row["best_avg_reward"] = best
        row["final_avg_reward"] = final
        if baseline != "" and best != "":
            row["best_delta_vs_baseline"] = float(best) - float(baseline)
        if baseline != "" and final != "":
            row["final_delta_vs_baseline"] = float(final) - float(baseline)

        rows.append(row)
        write_summary(rows, csv_path, md_path)
        elapsed = time.time() - started
        print(
            f"[{idx}/{len(args.districts)}] {district} 완료: "
            f"best={row.get('best_avg_reward', '')}, final={row.get('final_avg_reward', '')}, "
            f"elapsed={elapsed/60:.1f}min",
            flush=True,
        )

    write_summary(rows, csv_path, md_path)
    print("\n=== batch complete ===", flush=True)
    print(f"summary_csv={csv_path}", flush=True)
    print(f"summary_md={md_path}", flush=True)


def os_environ() -> dict[str, str]:
    """subprocess에 넘길 현재 환경변수를 반환한다."""
    import os

    return dict(os.environ)


if __name__ == "__main__":
    main()
