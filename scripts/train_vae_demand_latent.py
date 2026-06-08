"""정류소별 수요 패턴을 VAE latent feature로 압축하는 스크립트.

입력 데이터:
    data/processed_seoul_all/demand_10min.parquet
    data/processed_seoul_all/stations.parquet

출력 데이터:
    data/vae_latent_by_gu/vae_demand_latent_{구}.parquet

VAE 학습 아이디어:
    같은 정류소의 같은 요일/시간대 수요 통계
    [평균 대여, 평균 반납, 평균 net, 평균 총수요, 표준편차 net]을 입력으로 사용한다.
    VAE encoder가 이 고차원 수요 패턴을 latent vector z로 압축하고,
    RL agent는 z를 state feature로 추가한다.

실행 예:
    PYTHONPATH=. python scripts/train_vae_demand_latent.py --district 강남구
    PYTHONPATH=. python scripts/train_vae_demand_latent.py --district ALL
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


DISTRICTS = [
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


class DemandVAE(nn.Module):
    """수요 통계 벡터를 latent vector로 압축/복원하는 작은 VAE."""

    def __init__(self, input_dim: int, latent_dim: int = 4, hidden: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """입력 수요 패턴에서 latent 평균/분산을 계산한다."""
        h = self.encoder(x)
        return self.mu(h), self.logvar(h).clamp(-8.0, 8.0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = mu + sigma * eps 형태로 sampling해 VAE gradient가 흐르게 한다."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """복원값, latent 평균, latent 로그분산을 반환한다."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


def _slot_of_day(series: pd.Series) -> pd.Series:
    """timestamp를 하루 144개 10분 slot index로 변환한다."""
    return series.dt.hour * 6 + (series.dt.minute // 10)


def build_pattern_table(processed_dir: Path, district: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """구별 정류소/요일/시간대 수요 통계 table을 만든다."""
    stations = pd.read_parquet(processed_dir / "stations.parquet")
    demand = pd.read_parquet(processed_dir / "demand_10min.parquet")
    station_ids = stations.loc[stations["gu"] == district, "station_id"].astype(str).tolist()
    if not station_ids:
        raise ValueError(f"district not found: {district}")

    demand = demand[demand["station_id"].astype(str).isin(station_ids)].copy()
    demand["station_id"] = demand["station_id"].astype(str)
    demand["t"] = pd.to_datetime(demand["t"])
    demand["dow"] = demand["t"].dt.dayofweek.astype(int)
    demand["slot"] = _slot_of_day(demand["t"]).astype(int)
    demand["net"] = demand["returns"].astype(float) - demand["rentals"].astype(float)
    demand["total"] = demand["returns"].astype(float) + demand["rentals"].astype(float)

    grouped = (
        demand.groupby(["station_id", "dow", "slot"], observed=True)
        .agg(
            rental_mean=("rentals", "mean"),
            return_mean=("returns", "mean"),
            net_mean=("net", "mean"),
            total_mean=("total", "mean"),
            net_std=("net", "std"),
        )
        .reset_index()
        .fillna(0.0)
    )

    full_index = pd.MultiIndex.from_product(
        [station_ids, range(7), range(144)],
        names=["station_id", "dow", "slot"],
    )
    grouped = grouped.set_index(["station_id", "dow", "slot"]).reindex(full_index).fillna(0.0).reset_index()

    # 출력 parquet은 실제 demand timestamp별로 생성한다.
    timestamps = demand[["t"]].drop_duplicates().sort_values("t")
    timestamps["dow"] = timestamps["t"].dt.dayofweek.astype(int)
    timestamps["slot"] = _slot_of_day(timestamps["t"]).astype(int)
    return grouped, timestamps


def fit_vae(features: np.ndarray, args: argparse.Namespace) -> tuple[DemandVAE, dict[str, np.ndarray]]:
    """수요 통계 feature를 표준화한 뒤 VAE를 학습한다."""
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-6
    x = ((features - mean) / std).astype(np.float32)

    device = torch.device(args.device)
    model = DemandVAE(x.shape[1], args.latent_dim, args.hidden).to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    pbar = tqdm(range(args.epochs), desc="VAE train", unit="epoch", dynamic_ncols=True) if args.progress else range(args.epochs)
    for epoch in pbar:
        total = 0.0
        count = 0
        for (batch,) in loader:
            batch = batch.to(device)
            recon, mu, logvar = model(batch)

            # VAE loss = reconstruction loss + beta * KL divergence
            # reconstruction은 수요 통계 복원, KL은 latent 분포를 표준정규분포에 가깝게 만든다.
            recon_loss = F.mse_loss(recon, batch, reduction="mean")
            kl_loss = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + args.beta * kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(batch)
            count += len(batch)
        if args.progress:
            pbar.set_postfix(loss=f"{total / max(count, 1):.4f}")
    return model, {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def export_latent(patterns: pd.DataFrame, timestamps: pd.DataFrame, model: DemandVAE, scaler: dict[str, np.ndarray], args: argparse.Namespace, out_path: Path) -> None:
    """학습된 encoder 평균 mu를 timestamp/station_id별 latent parquet으로 저장한다."""
    feature_cols = ["rental_mean", "return_mean", "net_mean", "total_mean", "net_std"]
    features = patterns[feature_cols].to_numpy(dtype=np.float32)
    x = (features - scaler["mean"]) / scaler["std"]
    device = torch.device(args.device)
    with torch.no_grad():
        mu, _ = model.encode(torch.from_numpy(x).to(device))
    latent = mu.cpu().numpy().astype(np.float32)

    latent_cols = [f"vae_z_{i}" for i in range(args.latent_dim)]
    latent_table = patterns[["station_id", "dow", "slot"]].copy()
    for i, col in enumerate(latent_cols):
        latent_table[col] = latent[:, i]

    # 실제 episode timestamp와 매칭하기 쉽도록 t, station_id 단위로 펼친다.
    # timestamp별로 펼치면 파일이 너무 커지므로 compact profile만 저장한다.
    # RL wrapper는 episode timestamp의 요일/slot을 계산해 이 table을 조회한다.
    rows = latent_table[["station_id", "dow", "slot", *latent_cols]].sort_values(["station_id", "dow", "slot"]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(out_path, index=False)
    print(f"saved: {out_path} rows={len(rows):,} latent_dim={args.latent_dim}")


def parse_args() -> argparse.Namespace:
    """VAE latent 생성 CLI 옵션을 정의한다."""
    parser = argparse.ArgumentParser(description="Train VAE demand latent features by district.")
    parser.add_argument("--district", default="강남구", help="구 이름 또는 ALL")
    parser.add_argument("--processed-dir", default="data/processed_seoul_all")
    parser.add_argument("--out-dir", default="data/vae_latent_by_gu")
    parser.add_argument("--latent-dim", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    """선택한 구의 VAE latent parquet을 생성한다."""
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    out_dir = Path(args.out_dir)
    districts = DISTRICTS if args.district.upper() == "ALL" else [args.district]

    for index, district in enumerate(districts, start=1):
        print(f"\n[{index}/{len(districts)}] {district} VAE latent 생성")
        patterns, timestamps = build_pattern_table(processed_dir, district)
        feature_cols = ["rental_mean", "return_mean", "net_mean", "total_mean", "net_std"]
        model, scaler = fit_vae(patterns[feature_cols].to_numpy(dtype=np.float32), args)
        export_latent(
            patterns,
            timestamps,
            model,
            scaler,
            args,
            out_dir / f"vae_demand_latent_{district}.parquet",
        )


if __name__ == "__main__":
    main()
