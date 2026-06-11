#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DQN 실험 종합 정리용 시각화 생성."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"

POS = "#128a4a"
NEG = "#c0392b"
ACC = "#2563eb"
OUT = "/Users/son-yejin/projects/rl_project/docs/figures/"


def signed_colors(vals):
    return [POS if v > 0 else NEG for v in vals]


def fig_seed():
    gus = ["강남", "송파", "성북", "구로", "강서", "영등포"]
    s42 = [10.9, 2.6, -0.8, -32.5, -30.1, -27.7]
    s123 = [7.1, -3.7, -1.7, -17.2, -39.0, -28.8]
    s777 = [6.0, 8.9, -2.6, -18.0, -45.7, -31.9]
    mean = [(a + b + c) / 3 for a, b, c in zip(s42, s123, s777)]
    x = np.arange(len(gus))
    w = 0.2
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - 1.5 * w, s42, w, label="seed 42", color="#90c2f0")
    ax.bar(x - 0.5 * w, s123, w, label="seed 123", color="#5b9bd5")
    ax.bar(x + 0.5 * w, s777, w, label="seed 777", color="#2563eb")
    ax.bar(x + 1.5 * w, mean, w, label="평균", color="#15233f")
    ax.axhline(0, color="#888", lw=0.8)
    # best3(시드42 상위) / worst3(하위) 그룹 구분
    ax.axvline(2.5, color="#444", lw=1.0, ls="-")
    top = max(max(s) for s in (s42, s123, s777)) + 2
    bot = min(min(s) for s in (s42, s123, s777)) - 2
    ax.axvspan(-0.5, 2.5, color=POS, alpha=0.05)
    ax.axvspan(2.5, 5.5, color=NEG, alpha=0.05)
    ax.text(1.0, top, "BEST 3 (시드42 상위)", ha="center", va="top",
            fontsize=10, fontweight="bold", color=POS)
    ax.text(4.0, top, "WORST 3 (시드42 하위)", ha="center", va="top",
            fontsize=10, fontweight="bold", color=NEG)
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(bot, top + 1)
    ax.set_xticks(x)
    ax.set_xticklabels(gus)
    ax.set_ylabel("Δ (DQN - 휴리스틱)")
    ax.set_title("시드 민감도 (k15, 42/123/777) — best3 vs worst3, 견고한 추월은 강남뿐",
                 fontweight="bold")
    ax.legend(ncol=4, fontsize=9, loc="lower center")
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.savefig(OUT + "dqn_seed_sensitivity.png")
    plt.close(fig)


def fig_stations():
    k = [8, 10, 12, 15]
    gn = [-5.5, -1.1, 5.4, 10.9]
    yd = [-20.8, -23.8, -17.9, -27.7]
    gs = [-43.3, -38.3, -55.9, -30.1]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(k, gn, "o-", color=POS, lw=2.2, label="강남 (추월)")
    ax.plot(k, yd, "s--", color="#e08e0b", lw=1.8, label="영등포")
    ax.plot(k, gs, "^--", color=NEG, lw=1.8, label="강서")
    ax.axhline(0, color="#888", lw=0.8)
    ax.fill_between(k, 0, max(gn) + 3, color=POS, alpha=0.05)
    ax.set_xticks(k)
    ax.set_xlabel("정류소 수 (top-k)")
    ax.set_ylabel("Δ (DQN - 휴리스틱)")
    ax.set_title("정류소 수 스윕 — 강남은 k↑일수록 추월(단조)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(ls=":", alpha=0.4)
    fig.savefig(OUT + "dqn_station_sweep.png")
    plt.close(fig)


def fig_trucks():
    t = [1, 2, 3]
    gn = [10.9, -4.0, -9.6]
    yd = [-27.7, -54.9, -79.0]
    gs = [-30.1, -115.0, -162.6]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(t, gn, "o-", color=POS, lw=2.2, label="강남")
    ax.plot(t, yd, "s--", color="#e08e0b", lw=1.8, label="영등포")
    ax.plot(t, gs, "^--", color=NEG, lw=1.8, label="강서")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xticks(t)
    ax.set_xlabel("트럭 수")
    ax.set_ylabel("Δ (DQN - 휴리스틱)")
    ax.set_title("트럭 수 ↑ — 멀티트럭 협응 실패로 전부 악화", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(ls=":", alpha=0.4)
    fig.savefig(OUT + "dqn_truck_count.png")
    plt.close(fig)


def fig_25gu():
    data = [
        ("송파", 11.1), ("강남", 8.3), ("성북", 3.3), ("은평", 1.8), ("중구", 1.6),
        ("중랑", 1.2), ("강동", -0.6), ("도봉", -1.5), ("마포", -2.4), ("성동", -2.4),
        ("동작", -3.3), ("서초", -5.3), ("서대문", -5.9), ("관악", -6.5), ("종로", -6.6),
        ("강북", -6.8), ("구로", -7.1), ("노원", -8.5), ("용산", -9.0), ("금천", -9.6),
        ("양천", -12.9), ("광진", -13.2), ("동대문", -13.3), ("영등포", -22.0), ("강서", -22.3),
    ]
    names = [d[0] for d in data][::-1]
    vals = [d[1] for d in data][::-1]
    fig, ax = plt.subplots(figsize=(7.5, 8))
    ax.barh(names, vals, color=signed_colors(vals))
    ax.axvline(0, color="#444", lw=1)
    for i, v in enumerate(vals):
        ax.text(v + (0.3 if v >= 0 else -0.3), i, f"{v:+.1f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=8)
    ax.set_xlabel("Δ (DQN-small - 휴리스틱), holdout 73일")
    ax.set_title("서울 25개 자치구 추월 분포 — 추월 6/25 (견고 3)", fontweight="bold")
    ax.grid(axis="x", ls=":", alpha=0.4)
    fig.savefig(OUT + "dqn_25gu_delta.png")
    plt.close(fig)


def fig_algo():
    algos = ["A2C", "REINFORCE", "PPO", "BANDIT", "DQN"]
    win = [17, 13, 13, 7, 2]
    mean_best = [16.85, -0.75, -3.51, -33.96, -19.12]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    a1.bar(algos, win, color=["#15233f" if a != "DQN" else NEG for a in algos])
    a1.set_ylabel("best 추월 구 수 (/25)")
    a1.set_title("추월한 자치구 수", fontweight="bold")
    a1.grid(axis="y", ls=":", alpha=0.4)
    for i, v in enumerate(win):
        a1.text(i, v + 0.3, str(v), ha="center", fontsize=9)
    a2.bar(algos, mean_best, color=signed_colors(mean_best))
    a2.axhline(0, color="#888", lw=0.8)
    a2.set_ylabel("mean best Δ")
    a2.set_title("평균 best Δ", fontweight="bold")
    a2.grid(axis="y", ls=":", alpha=0.4)
    for i, v in enumerate(mean_best):
        a2.text(i, v + (0.6 if v >= 0 else -0.6), f"{v:+.1f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    fig.suptitle("전체환경 25구 알고리즘 비교 — DQN 최하위", fontweight="bold", y=1.03)
    fig.savefig(OUT + "dqn_algo_compare.png")
    plt.close(fig)


def fig_levers():
    levers = ["baseline", "BC v2", "h12·400k", "PBRS s10", "2트럭"]
    gn = [10.9, -0.2, 8.4, 10.4, -4.0]
    yd = [-27.7, -12.3, -39.1, -27.3, -54.9]
    gs = [-30.1, -19.7, -42.1, -46.8, -115.0]
    x = np.arange(len(levers))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    ax.bar(x - w, gn, w, label="강남", color=POS)
    ax.bar(x, yd, w, label="영등포", color="#e08e0b")
    ax.bar(x + w, gs, w, label="강서", color=NEG)
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(levers)
    ax.set_ylabel("Δ (DQN - 휴리스틱)")
    ax.set_title("확장 레버 6종 — baseline 못 넘김 (강서 2트럭 -115 축약)", fontweight="bold")
    ax.set_ylim(-70, 20)
    ax.annotate("-115", xy=(4 + w, -68), ha="center", color=NEG, fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.savefig(OUT + "dqn_levers.png")
    plt.close(fig)


def fig_seed42_25gu():
    """시드 42 전체 25구 (k15, 200k, chronological). 출처: logs/runs/seedgrid/results_k15_seed42.csv"""
    data = [
        ("강남", 10.9), ("송파", 2.6), ("성북", -0.8), ("도봉", -1.2), ("동작", -1.6),
        ("서초", -1.8), ("성동", -4.4), ("중구", -4.5), ("서대문", -4.8), ("강북", -4.9),
        ("강동", -6.2), ("용산", -6.6), ("중랑", -6.9), ("관악", -8.4), ("은평", -9.5),
        ("종로", -9.6), ("마포", -12.0), ("금천", -14.0), ("노원", -15.3), ("광진", -17.1),
        ("동대문", -17.2), ("양천", -25.3), ("영등포", -27.7), ("강서", -30.1), ("구로", -32.5),
    ]
    names = [d[0] for d in data][::-1]
    vals = [d[1] for d in data][::-1]
    mean = sum(v for _, v in data) / len(data)
    fig, ax = plt.subplots(figsize=(7.5, 8))
    ax.barh(names, vals, color=signed_colors(vals))
    ax.axvline(0, color="#444", lw=1)
    ax.axvline(mean, color=ACC, lw=1.2, ls="--", label=f"평균 {mean:+.1f}")
    for i, v in enumerate(vals):
        ax.text(v + (0.4 if v >= 0 else -0.4), i, f"{v:+.1f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=8)
    ax.set_xlabel("Δ (DQN - 휴리스틱), seed 42 · k15 · 200k · chronological")
    ax.set_title("시드 42 전체 25개 자치구 - 추월 2/25 (강남·송파)", fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(axis="x", ls=":", alpha=0.4)
    fig.savefig(OUT + "dqn_seed42_25gu.png")
    plt.close(fig)
    print(f"  seed42 25gu mean delta = {mean:+.2f}")


def fig_learning_curves():
    """6개 대표구 × 3시드 timestep별 학습곡선 (eval_reward, k15·200k).
    history.npy: [{'timesteps':..., 'eval_reward':...}, ...]. 휴리스틱(점선) 위 = 추월."""
    import os
    # (표시명, 휴리스틱 reward) — 휴리스틱은 시드 무관 결정적 (출처: §3.2 표)
    best3 = [("강남", -138.2), ("송파", -560.0), ("성북", -44.5)]
    worst3 = [("구로", -301.7), ("강서", -683.2), ("영등포", -477.4)]
    seeds = [("seed 42", 42, "#2563eb"), ("seed 123", 123, "#e08e0b"),
             ("seed 777", 777, "#128a4a")]

    def hist_path(seed, gu):
        # seed42는 _dqn_small_, seed123/777은 _k15_dqn_small_ 패턴
        cands = [
            f"logs/dqn_seed{seed}_dqn_small_{gu}구/history.npy",
            f"logs/dqn_seed{seed}_k15_dqn_small_{gu}구/history.npy",
        ]
        for c in cands:
            if os.path.exists(c):
                return c
        return None

    rows = [("BEST 3", best3), ("WORST 3", worst3)]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for r, (grp, gus) in enumerate(rows):
        for c, (gu, heur) in enumerate(gus):
            ax = axes[r][c]
            ymax = heur
            for label, seed, color in seeds:
                p = hist_path(seed, gu)
                if p is None:
                    continue
                h = np.load(p, allow_pickle=True)
                xs = [d["timesteps"] / 1000 for d in h]
                ys = [d["eval_reward"] for d in h]
                ax.plot(xs, ys, "o-", color=color, lw=1.8, ms=4, label=label)
                ymax = max(ymax, max(ys))
            ax.axhline(heur, color="#444", ls="--", lw=1.3, label="휴리스틱")
            # 추월 영역(휴리스틱 위) 음영
            ax.axhspan(heur, ymax + abs(ymax) * 0.05 + 5, color=POS, alpha=0.06)
            ax.set_title(f"{grp} · {gu}  (휴={heur:.0f})", fontsize=11, fontweight="bold")
            ax.set_xlabel("timesteps (k)")
            ax.set_ylabel("eval reward")
            ax.grid(ls=":", alpha=0.4)
            if r == 0 and c == 0:
                ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("dqn_small 3-seed 학습곡선 (k15) — 휴리스틱(점선) 위로 올라가면 추월",
                 fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT + "dqn_learning_curves.png")
    plt.close(fig)


def fig_longtrain_plateau():
    """400k까지 학습해도 200k 이후는 평평(수렴)함을 보임. 출처: logs/dqn_small400_<구>/history.npy.
    주의: small400 학습-중 eval 스케일은 §3.3 holdout과 절대값이 다르므로 휴리스틱 선은 생략,
    '추세(plateau)'만 본다."""
    gus = ["강남", "구로", "강서", "영등포"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, gu in zip(axes.ravel(), gus):
        h = np.load(f"logs/dqn_small400_{gu}구/history.npy", allow_pickle=True)
        ts = [d["timesteps"] / 1000 for d in h]
        ev = [d["eval_reward"] for d in h]
        i200 = min(range(len(ts)), key=lambda i: abs(ts[i] - 200))
        gain = ev[-1] - ev[i200]
        early = ev[i200] - ev[0]  # 0~200k 상승폭
        ax.plot(ts, ev, "o-", color=ACC, lw=2, ms=5)
        ax.axvline(200, color=NEG, ls="--", lw=1.3)
        ax.text(200, ax.get_ylim()[0], " 200k\n (§3.4 컷오프)", color=NEG,
                fontsize=8, va="bottom", ha="left")
        # 200k 이후 구간 음영
        ax.axvspan(200, 400, color="#888", alpha=0.08)
        ax.set_title(f"{gu}  |  0→200k {early:+.0f}  vs  200k→400k {gain:+.0f}",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("timesteps (k)")
        ax.set_ylabel("eval reward (학습 중)")
        ax.grid(ls=":", alpha=0.4)
    fig.suptitle("400k까지 늘려도 200k 이후는 평평(수렴) — 초반 급상승 ≫ 후반 이득, 추월 안 늘어남",
                 fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT + "dqn_longtrain_plateau.png")
    plt.close(fig)


for f in (fig_seed, fig_stations, fig_trucks, fig_25gu, fig_algo, fig_levers,
          fig_seed42_25gu, fig_learning_curves, fig_longtrain_plateau):
    f()
    print("ok:", f.__name__)
print("DONE")
