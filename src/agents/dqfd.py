"""DQfD — Deep Q-learning from Demonstrations (Hester et al. 2018) on MaskableDQN.

배경: BC fine-tune은 best가 step 5~10k(BC 직후)에 나오고 이후 forgetting으로 하락한다.
원인은 "목적함수 불일치" — BC는 q_net을 CrossEntropy logit으로 학습(크기 무의미)하는데,
DQN이 시작되면 Bellman target(누적 -500~-700 규모)과의 거대 오차가 BC argmax 구조를
수천 step 안에 덮어쓴다. LR을 낮춰도 "덮어쓰는 속도"만 줄 뿐 방향은 그대로다.

DQfD는 이 forgetting을 직격한다:
  1. demo transition을 별도 buffer에 **상주** — 학습 내내 절대 덮어쓰지 않음
  2. **large-margin supervised loss**: demo 행동 Q를 항상 margin만큼 1등으로 유지
       J_E = mean( max_a[Q(s,a)+ℓ(a_E,a)] − Q(s,a_E) ),  ℓ=margin(a≠a_E), 0(a=a_E)
  3. **pre-training**: 환경 상호작용 전 demo만으로 K step 학습 → Q를 실제 return 규모로
     보정(BC의 logit-크기 불일치 제거). 이 단계가 곧 "RL이 폭발 안 하는 출발점".
  4. **본 학습**: agent batch(TD) + demo batch(TD + margin) 결합 loss

  loss = TD(agent, double-Q) + TD(demo, double-Q) + λ_margin·J_E(demo) + λ_l2·‖θ‖²

MaskableDQN을 상속해 마스킹·탐색·predict는 그대로 재사용. margin의 내부 max에도
demo 상태 마스크를 적용해 무효 행동이 max를 차지하지 못하게 한다.
"""

from __future__ import annotations

import numpy as np
import torch as th
from stable_baselines3.common.utils import polyak_update
from torch.nn import functional as F

from src.agents.masked_dqn import MaskableDQN


class DemoBuffer:
    """demo full-transition을 device 텐서로 상주시키는 경량 버퍼.

    SB3 ReplayBuffer는 샘플 인덱스·mask를 노출하지 않아 직접 구현한다.
    demo 규모(≈60일×144step×3트럭 ≈ 2.6만)는 GPU/CPU 메모리에 충분히 올라간다.
    """

    def __init__(self, obs, actions, rewards, next_obs, dones, masks, device):
        self.obs = th.as_tensor(obs, dtype=th.float32, device=device)
        self.actions = th.as_tensor(actions, dtype=th.long, device=device).view(-1, 1)
        self.rewards = th.as_tensor(rewards, dtype=th.float32, device=device).view(-1, 1)
        self.next_obs = th.as_tensor(next_obs, dtype=th.float32, device=device)
        self.dones = th.as_tensor(dones, dtype=th.float32, device=device).view(-1, 1)
        self.masks = th.as_tensor(masks, dtype=th.bool, device=device)
        self.n = int(self.obs.shape[0])

    def sample(self, batch_size: int):
        idx = th.randint(0, self.n, (batch_size,), device=self.obs.device)
        return (self.obs[idx], self.actions[idx], self.rewards[idx],
                self.next_obs[idx], self.dones[idx], self.masks[idx])


class DQfDDQN(MaskableDQN):
    def __init__(self, *args,
                 demo_buffer: DemoBuffer | None = None,
                 margin: float = 0.8,
                 lambda_margin: float = 1.0,
                 lambda_l2: float = 1e-5,
                 lambda_bc: float = 1.0,
                 lambda_margin_final: float | None = None,
                 lambda_bc_final: float | None = None,
                 demo_batch_size: int | None = None,
                 double_q: bool = True,
                 **kwargs):
        super().__init__(*args, double_q=double_q, **kwargs)
        self.demo_buffer = demo_buffer
        self.margin = float(margin)
        self.lambda_margin = float(lambda_margin)
        self.lambda_l2 = float(lambda_l2)
        self.lambda_bc = float(lambda_bc)
        # 앵커 annealing: 본 학습 동안 init → final로 선형 감쇠 (None이면 상수).
        # 전반엔 강한 앵커로 forgetting 차단, 후반엔 약화해 RL이 휴리스틱 위로 개선.
        self.lambda_margin_final = lambda_margin_final
        self.lambda_bc_final = lambda_bc_final
        self.demo_batch_size = demo_batch_size

    def set_demo_buffer(self, demo_buffer: DemoBuffer) -> None:
        self.demo_buffer = demo_buffer

    def _eff_lambda(self, init: float, final: float | None) -> float:
        """진행도 기반 선형 anneal. final=None이면 상수 init.

        _current_progress_remaining: 1.0(시작) → 0.0(끝). SB3가 매 step 갱신.
        """
        if final is None:
            return init
        remaining = getattr(self, "_current_progress_remaining", 1.0)
        return final + (init - final) * remaining

    # ------------------------------------------------------------------
    # loss 구성요소
    # ------------------------------------------------------------------
    def _td_loss(self, obs, actions, rewards, next_obs, dones, discounts) -> th.Tensor:
        """Double-DQN 1-step TD (smooth_l1). discounts는 스칼라 gamma 또는 텐서."""
        with th.no_grad():
            next_actions = self.q_net(next_obs).argmax(dim=1, keepdim=True)
            next_q = self.q_net_target(next_obs).gather(1, next_actions)
            target = rewards + (1.0 - dones) * discounts * next_q
        current = self.q_net(obs).gather(1, actions)
        return F.smooth_l1_loss(current, target)

    def _margin_loss(self, obs, demo_actions, masks=None) -> th.Tensor:
        """large-margin supervised loss — demo 행동을 margin만큼 1등으로 강제.

        J_E = mean( max_a[Q(s,a)+ℓ(a,a_E)] − Q(s,a_E) ),  ℓ=margin(a≠a_E), 0(a=a_E)

        max에 demo 행동(ℓ=0)이 항상 포함되므로 J_E ≥ 0 (하한 보장). 표준 DQfD대로
        마스킹을 적용하지 않는다 — 마스킹하면 stay-폴백 등으로 demo 행동이 max에서
        빠져 loss가 음수로 발산할 수 있다(무효 행동 Q가 낮아지는 건 오히려 바람직).
        masks 인자는 호환을 위해 받되 사용하지 않는다.
        """
        q = self.q_net(obs)                                  # (B, N)
        margin_mat = self.margin * th.ones_like(q)
        margin_mat.scatter_(1, demo_actions, 0.0)            # demo 행동엔 margin 0
        max_term = (q + margin_mat).max(dim=1, keepdim=True).values
        q_demo = q.gather(1, demo_actions)
        return (max_term - q_demo).mean()

    def _bc_loss(self, obs, demo_actions) -> th.Tensor:
        """behavior-cloning CrossEntropy — Q를 logit으로 보고 demo 행동을 분류 학습.

        margin(hinge)은 샘플당 max 행동 1개에만 gradient를 줘 모방 신호가 약하다.
        CE는 전체 softmax로 demo 행동 확률을 직접 올려 BC 수준의 강한 모방을 준다.
        (DQfD를 self-contained하게 — 외부 BC 모델/config 정합성 의존 제거.)
        """
        logits = self.q_net(obs)                             # (B, N) — Q를 logit으로 해석
        return F.cross_entropy(logits, demo_actions.squeeze(1))

    def _l2_loss(self) -> th.Tensor:
        return sum((p ** 2).sum() for p in self.q_net.parameters())

    # ------------------------------------------------------------------
    # pre-training: demo만으로 Q 보정 (환경 상호작용 전)
    # ------------------------------------------------------------------
    def pretrain_on_demos(self, n_steps: int, batch_size: int | None = None,
                          lr: float | None = None) -> None:
        assert self.demo_buffer is not None, "demo_buffer가 필요합니다"
        bs = batch_size or self.batch_size
        self.policy.set_training_mode(True)
        # pretrain은 본질적으로 지도학습 — 본 RL lr(보통 1e-4)보다 높은 lr로 빠르게 수렴.
        saved_lr = [g["lr"] for g in self.policy.optimizer.param_groups]
        if lr is not None:
            for g in self.policy.optimizer.param_groups:
                g["lr"] = lr
        print(f"  [DQfD] pre-training on demos: {n_steps} steps (batch={bs}, "
              f"margin={self.margin}, λ_margin={self.lambda_margin}, λ_bc={self.lambda_bc}, "
              f"lr={lr or saved_lr[0]:.1e})")
        for step in range(n_steps):
            d_obs, d_act, d_rew, d_next, d_done, d_mask = self.demo_buffer.sample(bs)
            td = self._td_loss(d_obs, d_act, d_rew, d_next, d_done, self.gamma)
            margin = self._margin_loss(d_obs, d_act, d_mask)
            bc = self._bc_loss(d_obs, d_act)
            loss = (td + self.lambda_margin * margin + self.lambda_bc * bc
                    + self.lambda_l2 * self._l2_loss())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            if (step + 1) % self.target_update_interval == 0:
                polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)
            if (step + 1) % max(n_steps // 10, 1) == 0:
                print(f"    pretrain {step+1}/{n_steps}: loss={loss.item():.4f} "
                      f"(td={td.item():.4f}, margin={margin.item():.4f}, bc={bc.item():.4f})")
        # 마지막 target 동기화
        polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)

    # ------------------------------------------------------------------
    # 본 학습: agent batch(TD) + demo batch(TD + margin) 결합
    # ------------------------------------------------------------------
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        assert self.demo_buffer is not None, "demo_buffer가 필요합니다"
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        n_demo = self.demo_batch_size or batch_size

        # 이 train() 호출 시점의 앵커 강도 (annealing 반영)
        lam_margin = self._eff_lambda(self.lambda_margin, self.lambda_margin_final)
        lam_bc = self._eff_lambda(self.lambda_bc, self.lambda_bc_final)

        losses, td_a_log, td_d_log, margin_log, bc_log = [], [], [], [], []
        for _ in range(gradient_steps):
            # agent transition (online replay buffer)
            rb = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = getattr(rb, "discounts", None)
            discounts = discounts if discounts is not None else self.gamma
            td_agent = self._td_loss(
                rb.observations, rb.actions.long(), rb.rewards,
                rb.next_observations, rb.dones, discounts,
            )

            # demo transition (상주 buffer) — TD + margin + BC(CE)로 forgetting 방지
            d_obs, d_act, d_rew, d_next, d_done, d_mask = self.demo_buffer.sample(n_demo)
            td_demo = self._td_loss(d_obs, d_act, d_rew, d_next, d_done, self.gamma)
            margin = self._margin_loss(d_obs, d_act, d_mask)
            bc = self._bc_loss(d_obs, d_act)

            loss = (td_agent + td_demo + lam_margin * margin
                    + lam_bc * bc + self.lambda_l2 * self._l2_loss())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            losses.append(loss.item())
            td_a_log.append(td_agent.item())
            td_d_log.append(td_demo.item())
            margin_log.append(margin.item())
            bc_log.append(bc.item())

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", float(np.mean(losses)))
        self.logger.record("train/td_agent", float(np.mean(td_a_log)))
        self.logger.record("train/td_demo", float(np.mean(td_d_log)))
        self.logger.record("train/margin_loss", float(np.mean(margin_log)))
        self.logger.record("train/bc_loss", float(np.mean(bc_log)))
        self.logger.record("train/lam_margin", lam_margin)
        self.logger.record("train/lam_bc", lam_bc)


def collect_demo_transitions(env, policy_name: str = "most_imbalanced",
                             reward_scale: float = 1.0, n_episodes: int | None = None,
                             policy=None):
    """teacher 정책을 train episode에서 굴려 full transition (s,a,r,s',done,mask) 수집.

    env: use_action_mask=True 환경. raw(146) 또는 추상 wrapper 모두 가능.
      - raw + most_imbalanced → 기존 DQfD
      - 추상 + ConstantIntentPolicy(5) → "항상 predictive" warm-start demo
    policy: 직접 정책 객체 주면 그걸로(추상 의도 등), 없으면 policy_name으로 get_policy.
    reward_scale: 학습 env(RewardScale)와 동일 배수로 demo reward도 스케일 (TD 일관성).
    """
    from src.agents.baselines import get_policy

    if policy is None:
        policy = get_policy(policy_name)
    base = env
    while not hasattr(base, "trucks"):   # policy.act은 RebalanceEnv 본체가 필요
        base = base.env
    n_eps = n_episodes if n_episodes is not None else len(base._episodes)

    O, A, R, NO, D, M = [], [], [], [], [], []
    for ei in range(n_eps):
        obs, _ = env.reset(options={"episode_idx": ei})
        done = False
        while not done:
            mask = np.asarray(env.action_masks(), dtype=bool)
            a = int(policy.act(base))
            next_obs, r, done, trunc, _ = env.step(a)
            O.append(np.asarray(obs, dtype=np.float32))
            A.append(a)
            R.append(float(r) * reward_scale)
            NO.append(np.asarray(next_obs, dtype=np.float32))
            D.append(1.0 if done else 0.0)
            M.append(mask)
            obs = next_obs
            if trunc:
                break
    return (np.asarray(O, dtype=np.float32), np.asarray(A, dtype=np.int64),
            np.asarray(R, dtype=np.float32), np.asarray(NO, dtype=np.float32),
            np.asarray(D, dtype=np.float32), np.asarray(M, dtype=bool))


__all__ = ["DQfDDQN", "DemoBuffer", "collect_demo_transitions"]
