"""Maskable QR-DQN (Quantile Regression DQN).

QR-DQN은 Q-값을 단일 평균값 대신 분위수(Quantiles)의 집합으로 예측하여
리턴의 분포를 더 정확하게 학습한다.
본 구현은 MaskableDQN과 마찬가지로 invalid-action masking을 지원한다.
"""

from __future__ import annotations

import numpy as np
import torch as th
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from src.agents.masked_dqn import MaskableDQN


class QuantileNetwork(QNetwork):
    """분위수를 출력하는 Q-네트워크."""
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: nn.Module,
        features_dim: int,
        net_arch: list[int] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        n_quantiles: int = 200,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor,
            features_dim,
            net_arch,
            activation_fn,
            normalize_images,
        )
        self.n_quantiles = n_quantiles
        action_dim = int(self.action_space.n)
        
        # 기본 QNetwork의 q_net을 분위수 출력용으로 재정의
        q_layers = []
        last_layer_dim = features_dim
        if net_arch is not None:
            for layer_dim in net_arch:
                q_layers.append(nn.Linear(last_layer_dim, layer_dim))
                q_layers.append(activation_fn())
                last_layer_dim = layer_dim
        
        q_layers.append(nn.Linear(last_layer_dim, action_dim * n_quantiles))
        self.q_net = nn.Sequential(*q_layers)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """(batch_size, n_actions * n_quantiles) 반환."""
        return self.q_net(self.extract_features(obs, self.features_extractor))


class QRDQNPolicy(DQNPolicy):
    """QR-DQN용 정책 클래스."""
    def __init__(self, *args, n_quantiles: int = 200, **kwargs):
        self.n_quantiles = n_quantiles
        super().__init__(*args, **kwargs)
        
        # 분위수 타겟을 위한 cumulative probabilities (tau_hat)
        # buffer로 등록하여 device 이동 등에 대응
        tau = th.arange(0, self.n_quantiles + 1).float() / self.n_quantiles
        tau_hat = (tau[:-1] + tau[1:]) / 2.0
        self.register_buffer("tau_hat", tau_hat.view(1, 1, -1))

    def make_q_net(self) -> QuantileNetwork:
        # SB3 2.8.0의 DQNPolicy.make_q_net 패턴을 따름
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        net_args["n_quantiles"] = self.n_quantiles
        return QuantileNetwork(**net_args).to(self.device)


class MaskableQRDQN(MaskableDQN):
    """분위수 회귀 DQN + Action Masking."""
    def __init__(
        self,
        policy: str | type[QRDQNPolicy],
        env,
        n_quantiles: int = 200,
        kappa: float = 1.0,
        **kwargs,
    ):
        if isinstance(policy, str) and policy == "MlpPolicy":
            policy = QRDQNPolicy
            
        # policy_kwargs에 n_quantiles 전달
        policy_kwargs = kwargs.get("policy_kwargs", {})
        policy_kwargs["n_quantiles"] = n_quantiles
        kwargs["policy_kwargs"] = policy_kwargs
        
        super().__init__(policy, env, **kwargs)
        self.n_quantiles = n_quantiles
        self.kappa = kappa # Huber loss threshold

    def _masked_argmax(self, obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """분위수의 평균(Q-값)에 마스크 적용 후 argmax."""
        with th.no_grad():
            obs_t = th.as_tensor(obs, device=self.device)
            # quantiles: (n_envs, n_actions * n_quantiles)
            quantiles = self.q_net(obs_t)
            # (n_envs, n_actions, n_quantiles)
            action_dim = int(self.action_space.n)
            quantiles = quantiles.view(-1, action_dim, self.n_quantiles)
            # Q-값은 분위수의 평균
            q = quantiles.mean(dim=2).cpu().numpy()
            
        q = q.copy()
        q[~masks] = -np.inf
        return q.argmax(axis=1)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        
        action_dim = int(self.action_space.n)

        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                # Next state 분위수 계산
                next_quantiles_target = self.q_net_target(replay_data.next_observations)
                next_quantiles_target = next_quantiles_target.view(batch_size, action_dim, self.n_quantiles)
                
                if self.double_q:
                    # Double DQN: online net의 평균 Q로 action 선택
                    next_quantiles_online = self.q_net(replay_data.next_observations)
                    next_quantiles_online = next_quantiles_online.view(batch_size, action_dim, self.n_quantiles)
                    next_actions = next_quantiles_online.mean(dim=2).argmax(dim=1, keepdim=True)
                else:
                    # Vanilla: target net의 평균 Q로 action 선택
                    next_actions = next_quantiles_target.mean(dim=2).argmax(dim=1, keepdim=True)
                
                # 선택된 action의 분위수 추출: (batch_size, 1, n_quantiles)
                next_actions_idx = next_actions.unsqueeze(-1).expand(batch_size, 1, self.n_quantiles)
                next_quantiles = next_quantiles_target.gather(1, next_actions_idx).squeeze(1)
                
                # Target 분위수: (batch_size, n_quantiles)
                target_quantiles = (
                    replay_data.rewards + (1 - replay_data.dones) * discounts * next_quantiles
                )

            # 현재 state 분위수 계산: (batch_size, n_actions * n_quantiles)
            current_quantiles_all = self.q_net(replay_data.observations)
            current_quantiles_all = current_quantiles_all.view(batch_size, action_dim, self.n_quantiles)
            
            # 취한 action의 분위수: (batch_size, n_quantiles)
            actions_idx = replay_data.actions.long().unsqueeze(-1).expand(batch_size, 1, self.n_quantiles)
            current_quantiles = current_quantiles_all.gather(1, actions_idx).squeeze(1)

            # Quantile Huber Loss
            # pairwise diff: (batch_size, n_quantiles (target), n_quantiles (current))
            diff = target_quantiles.unsqueeze(2) - current_quantiles.unsqueeze(1)
            abs_diff = diff.abs()
            
            # Huber loss
            huber_loss = th.where(
                abs_diff <= self.kappa,
                0.5 * diff.pow(2),
                self.kappa * (abs_diff - 0.5 * self.kappa)
            )
            
            # Quantile weight
            # policy.tau_hat: (1, 1, n_quantiles) -> weight: (batch_size, n_quantiles_target, n_quantiles_current)
            # diff < 0 이면 current가 target보다 크다는 의미 (I_u<0)
            weight = th.abs(self.policy.tau_hat - (diff < 0).float())
            
            # Loss: target 분위수들에 대해 sum, current 분위수들에 대해 mean
            # QR-DQN loss = E_j [ rho_tau_i (target_j - current_i) ]
            loss = (weight * huber_loss / self.kappa).sum(dim=1).mean(dim=1).mean()
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))


__all__ = ["MaskableQRDQN"]
