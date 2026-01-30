from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

try:
    import torch
    from torch import nn, optim
    from torch.distributions.normal import Normal
except ImportError as exc:  # pragma: no cover - torch optional
    raise ImportError("PyTorch is required for PPO. Install with `pip install '.[rl]'`.") from exc


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(64, 64)):
        super().__init__()
        self.pi = mlp((obs_dim, *hidden, act_dim), activation=nn.Tanh, output_activation=nn.Identity)
        self.v = mlp((obs_dim, *hidden, 1), activation=nn.Tanh, output_activation=nn.Identity)
        # log-std is a learnable parameter (diagonal Gaussian)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def step(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.pi(obs)
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)
        x = dist.sample()
        act = torch.tanh(x)  # squash to [-1, 1]
        logp = dist.log_prob(x).sum(axis=-1)
        val = self.v(obs).squeeze(-1)
        return act, logp, val

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        return self.step(obs)[0]


@dataclass
class PPOConfig:
    steps_per_epoch: int = 2048
    epochs: int = 10
    train_iters: int = 10
    mini_batch: int = 64
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    pi_lr: float = 3e-4
    vf_lr: float = 1e-3
    target_kl: float = 0.015
    entropy_coef: float = 0.0
    device: str = "cpu"


class RolloutBuffer:
    def __init__(self, size: int, obs_dim: int, act_dim: int, device: str):
        self.obs = torch.zeros((size, obs_dim), device=device)
        self.act = torch.zeros((size, act_dim), device=device)
        self.logp = torch.zeros(size, device=device)
        self.rew = torch.zeros(size, device=device)
        self.val = torch.zeros(size, device=device)
        self.done = torch.zeros(size, device=device)
        self.adv = torch.zeros(size, device=device)
        self.ret = torch.zeros(size, device=device)
        self.ptr = 0
        self.path_start = 0
        self.max_size = size

    def store(self, obs, act, logp, rew, val, done):
        assert self.ptr < self.max_size
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.logp[self.ptr] = logp
        self.rew[self.ptr] = rew
        self.val[self.ptr] = val
        self.done[self.ptr] = done
        self.ptr += 1

    def finish_path(self, last_val: float, gamma: float, lam: float):
        path_slice = slice(self.path_start, self.ptr)
        rews = torch.cat([self.rew[path_slice], torch.tensor([last_val], device=self.rew.device)])
        vals = torch.cat([self.val[path_slice], torch.tensor([last_val], device=self.val.device)])
        deltas = rews[:-1] + gamma * vals[1:] - vals[:-1]
        adv = torch.zeros_like(deltas)
        gae = 0
        for t in reversed(range(len(deltas))):
            gae = deltas[t] + gamma * lam * gae * (1 - self.done[path_slice][t])
            adv[t] = gae
        self.adv[path_slice] = adv
        self.ret[path_slice] = adv + self.val[path_slice]
        self.path_start = self.ptr

    def get(self, mini_batch: int):
        assert self.ptr == self.max_size
        adv_norm = (self.adv - self.adv.mean()) / (self.adv.std() + 1e-8)
        idx = torch.randperm(self.max_size, device=self.obs.device)
        for start in range(0, self.max_size, mini_batch):
            j = idx[start : start + mini_batch]
            yield self.obs[j], self.act[j], self.logp[j], adv_norm[j], self.ret[j]


class PPOAgent:
    def __init__(self, obs_dim: int, act_dim: int, cfg: PPOConfig | None = None):
        self.cfg = cfg or PPOConfig()
        self.device = torch.device(self.cfg.device)
        self.ac = ActorCritic(obs_dim, act_dim).to(self.device)
        self.pi_opt = optim.Adam(self.ac.pi.parameters(), lr=self.cfg.pi_lr)
        self.vf_opt = optim.Adam(list(self.ac.v.parameters()) + [self.ac.log_std], lr=self.cfg.vf_lr)

    def collect(self, env) -> RolloutBuffer:
        obs_np = env.reset()
        obs_dim = obs_np.shape[0]
        act_dim = env.action_dim
        buf = RolloutBuffer(self.cfg.steps_per_epoch, obs_dim, act_dim, str(self.device))
        obs = torch.tensor(obs_np, device=self.device, dtype=torch.float32)
        for t in range(self.cfg.steps_per_epoch):
            with torch.no_grad():
                act, logp, val = self.ac.step(obs)
            act_np = act.cpu().numpy()
            obs_next, rew, done, _ = env.step(act_np)
            buf.store(obs, act, logp, torch.tensor(rew, device=self.device), val, torch.tensor(done, device=self.device))
            obs = torch.tensor(obs_next, device=self.device, dtype=torch.float32)
            if done:
                with torch.no_grad():
                    _, _, last_val = self.ac.step(obs)
                buf.finish_path(float(last_val), self.cfg.gamma, self.cfg.lam)
                if t < self.cfg.steps_per_epoch - 1:
                    obs = torch.tensor(env.reset(), device=self.device, dtype=torch.float32)
        # If episode ended exactly at buffer end, we already finished. If not, bootstrap.
        if buf.ptr < buf.max_size:
            with torch.no_grad():
                _, _, last_val = self.ac.step(obs)
            buf.finish_path(float(last_val), self.cfg.gamma, self.cfg.lam)
        return buf

    def update(self, buf: RolloutBuffer):
        for _ in range(self.cfg.train_iters):
            for obs, act, logp_old, adv, ret in buf.get(self.cfg.mini_batch):
                mean = self.ac.pi(obs)
                std = torch.exp(self.ac.log_std)
                dist = Normal(mean, std)
                logp = dist.log_prob(torch.atanh(torch.clamp(act, -0.999, 0.999))).sum(axis=-1)
                ratio = torch.exp(logp - logp_old)
                clip_adv = torch.clamp(ratio, 1 - self.cfg.clip_ratio, 1 + self.cfg.clip_ratio) * adv
                pi_loss = -(torch.min(ratio * adv, clip_adv)).mean() - self.cfg.entropy_coef * dist.entropy().mean()

                val = self.ac.v(obs).squeeze(-1)
                vf_loss = ((val - ret) ** 2).mean()

                kl = (logp_old - logp).mean().item()
                self.pi_opt.zero_grad()
                pi_loss.backward()
                self.pi_opt.step()

                self.vf_opt.zero_grad()
                vf_loss.backward()
                self.vf_opt.step()

                if kl > 1.5 * self.cfg.target_kl:
                    return {"pi_loss": pi_loss.item(), "vf_loss": vf_loss.item(), "kl": kl}
        return {}

    def train(self, env):
        history = []
        for epoch in range(self.cfg.epochs):
            buf = self.collect(env)
            stats = self.update(buf)
            history.append({"epoch": epoch, **stats})
        return history
