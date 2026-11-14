# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from torchrl.data import CompositeSpec, TensorSpec
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictModule, TensorDictSequential

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from typing import Union
import einops

from ..utils.valuenorm import ValueNorm1
from ..modules.distributions import IndependentNormal
from .common import GAE

@dataclass
class PPOConfig:
    name: str = "ppo"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 16

    # whether to use privileged information
    priv_actor: bool = False
    priv_critic: bool = False

    checkpoint_path: Union[str, None] = None

    # Initial log-std for the actor Gaussian.
    # Default 0.0 → std=1.0 (wide exploration).
    # Set to -1.0 → std≈0.37 for drones where large per-rotor variance
    # causes destabilizing torques early in training (e.g. heavy quadrotors).
    actor_log_std_init: float = 0.0

    # Bounds on the learned log-std. The floor matters whenever the reward contains a
    # quadratic penalty on the action: for a state-independent Gaussian,
    # E[||a||^2] = ||mu||^2 + sum(sigma^2), so such a penalty pushes log_std down with a
    # gradient proportional to its weight, while `entropy_coef` pushes back only with
    # 4*entropy_coef/sigma. With the default coefficient the penalty wins by orders of
    # magnitude and exploration collapses within a few hundred iterations, long before
    # the task is learned. Raise the floor (e.g. -1.6 -> sigma >= 0.2) to prevent that.
    actor_log_std_min: float = -5.0
    actor_log_std_max: float = 2.0

    entropy_coef: float = 0.001

cs = ConfigStore.instance()
cs.store("ppo", node=PPOConfig, group="algo")
cs.store("ppo_priv", node=PPOConfig(priv_actor=True, priv_critic=True), group="algo")
cs.store("ppo_priv_critic", node=PPOConfig(priv_critic=True), group="algo")


def make_mlp(num_units):
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))
        layers.append(nn.LeakyReLU())
        layers.append(nn.LayerNorm(n))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(
        self,
        action_dim: int,
        log_std_init: float = 0.0,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.actor_mean = nn.LazyLinear(action_dim)
        self.actor_std = nn.Parameter(torch.full((action_dim,), log_std_init))
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self._mean_zero_init_done = False

    def forward(self, features: torch.Tensor):
        # On the very first forward pass, zero-init the output layer so that
        # actor_mean(obs) = 0 for any observation.  Combined with the
        # hover_cmd_offset applied in the env, action=0 maps exactly to hover
        # thrust, giving a neutral starting point that prevents systematic
        # thrust deficit early in training.
        if not self._mean_zero_init_done:
            with torch.no_grad():
                self.actor_mean(features)          # materialise LazyLinear
                nn.init.zeros_(self.actor_mean.weight)
                nn.init.zeros_(self.actor_mean.bias)
            self._mean_zero_init_done = True
        loc = self.actor_mean(features)
        scale = torch.exp(
            self.actor_std.clamp(self.log_std_min, self.log_std_max)
        ).expand_as(loc)
        return loc, scale


class PPOPolicy(TensorDictModuleBase):

    @staticmethod
    def _is_finite(*tensors: torch.Tensor) -> bool:
        return all(torch.isfinite(t).all() for t in tensors)

    @staticmethod
    def _zero_info(device) -> TensorDict:
        z = torch.zeros((), device=device)
        return TensorDict(
            {
                "policy_loss": z,
                "value_loss": z,
                "entropy": z,
                "actor_grad_norm": z,
                "critic_grad_norm": z,
                "explained_var": z,
            },
            [],
        )

    def __init__(
        self,
        cfg: PPOConfig,
        observation_spec: CompositeSpec,
        action_spec: CompositeSpec,
        reward_spec: TensorSpec,
        device
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.entropy_coef = cfg.entropy_coef
        self.clip_param = 0.1
        self.critic_loss_fn = nn.HuberLoss(delta=10)
        self.n_agents, self.action_dim = action_spec.shape[-2:]
        self.gae = GAE(0.99, 0.95)

        fake_input = observation_spec.zero()

        if self.cfg.priv_actor:
            intrinsics_dim = observation_spec[("agents", "intrinsics")].shape[-1]
            actor_module = TensorDictSequential(
                TensorDictModule(make_mlp([128, 128]), [("agents", "observation")], ["feature"]),
                TensorDictModule(
                    nn.Sequential(nn.LayerNorm(intrinsics_dim), make_mlp([64, 64])),
                    [("agents", "intrinsics")], ["context"]
                ),
                CatTensors(["feature", "context"], "feature"),
                TensorDictModule(
                    nn.Sequential(make_mlp([256, 256]), Actor(self.action_dim, cfg.actor_log_std_init,
                                          cfg.actor_log_std_min, cfg.actor_log_std_max)),
                    ["feature"], ["loc", "scale"]
                )
            )
        else:
            actor_module=TensorDictModule(
                nn.Sequential(make_mlp([256, 256, 256]), Actor(self.action_dim, cfg.actor_log_std_init,
                                          cfg.actor_log_std_min, cfg.actor_log_std_max)),
                [("agents", "observation")], ["loc", "scale"]
            )
        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=[("agents", "action")],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)

        if self.cfg.priv_critic:
            intrinsics_dim = observation_spec[("agents", "intrinsics")].shape[-1]
            self.critic = TensorDictSequential(
                TensorDictModule(make_mlp([128, 128]), [("agents", "observation")], ["feature"]),
                TensorDictModule(
                    nn.Sequential(nn.LayerNorm(intrinsics_dim), make_mlp([64, 64])),
                    [("agents", "intrinsics")], ["context"]
                ),
                CatTensors(["feature", "context"], "feature"),
                TensorDictModule(
                    nn.Sequential(make_mlp([256, 256]), nn.LazyLinear(1)),
                    ["feature"], ["state_value"]
                )
            ).to(self.device)
        else:
            self.critic = TensorDictModule(
                nn.Sequential(make_mlp([256, 256, 256]), nn.LazyLinear(1)),
                [("agents", "observation")], ["state_value"]
            ).to(self.device)

        self.actor(fake_input)
        self.critic(fake_input)

        if self.cfg.checkpoint_path is not None:
            state_dict = torch.load(self.cfg.checkpoint_path)
            self.load_state_dict(state_dict, strict=False)
        else:
            def init_(module):
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, 0.01)
                    nn.init.constant_(module.bias, 0.)

            self.actor.apply(init_)
            self.critic.apply(init_)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=5e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=5e-4)
        self.value_norm = ValueNorm1(reward_spec.shape[-2:]).to(self.device)

    def __call__(self, tensordict: TensorDict):
        self.actor(tensordict)
        self.critic(tensordict)
        tensordict.exclude("loc", "scale", "feature", inplace=True)
        return tensordict

    def train_op(self, tensordict: TensorDict):
        next_tensordict = tensordict["next"]
        with torch.no_grad():
            next_values = self.critic(next_tensordict)["state_value"]
        rewards = torch.nan_to_num(
            tensordict[("next", "agents", "reward")], nan=0.0, posinf=0.0, neginf=0.0
        )
        dones = einops.repeat(
            tensordict[("next", "terminated")],
            "t e 1 -> t e a 1",
            a=self.n_agents
        )
        dones = torch.nan_to_num(dones.float(), nan=1.0, posinf=1.0, neginf=1.0).bool()
        values = self.value_norm.denormalize(tensordict["state_value"])
        next_values = self.value_norm.denormalize(next_values)
        values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        next_values = torch.nan_to_num(next_values, nan=0.0, posinf=0.0, neginf=0.0)

        adv, ret = self.gae(rewards, dones, values, next_values)
        adv = torch.nan_to_num(adv, nan=0.0, posinf=0.0, neginf=0.0)
        ret = torch.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)
        adv_mean = adv.mean()
        adv_std = adv.std().clamp_min(1e-6)
        adv = (adv - adv_mean) / adv_std
        adv = torch.nan_to_num(adv, nan=0.0, posinf=0.0, neginf=0.0)
        self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)
        ret = torch.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)

        tensordict.set("adv", adv)
        tensordict.set("ret", ret)

        infos = []
        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(self._update(minibatch))

        infos: TensorDict = torch.stack(infos).to_tensordict()
        infos = infos.apply(torch.mean, batch_size=[])
        return {k: v.item() for k, v in infos.items()}

    def _update(self, tensordict: TensorDict):
        dist = self.actor.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[("agents", "action")])
        entropy = dist.entropy()

        adv = tensordict["adv"]
        if not self._is_finite(
            log_probs, entropy, adv, tensordict["sample_log_prob"], tensordict["ret"]
        ):
            return self._zero_info(log_probs.device)
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
        policy_loss = - torch.mean(torch.min(surr1, surr2)) * self.action_dim
        entropy_loss = - self.entropy_coef * torch.mean(entropy)

        b_values = tensordict["state_value"]
        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        values_clipped = b_values + (values - b_values).clamp(
            -self.clip_param, self.clip_param
        )
        value_loss_clipped = self.critic_loss_fn(b_returns, values_clipped)
        value_loss_original = self.critic_loss_fn(b_returns, values)
        value_loss = torch.max(value_loss_original, value_loss_clipped)

        loss = policy_loss + entropy_loss + value_loss
        if not torch.isfinite(loss):
            return self._zero_info(loss.device)
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 5)
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 5)
        self.actor_opt.step()
        self.critic_opt.step()
        explained_var = 1 - F.mse_loss(values, b_returns) / b_returns.var().clamp_min(1e-6)
        actor_grad_norm = torch.nan_to_num(actor_grad_norm, nan=0.0, posinf=0.0, neginf=0.0)
        critic_grad_norm = torch.nan_to_num(critic_grad_norm, nan=0.0, posinf=0.0, neginf=0.0)
        entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
        policy_loss = torch.nan_to_num(policy_loss, nan=0.0, posinf=0.0, neginf=0.0)
        value_loss = torch.nan_to_num(value_loss, nan=0.0, posinf=0.0, neginf=0.0)
        explained_var = torch.nan_to_num(explained_var, nan=0.0, posinf=0.0, neginf=0.0)
        return TensorDict({
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "explained_var": explained_var
        }, [])


def make_batch(tensordict: TensorDict, num_minibatches: int):
    tensordict = tensordict.reshape(-1)
    perm = torch.randperm(
        (tensordict.shape[0] // num_minibatches) * num_minibatches,
        device=tensordict.device,
    ).reshape(num_minibatches, -1)
    for indices in perm:
        yield tensordict[indices]
