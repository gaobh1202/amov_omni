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


import logging
import omni_drones.utils.kit as kit_utils
from omni_drones.utils.torch import euler_to_quaternion, quat_rotate
import omni.isaac.core.utils.prims as prim_utils
import torch
import torch.distributions as D
from torch.func import vmap

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec
from omni.isaac.debug_draw import _debug_draw

from ..utils import lemniscate, scale_time

class Track(IsaacEnv):
    r"""
    A basic control task. The goal for the agent is to track a reference
    lemniscate trajectory in the 3D space.

    ## Observation

    - `rpos` (3 * `future_traj_steps`): The relative position of the drone to the
      reference positions in the future `future_traj_steps` time steps.
    - `drone_state` (16 + `num_rotors`): The basic information of the drone (except its position),
      containing its rotation (in quaternion), velocities (linear and angular),
      heading and up vectors, and the current throttle.
    - `time_encoding` (optional): The time encoding, which is a 4-dimensional
      vector encoding the current progress of the episode.

    ## Reward

    - `pos`: Reward for tracking the trajectory, computed from the position
      error as {math}`\exp(-a * \text{pos_error})`.
    - `up`: Reward computed from the uprightness of the drone to discourage
      large tilting.
    - `spin`: Reward computed from the spin of the drone to discourage spinning.
    - `effort`: Reward computed from the effort of the drone to optimize the
      energy consumption.
    - `action_smoothness`: Reward that encourages smoother drone actions, computed based on the throttle difference of the drone.

    The total reward is computed as follows:

    ```{math}
        r = r_\text{pos} + r_\text{pos} * (r_\text{up} + r_\text{heading}) + r_\text{effort} + r_\text{action_smoothness}
    ```

    ## Episode End

    The episode ends when the tracking error is larger than `reset_thres`, or
    when the drone is too close to the ground, or when the episode reaches
    the maximum length.

    ## Config

    | Parameter               | Type  | Default       | Description                                                                                                                                                                                                                             |
    | ----------------------- | ----- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | `drone_model`           | str   | "hummingbird" | Specifies the model of the drone being used in the environment.                                                                                                                                                                         |
    | `reset_thres`           | float | 0.5           | Threshold for the distance between the drone and its target, upon exceeding which the episode will be reset.                                                                                                                            |
    | `future_traj_steps`     | int   | 4             | Number of future trajectory steps the drone needs to predict.                                                                                                                                                                           |
    | `reward_distance_scale` | float | 1.2           | Scales the reward based on the distance between the drone and its target.                                                                                                                                                               |
    | `time_encoding`         | bool  | True          | Indicates whether to include time encoding in the observation space. If set to True, a 4-dimensional vector encoding the current progress of the episode is included in the observation. If set to False, this feature is not included. |

    ### Trajectory shaping (optional)

    | `traj_w` | [low, high] | [0.8, 1.1] | Angular rate along the lemniscate. One lap takes about `2*pi/(w*dt)` control steps, so lowering `w` slows the demanded flight speed proportionally. |
    | `traj_scale` | [[x,y,z], [x,y,z]] | [[1.8,1.8,1.0],[3.2,3.2,1.5]] | Size of the figure-eight; smaller means gentler curvature and lower lateral acceleration. |
    | `traj_c` | [low, high] | [-0.6, 0.6] | Vertical amplitude coefficient. |
    | `init_rpy` | [[r,p,y],[r,p,y]] | [[-.2,-.2,0],[.2,.2,2]] | Spawn attitude range, in units of pi. |

    ### Smoothness / attitude shaping (optional, all default to 0 = original behaviour)

    | `penalty_spin_weight` | float | 0. | Penalty on squared body yaw rate. Unlike the saturating `reward_spin` bonus this keeps growing, so it actually suppresses fast spinning. |
    | `penalty_body_rate_weight` | float | 0. | Penalty on squared body roll/pitch rate. |
    | `penalty_tilt_weight` | float | 0. | Penalty on `1 - up_z` (roughly half the squared tilt angle). |
    | `penalty_action_weight` | float | 0. | Penalty on the squared commanded body rates of the *policy* action. Requires `action_transform: rate`. |
    | `penalty_action_rate_weight` | float | 0. | Penalty on the squared step-to-step change of the policy action. Requires `action_transform: rate`. |
    | `reward_heading_weight` | float | 0. | Bonus (scaled by `reward_pose`) for pointing along the heading reference. |
    | `heading_mode` | str | "none" | "fixed" tracks the constant world heading `heading_ref`; "tangent" tracks the trajectory direction. |
    | `heading_ref` | [x, y] | [1., 0.] | World-frame xy heading reference for `heading_mode: fixed`. |
    | `max_tilt_deg` | float | None | Terminate the episode once the drone tips past this angle. |
    """
    def __init__(self, cfg, headless):
        self.reset_thres = cfg.task.reset_thres
        self.reward_effort_weight = cfg.task.reward_effort_weight
        self.reward_action_smoothness_weight = cfg.task.reward_action_smoothness_weight
        self.reward_distance_scale = cfg.task.reward_distance_scale
        self.time_encoding = cfg.task.time_encoding
        self.future_traj_steps = int(cfg.task.future_traj_steps)
        assert self.future_traj_steps > 0
        self.intrinsics = cfg.task.intrinsics
        self.wind = cfg.task.wind
        self.ground_reset_height = float(cfg.task.get("ground_reset_height", 0.1))
        self._arm_hold_enabled = False
        self._arm_hold_targets = None

        # --- smoothness / attitude shaping (all default to 0 -> original behaviour) ---
        # Explicit penalties, subtracted from the reward. Unlike the saturating
        # `reward_spin` bonus, these keep pushing down as the motion gets more violent.
        self.penalty_spin_weight = float(cfg.task.get("penalty_spin_weight", 0.))
        self.penalty_body_rate_weight = float(cfg.task.get("penalty_body_rate_weight", 0.))
        self.penalty_tilt_weight = float(cfg.task.get("penalty_tilt_weight", 0.))
        self.penalty_action_weight = float(cfg.task.get("penalty_action_weight", 0.))
        self.penalty_action_rate_weight = float(cfg.task.get("penalty_action_rate_weight", 0.))
        self.reward_heading_weight = float(cfg.task.get("reward_heading_weight", 0.))
        self.heading_mode = str(cfg.task.get("heading_mode", "none")).lower()
        if self.heading_mode not in ("none", "fixed", "tangent"):
            raise ValueError(f"unknown heading_mode: {self.heading_mode}")
        # World-frame xy heading reference for heading_mode="fixed". A *constant*
        # reference keeps the task fully observable: the drone's own heading is part
        # of the observation, so it can tell how far off it is. Anchoring to a random
        # per-episode yaw instead would make the target unobservable.
        self._heading_ref_cfg = list(cfg.task.get("heading_ref", [1., 0.]))
        # Terminate when the drone tips past this angle (degrees); None disables it.
        _max_tilt = cfg.task.get("max_tilt_deg", None)
        self.max_tilt_cos = (
            None if _max_tilt is None
            else float(torch.cos(torch.tensor(float(_max_tilt) * torch.pi / 180.)))
        )

        super().__init__(cfg, headless)

        self.drone.initialize()
        randomization = self.cfg.task.get("randomization", None)
        if randomization is not None:
            if "drone" in self.cfg.task.randomization:
                self.drone.setup_randomization(self.cfg.task.randomization["drone"])

        _has_arm_joints = (
            hasattr(self.drone, "arm_joint_indices")
            and self.drone.arm_joint_indices is not None
        )
        _arm_hold_cfg = cfg.task.get("arm_hold", None)
        _arm_hold_requested = (
            _arm_hold_cfg is not None and _arm_hold_cfg.get("enabled", False)
        )
        if _arm_hold_requested and not _has_arm_joints:
            logging.warning("arm_hold is enabled but this drone has no arm joints.")
        elif _arm_hold_requested:
            raw_targets = _arm_hold_cfg.get("joint_positions", [])
            hold_targets = torch.as_tensor(
                raw_targets, dtype=torch.float32, device=self.device
            ).flatten()
            if hold_targets.numel() < self.drone.num_arm_joints:
                padded = torch.zeros(self.drone.num_arm_joints, device=self.device)
                padded[: hold_targets.numel()] = hold_targets
                hold_targets = padded
            elif hold_targets.numel() > self.drone.num_arm_joints:
                hold_targets = hold_targets[: self.drone.num_arm_joints]
            self._arm_hold_enabled = True
            self._arm_hold_targets = hold_targets
            logging.info(
                "Arm hold enabled with targets=%s",
                hold_targets.detach().cpu().tolist(),
            )

        if self.wind:
            if randomization is not None:
                if "wind" in self.cfg.task.randomization:
                    cfg = self.cfg.task.randomization["wind"]
                    # for phase in ("train", "eval"):
                    wind_intensity_scale = cfg['train'].get("intensity", None)
                    self.wind_intensity_low = wind_intensity_scale[0]
                    self.wind_intensity_high = wind_intensity_scale[1]
            else:
                self.wind_intensity_low = 0
                self.wind_intensity_high = 2
            self.wind_w = torch.zeros(self.num_envs, 3, 8, device=self.device)
            self.wind_i = torch.zeros(self.num_envs, 1, device=self.device)

        def _cfg_pair(key, default_low, default_high):
            """Read a [low, high] range from the task config, falling back to defaults."""
            rng = cfg.task.get(key, None)
            if rng is None:
                return default_low, default_high
            low, high = list(rng)
            return low, high

        # `init_rpy` is in units of pi; yaw spans a full turn by default.
        _rpy_low, _rpy_high = _cfg_pair("init_rpy", [-.2, -.2, 0.], [0.2, 0.2, 2.])
        self.init_rpy_dist = D.Uniform(
            torch.tensor(_rpy_low, dtype=torch.float32, device=self.device) * torch.pi,
            torch.tensor(_rpy_high, dtype=torch.float32, device=self.device) * torch.pi
        )
        self.traj_rpy_dist = D.Uniform(
            torch.tensor([0., 0., 0.], device=self.device) * torch.pi,
            torch.tensor([0., 0., 2.], device=self.device) * torch.pi
        )
        _c_low, _c_high = _cfg_pair("traj_c", -0.6, 0.6)
        self.traj_c_dist = D.Uniform(
            torch.tensor(float(_c_low), device=self.device),
            torch.tensor(float(_c_high), device=self.device)
        )
        _scale_low, _scale_high = _cfg_pair("traj_scale", [1.8, 1.8, 1.], [3.2, 3.2, 1.5])
        self.traj_scale_dist = D.Uniform(
            torch.tensor(_scale_low, dtype=torch.float32, device=self.device),
            torch.tensor(_scale_high, dtype=torch.float32, device=self.device)
        )
        # Angular rate along the lemniscate: one lap takes ~2*pi/(w*dt) control steps,
        # so halving `traj_w` halves the demanded flight speed.
        _w_low, _w_high = _cfg_pair("traj_w", 0.8, 1.1)
        self.traj_w_dist = D.Uniform(
            torch.tensor(float(_w_low), device=self.device),
            torch.tensor(float(_w_high), device=self.device)
        )
        _origin = list(cfg.task.get("traj_origin", [0., 0., 2.]))
        self.origin = torch.tensor(_origin, dtype=torch.float32, device=self.device)

        self.traj_t0 = torch.pi / 2
        self.traj_c = torch.zeros(self.num_envs, device=self.device)
        self.traj_scale = torch.zeros(self.num_envs, 3, device=self.device)
        self.traj_rot = torch.zeros(self.num_envs, 4, device=self.device)
        self.traj_w = torch.ones(self.num_envs, device=self.device)

        self.target_pos = torch.zeros(self.num_envs, self.future_traj_steps, 3, device=self.device)

        # Previous *policy* action (pre-controller-transform), for the action-rate penalty.
        self.prev_action = torch.zeros(self.num_envs, 1, 4, device=self.device)
        self.policy_action = torch.zeros(self.num_envs, 1, 4, device=self.device)
        self._has_policy_action = False
        self._action_captured = False
        ref = torch.tensor(self._heading_ref_cfg, dtype=torch.float32, device=self.device)
        self.ref_heading = torch.nn.functional.normalize(ref, dim=-1).expand(self.num_envs, 2)

        self.alpha = 0.8

        self.draw = _debug_draw.acquire_debug_draw_interface()

    def _design_scene(self):
        drone_model_cfg = self.cfg.task.drone_model
        self.drone, self.controller = MultirotorBase.make(
            drone_model_cfg.name, drone_model_cfg.controller
        )

        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        self.drone.spawn(translations=[(0.0, 0.0, 1.5)])
        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1]
        obs_dim = drone_state_dim + 3 * (self.future_traj_steps-1)
        if self.time_encoding:
            self.time_encoding_dim = 4
            obs_dim += self.time_encoding_dim
        if self.intrinsics:
            obs_dim += sum(spec.shape[-1] for name, spec in self.drone.info_spec.items())

        self.observation_spec = CompositeSpec({
            "agents": {
                "observation": UnboundedContinuousTensorSpec((1, obs_dim))
            },
            "info": CompositeSpec({
                "drone_state": UnboundedContinuousTensorSpec((1, drone_state_dim), device=self.device),
            }),
        }).expand(self.num_envs).to(self.device)
        self.action_spec = CompositeSpec({
            "agents": {
                "action": self.drone.action_spec.unsqueeze(0),
            }
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": {
                "reward": UnboundedContinuousTensorSpec((1, 1))
            }
        }).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
        )
        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "tracking_error": UnboundedContinuousTensorSpec(1),
            "tracking_error_ema": UnboundedContinuousTensorSpec(1),
            "action_smoothness": UnboundedContinuousTensorSpec(1),
            "body_rate_ema": UnboundedContinuousTensorSpec(1),
            "yaw_rate_ema": UnboundedContinuousTensorSpec(1),
            "tilt_deg_ema": UnboundedContinuousTensorSpec(1),
            "action_rate_ema": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids, self.training)
        if self._arm_hold_enabled:
            self.drone.reset_arm_joints(env_ids, self._arm_hold_targets)
        self.traj_c[env_ids] = self.traj_c_dist.sample(env_ids.shape)
        self.traj_rot[env_ids] = euler_to_quaternion(self.traj_rpy_dist.sample(env_ids.shape))
        self.traj_scale[env_ids] = self.traj_scale_dist.sample(env_ids.shape)
        traj_w = self.traj_w_dist.sample(env_ids.shape)
        self.traj_w[env_ids] = torch.randn_like(traj_w).sign() * traj_w

        t0 = torch.zeros(len(env_ids), device=self.device)
        pos = lemniscate(t0 + self.traj_t0, self.traj_c[env_ids]) + self.origin
        rot = euler_to_quaternion(self.init_rpy_dist.sample(env_ids.shape))
        vel = torch.zeros(len(env_ids), 1, 6, device=self.device)
        self.drone.set_world_poses(
            pos + self.envs_positions[env_ids], rot, env_ids
        )
        self.drone.set_velocities(vel, env_ids)

        self.prev_action[env_ids] = 0.
        self.policy_action[env_ids] = 0.

        self.stats[env_ids] = 0.

        if self._should_render(0) and (env_ids == self.central_env_idx).any() :
            # visualize the trajectory
            self.draw.clear_lines()

            traj_vis = self._compute_traj(self.max_episode_length, self.central_env_idx.unsqueeze(0))[0]
            traj_vis = traj_vis + self.envs_positions[self.central_env_idx]
            point_list_0 = traj_vis[:-1].tolist()
            point_list_1 = traj_vis[1:].tolist()
            colors = [(1.0, 1.0, 1.0, 1.0) for _ in range(len(point_list_0))]
            sizes = [1 for _ in range(len(point_list_0))]
            self.draw.draw_lines(point_list_0, point_list_1, colors, sizes)

        if self.wind:
            self.wind_i[env_ids] = torch.rand(*env_ids.shape, 1, device=self.device) * (self.wind_intensity_high-self.wind_intensity_low) + self.wind_intensity_low
            self.wind_w[env_ids] = torch.randn(*env_ids.shape, 3, 8, device=self.device)

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        # The RateController transform stashes the pre-transform policy action here;
        # that (not the rotor command) is what we want to keep small and smooth.
        # _pre_sim_step runs once per substep while the action only changes once per
        # control step, so capture it on the first substep only.
        raw_action = tensordict.get(("agents", "policy_action"), None)
        if raw_action is not None and not self._action_captured:
            self.prev_action[:] = self.policy_action
            self.policy_action[:] = raw_action
            self._has_policy_action = True
            self._action_captured = True
        self.effort = self.drone.apply_action(actions)

    def _post_sim_step(self, tensordict: TensorDictBase):
        self._action_captured = False

        if self.wind:
            t = (self.progress_buf * self.dt).reshape(-1, 1, 1)
            self.wind_force = self.wind_i * torch.sin(t * self.wind_w).sum(-1)
            wind_forces = self.drone.MASS_0 * self.wind_force
            wind_forces = wind_forces.unsqueeze(1).expand(*self.drone.shape, 3)
            self.drone.base_link.apply_forces(wind_forces, is_global=True)

    def _compute_state_and_obs(self):
        self.drone_state = self.drone.get_state()

        self.target_pos[:] = self._compute_traj(self.future_traj_steps, step_size=5)

        self.rpos = self.target_pos - self.drone_state[..., :3]
        obs = [
            self.rpos.flatten(1).unsqueeze(1),
            self.drone_state[..., 3:],
        ]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            obs.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))
        if self.intrinsics:
            obs.append(self.drone.get_info())

        obs = torch.cat(obs, dim=-1)

        self.stats["action_smoothness"].lerp_(-self.drone.throttle_difference, (1-self.alpha))

        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                },
                "info": {
                    "drone_state": self.drone_state,
                },
                "stats": self.stats.clone(),
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        # pos reward
        distance = torch.norm(self.rpos[:, [0]], dim=-1)
        self.stats["tracking_error"].add_(-distance)
        self.stats["tracking_error_ema"].lerp_(distance, (1-self.alpha))

        reward_pose = torch.exp(-self.reward_distance_scale * distance)

        # uprightness
        tiltage = torch.abs(1 - self.drone.up[..., 2])
        reward_up = 0.5 / (1.0 + torch.square(tiltage))

        # effort
        reward_effort = self.reward_effort_weight * torch.exp(-self.effort)
        reward_action_smoothness = self.reward_action_smoothness_weight * torch.exp(-self.drone.throttle_difference)

        # spin reward
        spin = torch.square(self.drone.vel[..., -1])
        reward_spin = 0.5 / (1.0 + torch.square(spin))

        # --- explicit smoothness / attitude penalties (body frame) ---
        body_rate = self.drone.vel[..., 3:]
        rate_xy_sq = torch.square(body_rate[..., :2]).sum(-1)
        yaw_rate_sq = torch.square(body_rate[..., 2])
        tilt_cos = self.drone.up[..., 2]

        penalty = (
            self.penalty_spin_weight * yaw_rate_sq
            + self.penalty_body_rate_weight * rate_xy_sq
            + self.penalty_tilt_weight * (1.0 - tilt_cos)  # ~ tilt_angle^2 / 2
        )
        if self._has_policy_action:
            action_sq = torch.square(self.policy_action[..., :3]).sum(-1)
            action_rate_sq = torch.square(
                self.policy_action - self.prev_action
            ).sum(-1)
            penalty = penalty + (
                self.penalty_action_weight * action_sq
                + self.penalty_action_rate_weight * action_rate_sq
            )
        else:
            action_rate_sq = torch.zeros_like(yaw_rate_sq)

        # heading reward: hold a fixed world heading, or face along the trajectory tangent
        if self.heading_mode == "none" or self.reward_heading_weight == 0.:
            reward_heading = torch.zeros_like(reward_pose)
        else:
            if self.heading_mode == "fixed":
                ref = self.ref_heading
            else:  # "tangent"
                tangent = self.target_pos[:, -1, :2] - self.target_pos[:, 0, :2]
                ref = torch.nn.functional.normalize(tangent, dim=-1, eps=1e-6)
            heading_xy = torch.nn.functional.normalize(
                self.drone.heading[..., :2], dim=-1, eps=1e-6
            )
            alignment = (heading_xy * ref.unsqueeze(1)).sum(-1)
            reward_heading = self.reward_heading_weight * (alignment + 1.0) / 2.0

        self.stats["body_rate_ema"].lerp_(rate_xy_sq.sqrt(), (1 - self.alpha))
        self.stats["yaw_rate_ema"].lerp_(yaw_rate_sq.sqrt(), (1 - self.alpha))
        self.stats["tilt_deg_ema"].lerp_(
            torch.rad2deg(torch.acos(tilt_cos.clamp(-1.0, 1.0))), (1 - self.alpha)
        )
        self.stats["action_rate_ema"].lerp_(action_rate_sq.sqrt(), (1 - self.alpha))

        reward = (
            reward_pose
            + reward_pose * (reward_up + reward_spin + reward_heading)
            + reward_effort
            + reward_action_smoothness
            - penalty
        )

        misbehave = (
            (self.drone.pos[..., 2] < self.ground_reset_height)
            | (distance > self.reset_thres)
        )
        if self.max_tilt_cos is not None:
            misbehave = misbehave | (tilt_cos < self.max_tilt_cos)
        hasnan = torch.isnan(self.drone_state).any(-1)

        terminated = misbehave | hasnan
        truncated = (self.progress_buf >= self.max_episode_length - 1).unsqueeze(-1)

        ep_len = self.progress_buf.unsqueeze(-1)
        self.stats["tracking_error"].div_(
            torch.where(terminated | truncated, ep_len, torch.ones_like(ep_len))
        )
        self.stats["return"] += reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)

        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(-1),
                },
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )

    def _compute_traj(self, steps: int, env_ids=None, step_size: float=1.):
        if env_ids is None:
            env_ids = ...
        t = self.progress_buf[env_ids].unsqueeze(1) + step_size * torch.arange(steps, device=self.device)
        t = self.traj_t0 + scale_time(self.traj_w[env_ids].unsqueeze(1) * t * self.dt)
        traj_rot = self.traj_rot[env_ids].unsqueeze(1).expand(-1, t.shape[1], 4)

        target_pos = vmap(lemniscate)(t, self.traj_c[env_ids])
        target_pos = vmap(quat_rotate)(traj_rot, target_pos) * self.traj_scale[env_ids].unsqueeze(1)

        return self.origin + target_pos
