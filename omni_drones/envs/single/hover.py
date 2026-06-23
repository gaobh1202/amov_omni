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
import math
import torch
import torch.distributions as D

import omni.isaac.core.utils.prims as prim_utils

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import ArticulationView, RigidPrimView
from omni_drones.utils.torch import euler_to_quaternion, quat_axis

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec


def attach_payload(parent_path):
    from omni.isaac.core import objects
    import omni.physx.scripts.utils as script_utils
    from pxr import UsdPhysics

    payload_prim = objects.DynamicCuboid(
        prim_path=parent_path + "/payload",
        scale=torch.tensor([0.1, 0.1, .15]),
        mass=0.0001
    ).prim

    parent_prim = prim_utils.get_prim_at_path(parent_path + "/base_link")
    stage = prim_utils.get_current_stage()
    joint = script_utils.createJoint(stage, "Prismatic", payload_prim, parent_prim)
    UsdPhysics.DriveAPI.Apply(joint, "linear")
    joint.GetAttribute("physics:lowerLimit").Set(-0.15)
    joint.GetAttribute("physics:upperLimit").Set(0.15)
    joint.GetAttribute("physics:axis").Set("Z")
    joint.GetAttribute("drive:linear:physics:damping").Set(10.)
    joint.GetAttribute("drive:linear:physics:stiffness").Set(10000.)


class Hover(IsaacEnv):
    r"""
    A basic control task. The goal for the agent is to maintain a stable
    position and heading in mid-air without drifting. This task is designed
    to serve as a sanity check.

    ## Observation

    The observation space consists of the following part:

    - `rpos` (3): The position relative to the target hovering position.
    - `drone_state` (16 + `num_rotors`): The basic information of the drone (except its position),
      containing its rotation (in quaternion), velocities (linear and angular),
      heading and up vectors, and the current throttle.
    - `rheading` (3): The difference between the reference heading and the current heading.
    - `time_encoding` (optional): The time encoding, which is a 4-dimensional vector encoding the current
      progress of the episode.

    ## Reward

    - `pos`: Reward computed from the position error to the target position.
    - `heading_alignment`: Reward computed from the alignment of the heading to the target heading.
    - `up`: Reward computed from the uprightness of the drone to discourage large tilting.
    - `spin`: Reward computed from the spin of the drone to discourage spinning.
    - `effort`: Reward computed from the effort of the drone to optimize the
      energy consumption.
    - `action_smoothness`: Reward that encourages smoother drone actions, computed based on the throttle difference of the drone.

    The total reward is computed as follows:

    ```{math}
        r = r_\text{pos} + r_\text{pos} * (r_\text{up} + r_\text{spin}) + r_\text{effort} + r_\text{action_smoothness}
    ```

    ## Episode End

    The episode ends when the drone misbehaves, i.e., it crashes into the ground or flies too far away:

    ```{math}
        d_\text{pos} > 4 \text{ or } x^w_z < 0.2
    ```

    or when the episode reaches the maximum length.

    ## Config

    | Parameter               | Type  | Default   | Description                                                                                                                                                                                                                             |
    | ----------------------- | ----- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | `drone_model`           | str   | "firefly" | Specifies the model of the drone being used in the environment.                                                                                                                                                                         |
    | `reward_distance_scale` | float | 1.2       | Scales the reward based on the distance between the drone and its target.                                                                                                                                                               |
    | `time_encoding`         | bool  | True      | Indicates whether to include time encoding in the observation space. If set to True, a 4-dimensional vector encoding the current progress of the episode is included in the observation. If set to False, this feature is not included. |
    | `has_payload`           | bool  | False     | Indicates whether the drone has a payload attached. If set to True, it means that a payload is attached; otherwise, if set to False, no payload is attached.                                                                            |
    | `hover_target.use_drone_visual` | bool | True | If True (default), spawn the hover goal marker from ``hover_target_usd_path`` / drone USD (e.g. Firefly-shaped ghost). If False, spawn a small cube instead so only the task drone appears as a full model. |
    | `hover_target.usd_path` | str or null | null | Optional USD path for the goal marker. When set, overrides ``use_drone_visual`` / drone defaults. |
    """
    def __init__(self, cfg, headless):
        self.reward_effort_weight = cfg.task.reward_effort_weight
        self.reward_action_smoothness_weight = cfg.task.reward_action_smoothness_weight
        self.reward_distance_scale = cfg.task.reward_distance_scale
        self.reward_up_weight = float(cfg.task.get("reward_up_weight", 0.5))
        self.time_encoding = cfg.task.time_encoding
        self.randomization = cfg.task.get("randomization", {})
        self.has_payload = "payload" in self.randomization.keys()
        self.ground_reset_height = float(cfg.task.get("ground_reset_height", 0.2))
        self.min_up_z = float(cfg.task.get("min_up_z", -1.0))
        _arm_log_cfg = cfg.task.get("arm_log", None)
        self._arm_log_enabled = bool(
            _arm_log_cfg is not None and _arm_log_cfg.get("enabled", False)
        )
        self._arm_log_interval = int(
            _arm_log_cfg.get("log_interval", 200) if _arm_log_cfg else 200
        )
        self._arm_log_counter = 0

        ht = cfg.task.get("hover_target")
        if ht is None:
            self._hover_target_use_drone_visual = True
            self._hover_target_usd_override = None
        else:
            self._hover_target_use_drone_visual = getattr(ht, "use_drone_visual", True)
            o = getattr(ht, "usd_path", None)
            self._hover_target_usd_override = o if o not in (None, "") else None
        self._hover_simple_target_marker = (
            self._hover_target_usd_override is None and not self._hover_target_use_drone_visual
        )

        super().__init__(cfg, headless)

        self.drone.initialize()

        # Arm control setup (for robots with extra arm joints, e.g. AmovDrone).
        self._arm_motion_enabled = False
        self._arm_hold_enabled = False
        self._arm_hold_targets = None
        self._arm_hold_zero_velocity_each_step = False
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
            self._arm_hold_zero_velocity_each_step = bool(
                _arm_hold_cfg.get("zero_velocity_each_step", False)
            )
            logging.info(
                "Arm hold enabled with targets=%s",
                hold_targets.detach().cpu().tolist(),
            )

        _arm_cfg = cfg.task.get("arm_motion", None)
        _arm_motion_requested = _arm_cfg is not None and _arm_cfg.get("enabled", False)
        if _arm_motion_requested and self._arm_hold_enabled:
            logging.info("arm_motion is ignored because arm_hold is enabled.")
        elif _arm_motion_requested and _has_arm_joints:
            self._arm_motion_enabled = True
            self._arm_active_joint = int(_arm_cfg.get("active_joint", 0))
            self._arm_amplitude = float(_arm_cfg.get("amplitude", 0.3))
            self._arm_frequency = float(_arm_cfg.get("frequency", 0.5))

        # Multi-joint sinusoidal trajectory (arm_trajectory).
        # Takes priority over arm_motion but is overridden by arm_hold.
        self._arm_trajectory_enabled = False
        self._arm_traj_amplitudes: torch.Tensor = None
        self._arm_traj_frequencies: torch.Tensor = None
        self._arm_traj_phases: torch.Tensor = None

        _arm_traj_cfg = cfg.task.get("arm_trajectory", None)
        _arm_traj_requested = (
            _arm_traj_cfg is not None
            and _arm_traj_cfg.get("enabled", False)
            and _arm_traj_cfg.get("joints", None) is not None
        )
        if _arm_traj_requested and self._arm_hold_enabled:
            logging.info("arm_trajectory is ignored because arm_hold is enabled.")
        elif _arm_traj_requested and _has_arm_joints:
            self._arm_trajectory_enabled = True
            # Disable single-joint arm_motion when trajectory is active.
            self._arm_motion_enabled = False
            n = self.drone.num_arm_joints
            joints_cfg = list(_arm_traj_cfg.get("joints", []))
            amplitudes = torch.zeros(n, device=self.device)
            frequencies = torch.zeros(n, device=self.device)
            phases = torch.zeros(n, device=self.device)
            for i, jcfg in enumerate(joints_cfg):
                if i >= n:
                    break
                amplitudes[i] = float(jcfg.get("amplitude", 0.3))
                frequencies[i] = float(jcfg.get("frequency", 0.3))
                phases[i] = float(jcfg.get("phase", 0.0))
            self._arm_traj_amplitudes = amplitudes
            self._arm_traj_frequencies = frequencies
            self._arm_traj_phases = phases
            # Apply PD gains for arm joints.
            stiffness_val = float(_arm_traj_cfg.get("stiffness", 500.0))
            damping_val = float(_arm_traj_cfg.get("damping", 50.0))
            stiffness = torch.full(
                (self.drone._view.count, n), stiffness_val, device=self.device
            )
            damping = torch.full(
                (self.drone._view.count, n), damping_val, device=self.device
            )
            self.drone._view.set_gains(
                stiffness, damping, joint_indices=self.drone.arm_joint_indices
            )
            logging.info(
                "arm_trajectory enabled: %d joints | amplitudes=%s | frequencies=%s | stiffness=%.1f | damping=%.1f",
                n, amplitudes.tolist(), frequencies.tolist(), stiffness_val, damping_val,
            )

        if "drone" in self.randomization:
            self.drone.setup_randomization(self.randomization["drone"])
        if "payload" in self.randomization:
            payload_cfg = self.randomization["payload"]
            self.payload_z_dist = D.Uniform(
                torch.tensor([payload_cfg["z"][0]], device=self.device),
                torch.tensor([payload_cfg["z"][1]], device=self.device)
            )
            self.payload_mass_dist = D.Uniform(
                torch.tensor([payload_cfg["mass"][0]], device=self.device),
                torch.tensor([payload_cfg["mass"][1]], device=self.device)
            )
            self.payload = RigidPrimView(
                f"/World/envs/env_*/{self.drone.name}_*/payload",
                reset_xform_properties=False,
                shape=(-1, self.drone.n)
            )
            self.payload.initialize()

        if self._hover_simple_target_marker:
            self.target_vis = RigidPrimView(
                "/World/envs/env_*/target",
                name="hover_target_marker",
                reset_xform_properties=False,
                shape=(-1,),
            )
        else:
            # ArticulationRootAPI may live on a sub-prim (e.g. base_link for AmovDrone).
            # hover_target_root_rel gives the relative path from the spawn root to that prim.
            _root_rel = getattr(self.drone.__class__, "hover_target_root_rel", None)
            _target_view_path = (
                f"/World/envs/env_*/target/{_root_rel}" if _root_rel else "/World/envs/env_*/target"
            )
            self.target_vis = ArticulationView(
                _target_view_path,
                reset_xform_properties=False,
            )
        self.target_vis.initialize()
        self.init_poses = self.drone.get_world_poses(clone=True)
        self.init_vels = torch.zeros_like(self.drone.get_velocities())

        _init_pos_cfg = cfg.task.get("init_pos_range", None)
        if _init_pos_cfg is None:
            _pos_low = torch.tensor([-2.5, -2.5, 1.0], device=self.device)
            _pos_high = torch.tensor([2.5, 2.5, 2.5], device=self.device)
        else:
            _pos_low = torch.tensor(_init_pos_cfg["low"], device=self.device)
            _pos_high = torch.tensor(_init_pos_cfg["high"], device=self.device)
        self.init_pos_dist = D.Uniform(_pos_low, _pos_high)
        init_rpy_cfg = cfg.task.get("init_rpy_range", None)
        if init_rpy_cfg is None:
            init_rpy_low = torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi
            init_rpy_high = torch.tensor([0.2, 0.2, 2.], device=self.device) * torch.pi
        else:
            init_rpy_low = torch.tensor(init_rpy_cfg["low"], device=self.device) * torch.pi
            init_rpy_high = torch.tensor(init_rpy_cfg["high"], device=self.device) * torch.pi
        self.init_rpy_dist = D.Uniform(init_rpy_low, init_rpy_high)
        self.target_rpy_dist = D.Uniform(
            torch.tensor([0., 0., 0.], device=self.device) * torch.pi,
            torch.tensor([0., 0., 2.], device=self.device) * torch.pi
        )

        self.target_pos = torch.tensor([[0.0, 0.0, 2.]], device=self.device)
        self.target_heading = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.alpha = 0.8
        self._has_invalid_state = torch.zeros(
            self.num_envs, 1, dtype=torch.bool, device=self.device
        )

    def _design_scene(self):
        import omni_drones.utils.kit as kit_utils
        import omni.isaac.core.utils.prims as prim_utils

        drone_model_cfg = self.cfg.task.drone_model
        self.drone, self.controller = MultirotorBase.make(
            drone_model_cfg.name, drone_model_cfg.controller
        )

        target_prim_path = "/World/envs/env_0/target"
        if self._hover_target_usd_override is not None:
            target_usd = self._hover_target_usd_override
            target_vis_prim = prim_utils.create_prim(
                prim_path=target_prim_path,
                usd_path=target_usd,
                translation=(0.0, 0.0, 2.0),
            )
            kit_utils.set_nested_collision_properties(
                target_vis_prim.GetPath(),
                collision_enabled=False,
            )
            kit_utils.set_nested_rigid_body_properties(
                target_vis_prim.GetPath(),
                disable_gravity=True,
            )
        elif self._hover_target_use_drone_visual:
            target_usd = getattr(
                self.drone.__class__, "hover_target_usd_path", None
            ) or self.drone.usd_path
            target_vis_prim = prim_utils.create_prim(
                prim_path=target_prim_path,
                usd_path=target_usd,
                translation=(0.0, 0.0, 2.0),
            )
            kit_utils.set_nested_collision_properties(
                target_vis_prim.GetPath(),
                collision_enabled=False,
            )
            kit_utils.set_nested_rigid_body_properties(
                target_vis_prim.GetPath(),
                disable_gravity=True,
            )
        else:
            from omni.isaac.core.objects import DynamicCuboid

            DynamicCuboid(
                target_prim_path,
                translation=torch.tensor([0.0, 0.0, 2.0]),
                scale=torch.tensor([0.22, 0.22, 0.22]),
                color=torch.tensor([0.15, 0.85, 0.25]),
                size=1.0,
            )
            kit_utils.set_collision_properties(
                target_prim_path,
                collision_enabled=False,
            )
            kit_utils.set_rigid_body_properties(
                target_prim_path,
                disable_gravity=True,
            )

        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 2.)])[0]
        if self.has_payload:
            attach_payload(drone_prim.GetPath().pathString)
        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1]
        observation_dim = drone_state_dim + 3

        if self.cfg.task.time_encoding:
            self.time_encoding_dim = 4
            observation_dim += self.time_encoding_dim

        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": UnboundedContinuousTensorSpec((1, observation_dim), device=self.device),
                "intrinsics": self.drone.intrinsics_spec.unsqueeze(0).to(self.device)
            })
        }).expand(self.num_envs).to(self.device)
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": self.drone.action_spec.unsqueeze(0),
            })
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1, 1))
            })
        }).expand(self.num_envs).to(self.device)

        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "intrinsics")
        )

        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "pos_error": UnboundedContinuousTensorSpec(1),
            "heading_alignment": UnboundedContinuousTensorSpec(1),
            "uprightness": UnboundedContinuousTensorSpec(1),
            "action_smoothness": UnboundedContinuousTensorSpec(1),
            "mean_throttle": UnboundedContinuousTensorSpec(1),
            "mean_thrust_n": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids, self.training)
        if self._arm_hold_enabled:
            self.drone.reset_arm_joints(env_ids, self._arm_hold_targets)
        elif self._arm_motion_enabled:
            self.drone.reset_arm_joints(env_ids)
        elif self._arm_trajectory_enabled:
            self.drone.reset_arm_joints(env_ids)

        pos = self.init_pos_dist.sample((*env_ids.shape, 1))
        rpy = self.init_rpy_dist.sample((*env_ids.shape, 1))
        rot = euler_to_quaternion(rpy)
        self.drone.set_world_poses(
            pos + self.envs_positions[env_ids].unsqueeze(1), rot, env_ids
        )
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)

        if self.has_payload:
            # TODO@btx0424: workout a better way
            payload_z = self.payload_z_dist.sample(env_ids.shape)
            joint_indices = torch.tensor([self.drone._view._dof_indices["PrismaticJoint"]], device=self.device)
            self.drone._view.set_joint_positions(
                payload_z, env_indices=env_ids, joint_indices=joint_indices)
            self.drone._view.set_joint_position_targets(
                payload_z, env_indices=env_ids, joint_indices=joint_indices)
            self.drone._view.set_joint_velocities(
                torch.zeros(len(env_ids), 1, device=self.device),
                env_indices=env_ids, joint_indices=joint_indices)

            payload_mass = self.payload_mass_dist.sample(env_ids.shape+(1,)) * self.drone.masses[env_ids]
            self.payload.set_masses(payload_mass, env_indices=env_ids)

        target_rpy = self.target_rpy_dist.sample((*env_ids.shape, 1))
        target_rot = euler_to_quaternion(target_rpy)
        self.target_heading[env_ids] = quat_axis(target_rot.squeeze(1), 0).unsqueeze(1)
        self.target_vis.set_world_poses(orientations=target_rot, env_indices=env_ids)

        self.stats[env_ids] = 0.

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
        actions = actions.clamp(-1.0, 1.0)

        # Normalize action space so that:
        #   policy output = 0  →  hover thrust  (center of action space)
        #   policy output = ±1 →  ±scale deviation around hover, NO clamping
        #
        # RotorGroup maps: target_throttle = sqrt((cmd+1)/2), thrust = throttle^2 * KF.
        # Hover requires cmd = 2*(mg/KF_sum) - 1.
        #
        # For AmovDrone: hover_cmd ≈ -0.484.
        # Without scaling, action + offset maps to [-1.484, 0.516] → ~33% of random
        # samples are clamped to cmd=-1 (zero thrust), biasing mean thrust far below hover.
        # For Firefly: hover_cmd ≈ -0.183, only ~18% clamped → trains OK without fix.
        # With scaling by (1 - |offset|), actions in [-1,1] map exactly to [-1, 0.032]
        # with zero clamping → symmetric exploration around hover.
        if not hasattr(self, "_hover_cmd_offset"):
            with torch.no_grad():
                _g = self.drone.gravity.mean()
                _kf = self.drone.KF.sum(-1).mean().clamp_min(1e-9)
                _hover_frac = (_g / _kf).clamp(0.0, 1.0)
                self._hover_cmd_offset = float((2.0 * _hover_frac - 1.0).item())
                # Symmetric scale: ±1 action maps to ±scale around hover_cmd, no clamp
                self._hover_cmd_scale = float(1.0 - abs(self._hover_cmd_offset))
            logging.info(
                "Hover action normalization: hover_frac=%.4f  hover_cmd=%.4f  scale=%.4f"
                "  (action=0 → hover, action=±1 → cmd=%.3f/%.3f)",
                float(_hover_frac.item()),
                self._hover_cmd_offset,
                self._hover_cmd_scale,
                self._hover_cmd_offset - self._hover_cmd_scale,
                self._hover_cmd_offset + self._hover_cmd_scale,
            )
        actions = (actions * self._hover_cmd_scale + self._hover_cmd_offset).clamp(-1.0, 1.0)

        self.effort = self.drone.apply_action(actions)
        if self._arm_hold_enabled:
            self.drone.hold_arm_joints(
                self._arm_hold_targets,
                zero_velocity=self._arm_hold_zero_velocity_each_step,
            )
        elif self._arm_motion_enabled:
            self.drone.apply_arm_motion(
                self.progress_buf,
                self.dt * self.substeps,
                active_joint=self._arm_active_joint,
                amplitude=self._arm_amplitude,
                frequency=self._arm_frequency,
            )
        elif self._arm_trajectory_enabled:
            self._apply_arm_trajectory()

    def _apply_arm_trajectory(self):
        """Drive all arm joints with independent sinusoidal trajectories."""
        if self.drone.arm_joint_indices is None or self.drone.num_arm_joints == 0:
            return
        t = self.progress_buf.float() * (self.dt * self.substeps)  # (num_envs,)
        pos = (
            self._arm_traj_amplitudes.unsqueeze(0)
            * torch.sin(
                2.0 * math.pi * self._arm_traj_frequencies.unsqueeze(0) * t.unsqueeze(1)
                + self._arm_traj_phases.unsqueeze(0)
            )
        )  # (num_envs, num_arm_joints)
        self.drone._view.set_joint_position_targets(
            pos, joint_indices=self.drone.arm_joint_indices
        )

    def _compute_state_and_obs(self):
        self.drone_state = self.drone.get_state()
        self._has_invalid_state = ~torch.isfinite(self.drone_state).all(-1)
        safe_drone_state = torch.nan_to_num(
            self.drone_state, nan=0.0, posinf=0.0, neginf=0.0
        )

        self.rpos = self.target_pos - safe_drone_state[..., :3]
        self.rheading = self.target_heading - safe_drone_state[..., 13:16]

        obs = [self.rpos, safe_drone_state[..., 3:], self.rheading]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            obs.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))
        obs = torch.cat(obs, dim=-1)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                    "intrinsics": self.drone.intrinsics,
                },
                "stats": self.stats.clone(),
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        safe_rpos = torch.nan_to_num(self.rpos, nan=0.0, posinf=0.0, neginf=0.0)
        safe_rheading = torch.nan_to_num(self.rheading, nan=0.0, posinf=0.0, neginf=0.0)
        safe_heading = torch.nan_to_num(self.drone.heading, nan=0.0, posinf=0.0, neginf=0.0)
        safe_up = torch.nan_to_num(self.drone.up, nan=0.0, posinf=0.0, neginf=0.0)
        safe_vel = torch.nan_to_num(self.drone.vel, nan=0.0, posinf=0.0, neginf=0.0)
        safe_effort = torch.nan_to_num(self.effort, nan=0.0, posinf=1e6, neginf=0.0)
        safe_throttle_diff = torch.nan_to_num(
            self.drone.throttle_difference, nan=0.0, posinf=1e6, neginf=0.0
        )

        # pose reward
        pos_error = torch.norm(safe_rpos, dim=-1)
        heading_alignment = torch.sum(safe_heading * self.target_heading, dim=-1)

        distance = torch.norm(torch.cat([safe_rpos, safe_rheading], dim=-1), dim=-1)

        reward_pose = 1.0 / (1.0 + torch.square(self.reward_distance_scale * distance))
        # pose_reward = torch.exp(-distance * self.reward_distance_scale)
        # uprightness
        reward_up = torch.square((safe_up[..., 2] + 1) / 2)

        # spin reward
        spinnage = torch.square(safe_vel[..., -1])
        reward_spin = 1.0 / (1.0 + torch.square(spinnage))

        # effort
        reward_effort = self.reward_effort_weight * torch.exp(-safe_effort)
        reward_action_smoothness = self.reward_action_smoothness_weight * torch.exp(-safe_throttle_diff)

        # survival: height-proportional reward normalized between ground and target height.
        # r_alive = 0 at ground_reset_height, 1.0 at target z (2.0m), capped above.
        # This provides a continuous upward gradient regardless of spawn height.
        safe_pos_z_for_alive = torch.nan_to_num(self.drone.pos[..., 2], nan=-1.0)
        target_z = float(self.target_pos[..., 2].mean().item())  # target height (default 2.0m)
        alive_denom = max(target_z - self.ground_reset_height, 0.1)
        reward_alive = torch.clamp(
            (safe_pos_z_for_alive - self.ground_reset_height) / alive_denom, 0.0, 1.0
        )

        # Standalone uprightness reward: incentivizes upright attitude independent
        # of position error.  reward_up (above) is gated by reward_pose (small when
        # far from target), so early in training when pos_error is large the pose-
        # gated signal is weak.  This standalone term provides a direct gradient to
        # keep the drone right-side-up from the first episode.
        reward_up_standalone = self.reward_up_weight * ((safe_up[..., 2] + 1.0) / 2.0)

        assert reward_pose.shape == reward_up.shape == reward_spin.shape
        reward = (
            reward_pose
            + reward_pose * (reward_up + reward_spin)
            + reward_effort
            + reward_action_smoothness
            + reward_alive
            + reward_up_standalone
        )
        reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

        safe_pos_z = torch.nan_to_num(self.drone.pos[..., 2], nan=-1.0, posinf=1e6, neginf=-1e6)
        misbehave = (safe_pos_z < self.ground_reset_height) | (distance > 4)
        if self.min_up_z > -1.0:
            misbehave = misbehave | (safe_up[..., 2] < self.min_up_z)
        hasnan = self._has_invalid_state

        terminated = misbehave | hasnan
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        reward = reward.masked_fill(hasnan, 0.0)

        # ── termination diagnosis (prints once every ~200 env-resets) ──────────
        if not hasattr(self, "_term_diag_counter"):
            self._term_diag_counter = 0
        any_done = (terminated | truncated).any()
        if any_done:
            self._term_diag_counter += int((terminated | truncated).sum().item())
            if self._term_diag_counter >= 50:
                self._term_diag_counter = 0
                env_ids = (terminated | truncated).squeeze(-1).nonzero(as_tuple=False).squeeze(-1)
                if env_ids.numel() > 0:
                    _eid = env_ids[0]
                    _z   = safe_pos_z[_eid].item()
                    _d   = distance[_eid].item()
                    _upz = safe_up[_eid, ..., 2].mean().item()
                    _nan = hasnan[_eid].any().item()
                    _len = self.progress_buf[_eid].item()
                    _thr = self.drone.throttle[_eid].mean().item()
                    _frc = self.drone.thrusts[_eid, ..., 2].sum().item()
                    logging.info(
                        "[TermDiag] env=%d  len=%d  z=%.3f(reset<%.2f)  dist=%.2f(>4?)  up_z=%.3f  hasnan=%s"
                        "  thr=%.4f  thrust=%.1fN  hover=%.1fN",
                        int(_eid), int(_len), _z, self.ground_reset_height,
                        _d, _upz, str(_nan), _thr, _frc,
                        float(self.drone.gravity.mean().item()),
                    )
        # ──────────────────────────────────────────────────────────────────────

        self.stats["pos_error"].lerp_(torch.nan_to_num(pos_error, nan=0.0), (1-self.alpha))
        self.stats["heading_alignment"].lerp_(torch.nan_to_num(heading_alignment, nan=0.0), (1-self.alpha))
        safe_uprightness = torch.nan_to_num(self.drone_state[..., 18], nan=0.0, posinf=0.0, neginf=0.0)
        self.stats["uprightness"].lerp_(safe_uprightness, (1-self.alpha))
        self.stats["action_smoothness"].lerp_(
            torch.nan_to_num(-safe_throttle_diff, nan=0.0), (1-self.alpha)
        )
        # throttle and thrust diagnostics (smoothed per-env)
        _mean_thr = self.drone.throttle.mean(dim=-1, keepdim=True)  # mean over rotors → (envs,1,1) or (envs,1)
        if _mean_thr.dim() == 3:
            _mean_thr = _mean_thr.squeeze(-1)  # → (envs, 1)
        self.stats["mean_throttle"].lerp_(torch.nan_to_num(_mean_thr, nan=0.0), (1 - self.alpha))
        _thrust_sum = self.drone.thrusts[..., 2].sum(dim=-1, keepdim=True)  # total thrust per env (N)
        if _thrust_sum.dim() == 3:
            _thrust_sum = _thrust_sum.squeeze(-1)
        self.stats["mean_thrust_n"].lerp_(torch.nan_to_num(_thrust_sum, nan=0.0), (1 - self.alpha))
        self.stats["return"] += reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)

        # Log arm joint positions periodically (only when arm joints exist).
        if self._arm_log_enabled:
            _has_arm = (
                hasattr(self.drone, "arm_joint_indices")
                and self.drone.arm_joint_indices is not None
            )
            if _has_arm:
                self._arm_log_counter += 1
                if self._arm_log_counter >= self._arm_log_interval:
                    self._arm_log_counter = 0
                    all_pos = self.drone._view.get_joint_positions()  # (num_envs, [1,] num_dof)
                    arm_pos = all_pos.flatten(0, -2)[..., self.drone.arm_joint_indices]  # (num_envs, num_arm_joints)
                    mean_pos = arm_pos.mean(dim=0).detach().cpu().flatten().tolist()
                    names = self.drone._arm_dof_names
                    joint_str = "  ".join(
                        f"{n}={v:.4f}" for n, v in zip(names, mean_pos)
                    )
                    logging.info("[ArmJoints] step=%d  %s", int(self.progress_buf.float().mean().item()), joint_str)

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
