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
import os
from pathlib import Path
from typing import List, Optional

import omni.isaac.core.utils.prims as prim_utils
import torch
from pxr import UsdPhysics
from tensordict.nn import make_functional
from torch.func import vmap

from omni_drones.actuators.rotor_group import RotorGroup
from omni_drones.robots.drone.multirotor import MultirotorBase
from omni_drones.robots.robot import ASSET_PATH
from omni_drones.utils.torch import quat_axis, quat_rotate
from omni_drones.views import RigidPrimView


def _resolve_amov_usd_path() -> str:
    default_path = str(Path(__file__).resolve().parents[3] / "amov_drone_arm_0.usd")
    return os.environ.get("OMNIDRONES_AMOV_USD_PATH", default_path)


def _resolve_amov_param_path() -> str:
    """Resolve rotor/airframe params to match the selected AMOV USD.

    Priority:
    1) explicit override via OMNIDRONES_AMOV_PARAM_PATH
    2) YAML with the same stem as the selected USD under assets/usd
    3) legacy fallback amov_drone_arm_0.yaml
    """
    explicit = os.environ.get("OMNIDRONES_AMOV_PARAM_PATH")
    if explicit:
        return explicit

    usd_path = Path(_resolve_amov_usd_path())
    candidate = Path(ASSET_PATH) / "usd" / f"{usd_path.stem}.yaml"
    if candidate.exists():
        return str(candidate)

    return str(Path(ASSET_PATH) / "usd" / "amov_drone_arm_0.yaml")


class AmovDrone(MultirotorBase):
    """Amov URDF→USD（Isaac Sim）：``ArticulationRootAPI`` 在 ``base_link`` 上。

    USD 结构（amov_drone.usd）：
      - /amov_drone  [Xform]
        - base_link  [RigidBody, MassAPI, ArticulationRootAPI]  ← 关节链根
          - rotor_joint_1…4 [RevoluteJoint, damping=0]          ← 旋翼接头
        - rotor_link_1…4 [RigidBody, MassAPI ~5g]               ← 旋翼连杆（关节链子体）

    旋翼连杆是关节链的子刚体，而非独立 RigidDynamic。
    spawn 时在候选路径上探测 ``ArticulationRootAPI``，
    并缓存相对路径供 ``initialize`` 与旋翼 glob 使用。
    """

    usd_path: str = _resolve_amov_usd_path()
    param_path: str = _resolve_amov_param_path()
    hover_target_usd_path: str = _resolve_amov_usd_path()
    # ArticulationRootAPI lives on base_link (not the spawn root Xform).
    hover_target_root_rel: str = "base_link"

    def __init__(self, name: str = None, cfg=None, is_articulation: bool = True) -> None:
        super().__init__(name, cfg, is_articulation=is_articulation)
        self._amov_articulation_rel: Optional[str] = None
        self._amov_rotor_glob_rel: str = "rotor_link_*"

    @staticmethod
    def _env_flag(name: str, default: str = "0") -> bool:
        val = os.environ.get(name, default).strip().lower()
        return val in {"1", "true", "yes", "on"}

    def _articulation_properties_prim_path(self, spawn_root_prim_path: str) -> str:
        spawn_root_prim_path = spawn_root_prim_path.rstrip("/")
        candidates = [
            # USD default prim (/amov_drone_arm) is merged into spawn root on reference,
            # so base_link appears directly under the spawn root.
            f"{spawn_root_prim_path}/base_link",
            f"{spawn_root_prim_path}/Geometry/base_link",
            f"{spawn_root_prim_path}/amov_drone/base_link",
            # Legacy: root_joint (old USD before floating-base fix)
            f"{spawn_root_prim_path}/root_joint",
            spawn_root_prim_path,
        ]
        for abs_path in candidates:
            prim = prim_utils.get_prim_at_path(abs_path)
            if prim and prim.IsValid() and UsdPhysics.ArticulationRootAPI(prim):
                rel = abs_path[len(spawn_root_prim_path) :].lstrip("/")
                self._amov_articulation_rel = rel
                if abs_path.endswith("/Geometry/base_link"):
                    self._amov_rotor_glob_rel = "Geometry/base_link/rotor_link_*"
                elif abs_path.endswith("/base_link"):
                    parent_of_base = abs_path[: -len("/base_link")]
                    rotor_prefix = parent_of_base[len(spawn_root_prim_path) :].lstrip("/")
                    self._amov_rotor_glob_rel = (
                        f"{rotor_prefix}/rotor_link_*" if rotor_prefix else "rotor_link_*"
                    )
                else:
                    self._amov_rotor_glob_rel = "rotor_link_*"
                return abs_path
        tried = ", ".join(candidates)
        raise ValueError(
            "AmovDrone：在 spawn 根下未找到带 UsdPhysics.ArticulationRootAPI 的 prim。"
            f" 已尝试: {tried}。请确认 URDF→USD 关节链完整，或在 Isaac 中检查 articulation root。"
        )

    def initialize(self, prim_paths_expr: str = None, track_contact_forces: bool = False):
        if not self.is_articulation:
            raise ValueError(
                "AmovDrone 为 URDF 关节链资产，必须 is_articulation=True；"
                "请勿对 AmovDrone 使用单刚体 RigidPrimView 模式。"
            )

        if prim_paths_expr is None:
            root_expr = f"/World/envs/.*/{self.name}_.*"
        else:
            root_expr = prim_paths_expr

        if self._amov_articulation_rel is None:
            raise RuntimeError(
                "AmovDrone.initialize 须在 spawn 之后调用；未缓存 articulation 路径。"
            )
        articulation_expr = (
            f"{root_expr}/{self._amov_articulation_rel}"
            if self._amov_articulation_rel
            else root_expr
        )
        super(MultirotorBase, self).initialize(prim_paths_expr=articulation_expr)

        # articulation 根 prim 与 RigidBodyView 行数不一致时，再包一层 RigidPrimView 会在 get_world_poses 触发 CUDA OOB。
        # 与 MultirotorBase 非 articulation 分支一致：底座即 ``ArticulationView``（见 views.ArticulationView 根连杆 API）。
        self.base_link = self._view

        all_dof_names_view = list(self._view._dof_names)
        logging.info(
            "AmovDrone: ArticulationView num_dof=%d, dof_names=%s",
            self._view._num_dof,
            all_dof_names_view,
        )
        rotor_joint_indices = [
            i for i, dof_name in enumerate(all_dof_names_view) if dof_name.startswith("rotor")
        ]
        logging.info(
            "AmovDrone: rotor_joint_indices=%s",
            rotor_joint_indices if rotor_joint_indices else "None (no rotor DOFs detected!)",
        )
        if len(rotor_joint_indices):
            self.rotor_joint_indices = torch.tensor(rotor_joint_indices, device=self.device)
        else:
            self.rotor_joint_indices = None

        self.rotors_view = RigidPrimView(
            prim_paths_expr=f"{root_expr}/{self._amov_rotor_glob_rel}",
            name="rotors",
            shape=(*self.shape, self.num_rotors),
        )
        self.rotors_view.initialize()

        if self._env_flag("OMNIDRONES_AMOV_DISABLE_ROTOR_COLLISIONS"):
            disabled_cnt = 0
            rotor_paths = getattr(self.rotors_view, "prim_paths", [])
            for rotor_path in rotor_paths:
                for candidate in (f"{rotor_path}/collisions", rotor_path):
                    prim = prim_utils.get_prim_at_path(candidate)
                    if not prim or not prim.IsValid():
                        continue
                    collision_enabled = prim.GetAttribute("physics:collisionEnabled")
                    if collision_enabled and collision_enabled.IsValid():
                        collision_enabled.Set(False)
                        disabled_cnt += 1
                        break
            logging.warning(
                "AmovDrone test override: disabled collision on %d rotor prims",
                disabled_cnt,
            )

        if self._env_flag("OMNIDRONES_AMOV_LIGHT_ROTORS"):
            rotor_mass = float(os.environ.get("OMNIDRONES_AMOV_ROTOR_MASS", "0.001"))
            rotor_masses = torch.full(
                (*self.shape, self.num_rotors), rotor_mass, device=self.device
            )
            self.rotors_view.set_masses(rotor_masses)
            logging.warning(
                "AmovDrone test override: set rotor masses to %.6f kg",
                rotor_mass,
            )

        rotor_config = self.params["rotor_configuration"]
        self.rotors = RotorGroup(rotor_config, dt=self.dt).to(self.device)
        rotor_params = make_functional(self.rotors)
        self.KF_0 = rotor_params["KF"].clone()
        self.KM_0 = rotor_params["KM"].clone()
        self.MAX_ROT_VEL = torch.as_tensor(rotor_config["max_rotation_velocities"]).float().to(self.device)
        self.rotor_params = rotor_params.expand(self.shape).clone()

        self.tau_up = self.rotor_params["tau_up"]
        self.tau_down = self.rotor_params["tau_down"]
        self.KF = self.rotor_params["KF"]
        self.KM = self.rotor_params["KM"]
        self.throttle = self.rotor_params["throttle"]
        self.directions = self.rotor_params["directions"]

        # Optional yaw reaction-torque scaling for quick isolation tests.
        # Keep default at 1.0 so existing behavior is unchanged.
        km_scale = float(os.environ.get("OMNIDRONES_AMOV_KM_SCALE", "1.0"))
        if abs(km_scale - 1.0) > 1e-9:
            self.KM_0.mul_(km_scale)
            self.KM.mul_(km_scale)
            logging.warning(
                "AmovDrone: applying KM scale %.4f via OMNIDRONES_AMOV_KM_SCALE",
                km_scale,
            )

        self.thrusts = torch.zeros(*self.shape, self.num_rotors, 3, device=self.device)
        self.torques = torch.zeros(*self.shape, 3, device=self.device)
        self.forces = torch.zeros(*self.shape, 3, device=self.device)

        self.pos, self.rot = self.get_world_poses(True)
        self.throttle_difference = torch.zeros(self.throttle.shape[:-1], device=self.device)
        self.heading = torch.zeros(*self.shape, 3, device=self.device)
        self.up = torch.zeros(*self.shape, 3, device=self.device)
        self.vel = self.vel_w = torch.zeros(*self.shape, 6, device=self.device)
        self.vel_b = torch.zeros_like(self.vel_w)
        self.acc = self.acc_w = torch.zeros(*self.shape, 6, device=self.device)
        self.acc_b = torch.zeros_like(self.acc_w)

        self.alpha = 0.9
        # For articulated AMOV, root-link mass is smaller than total articulated mass.
        # Use summed body masses for hover-throttle and force scaling.
        self.root_link_masses = self.base_link.get_masses().clone()
        body_masses = self._view.get_body_masses(clone=True)
        self.masses = body_masses.sum(dim=-1, keepdim=True)
        self.gravity = self.masses * 9.81
        self.inertias = self.base_link.get_inertias().reshape(*self.shape, 3, 3).diagonal(0, -2, -1)
        self.MASS_0 = self.masses[0].clone()
        self.INERTIA_0 = self.base_link.get_inertias().reshape(*self.shape, 3, 3)[0].diagonal(0, -2, -1).clone()
        self.THRUST2WEIGHT_0 = self.KF_0 / (self.MASS_0 * 9.81)
        self.FORCE2MOMENT_0 = torch.broadcast_to(self.KF_0 / self.KM_0, self.THRUST2WEIGHT_0.shape)

        self.drag_coef = torch.zeros(*self.shape, 1, device=self.device) * self.params["drag_coef"]
        self.intrinsics = self.intrinsics_spec.expand(self.shape).zero()

        # Rotor geometry sanity check (first env, first drone instance).
        # Large XY offset from base center or strong radius asymmetry usually
        # indicates rotor prim binding/path issues and can cause heavy coupling.
        try:
            rotor_pos, _ = self.rotors_view.get_world_poses()
            base_pos, _ = self.base_link.get_world_poses()
            rel = (rotor_pos[0, 0] - base_pos[0, 0].unsqueeze(0)).float()
            rel_xy = rel[:, :2]
            mean_xy = rel_xy.mean(dim=0)
            radial = torch.norm(rel_xy - mean_xy, dim=-1)
            radial_spread = (radial.max() - radial.min()).item()
            mean_xy_norm = torch.norm(mean_xy).item()
            logging.info(
                "AmovDrone rotor geometry: mean_xy=(%.4f, %.4f), mean_xy_norm=%.4f, radial_spread=%.4f",
                mean_xy[0].item(),
                mean_xy[1].item(),
                mean_xy_norm,
                radial_spread,
            )
            if mean_xy_norm > 0.15 or radial_spread > 0.12:
                logging.warning(
                    "AmovDrone rotor geometry looks asymmetric (mean_xy_norm=%.4f, radial_spread=%.4f). "
                    "Check rotor prim binding and rotor frame alignment.",
                    mean_xy_norm,
                    radial_spread,
                )

            # Additional startup diagnostics: COM offset wrt rotor centroid and
            # root-link inertia principal axes.
            rotor_center = rotor_pos[0, 0].mean(dim=0)
            # Use prim-origin world position as COM approximation (get_body_coms
            # returns LOCAL frame offset, which is only correct near world origin).
            root_com = base_pos[0, 0]
            com_offset = (root_com - rotor_center).detach().cpu()

            inertia_flat = self.base_link.get_inertias()[0, 0].reshape(3, 3).float()
            eigvals, eigvecs = torch.linalg.eigh(inertia_flat)
            logging.info(
                "AmovDrone COM wrt rotor centroid: [%.4f, %.4f, %.4f] m",
                float(com_offset[0].item()),
                float(com_offset[1].item()),
                float(com_offset[2].item()),
            )
            logging.info(
                "AmovDrone root inertia principal moments: [%.6f, %.6f, %.6f]",
                float(eigvals[0].item()),
                float(eigvals[1].item()),
                float(eigvals[2].item()),
            )
            logging.info(
                "AmovDrone root inertia principal axis(v1): [%.4f, %.4f, %.4f]",
                float(eigvecs[0, 0].item()),
                float(eigvecs[1, 0].item()),
                float(eigvecs[2, 0].item()),
            )
        except Exception as exc:
            logging.warning("AmovDrone rotor geometry check skipped: %s", exc)

        # Detect arm joints (all DOFs that are not rotor joints)
        all_dof_names: List[str] = list(self._view._dof_names)
        arm_indices = [i for i, n in enumerate(all_dof_names) if not n.startswith("rotor")]
        if arm_indices:
            self.arm_joint_indices = torch.tensor(arm_indices, device=self.device)
            self.num_arm_joints = len(arm_indices)
            self._arm_dof_names = [all_dof_names[i] for i in arm_indices]
            logging.info(
                "AmovDrone: %d arm joints detected: %s",
                self.num_arm_joints,
                self._arm_dof_names,
            )
        else:
            self.arm_joint_indices = None
            self.num_arm_joints = 0
            self._arm_dof_names = []

        logging.info(str(self))

    def _expand_arm_targets(
        self, joint_targets: Optional[torch.Tensor], count: int
    ) -> torch.Tensor:
        if self.arm_joint_indices is None or self.num_arm_joints == 0:
            return torch.empty(count, 0, device=self.device)
        if joint_targets is None:
            base = torch.zeros(self.num_arm_joints, device=self.device)
        else:
            base = torch.as_tensor(joint_targets, dtype=torch.float32, device=self.device).flatten()
            if base.numel() < self.num_arm_joints:
                padded = torch.zeros(self.num_arm_joints, device=self.device)
                padded[: base.numel()] = base
                base = padded
            elif base.numel() > self.num_arm_joints:
                base = base[: self.num_arm_joints]
        return base.unsqueeze(0).expand(count, -1).contiguous()

    def reset_arm_joints(
        self, env_ids: torch.Tensor, joint_targets: Optional[torch.Tensor] = None
    ) -> None:
        """Reset arm joints to the specified position and zero velocity."""
        if self.arm_joint_indices is None:
            return
        targets = self._expand_arm_targets(joint_targets, len(env_ids))
        self._view.set_joint_positions(
            targets, env_indices=env_ids, joint_indices=self.arm_joint_indices
        )
        self._view.set_joint_velocities(
            torch.zeros_like(targets), env_indices=env_ids, joint_indices=self.arm_joint_indices
        )
        self._view.set_joint_position_targets(
            targets, env_indices=env_ids, joint_indices=self.arm_joint_indices
        )

    def hold_arm_joints(
        self, joint_targets: Optional[torch.Tensor] = None, zero_velocity: bool = False
    ) -> None:
        """Hold arm joints at the specified target during simulation steps."""
        if self.arm_joint_indices is None:
            return
        targets = self._expand_arm_targets(joint_targets, self._view.count)
        self._view.set_joint_position_targets(targets, joint_indices=self.arm_joint_indices)
        if zero_velocity:
            self._view.set_joint_velocities(
                torch.zeros_like(targets), joint_indices=self.arm_joint_indices
            )

    def apply_action(self, actions: torch.Tensor) -> torch.Tensor:
        """Override MultirotorBase.apply_action for AmovDrone articulation.

        Aggregates per-rotor thrust vectors into a net force + pitch/roll torques,
        then applies everything to base_link via the ArticulationView force API.
        This is required because applying forces to articulation child links via
        RigidPrimView is silently ignored by PhysX when the child links are proper
        articulation bodies.

        Rotor spinning is driven via set_joint_velocities (safe: damping=0 in USD).
        """
        rotor_cmds = actions.expand(*self.shape, self.num_rotors)
        last_throttle = self.throttle.clone()
        thrusts, moments = vmap(vmap(self.rotors, randomness="different"), randomness="same")(
            rotor_cmds, self.rotor_params
        )

        rotor_pos, rotor_rot = self.rotors_view.get_world_poses()
        # World-frame thrust direction of each rotor (unit vec along rotor z-axis)
        torque_axis = quat_axis(rotor_rot.flatten(end_dim=-2), axis=2).unflatten(
            0, (*self.shape, self.num_rotors)
        )

        self.thrusts[..., 2] = thrusts
        # Yaw reaction-torques from spinning rotors
        self.torques[:] = (moments.unsqueeze(-1) * torque_axis).sum(-2)

        # Drive visual rotor rotation (damping=0 in amov_drone.usd, so no braking torque).
        # At actual throttle (~0.5) × MAX_ROT_VEL (436 rad/s) the joint turns ~200° per
        # 16ms physics step, causing a stroboscopic aliasing effect where propellers appear
        # frozen.  Scale down to a perceptible visual speed (≤ _VISUAL_MAX_RAD_S) while
        # keeping physics forces correct (handled via base_link force aggregation).
        _VISUAL_MAX_RAD_S = 15.0  # ~143 RPM at full throttle, clearly visible
        if self.is_articulation and self.rotor_joint_indices is not None:
            rot_vel = self.throttle * self.directions * self.MAX_ROT_VEL
            vis_scale = _VISUAL_MAX_RAD_S / self.MAX_ROT_VEL.max().clamp(min=1.0)
            self._view.set_joint_velocities(
                (rot_vel * vis_scale).reshape(-1, self.num_rotors),
                joint_indices=self.rotor_joint_indices,
            )

        # Drag / downwash
        self.forces.zero_()
        if self.n > 1:
            self.forces[:] += vmap(self.downwash)(
                self.pos, self.pos,
                quat_rotate(self.rot, self.thrusts.sum(-2)),
                kz=0.3,
            ).sum(-2)
        self.forces[:] += (self.drag_coef * self.masses) * self.vel[..., :3]

        # ── aggregate rotor thrusts → base_link force + pitch/roll torques ──
        # thrust_world: (*shape, num_rotors, 3) – each rotor's force in world frame
        thrust_world = thrusts.unsqueeze(-1) * torque_axis

        # Net upward (and lateral, for a tilted drone) force
        net_thrust = thrust_world.sum(dim=-2)  # (*shape, 3)

        # Pitch/roll torques: Σ (r_i - r_ref) × F_i_world
        # IMPORTANT: Use get_world_poses() for the reference point, NOT get_coms().
        # ArticulationView.get_body_coms() returns the COM offset in the LOCAL body
        # frame (~[0.013, 0.010, -0.044] m for AmovDrone).  When environments are
        # spread across the simulation grid (8 m spacing), rotor_pos is in world
        # frame (up to ~80 m from origin) while the local COM offset is ~0.04 m.
        # Using the local offset as the reference creates lever arms of ~80 m
        # instead of ~0.48 m, producing tens of thousands of N·m of spurious
        # pitch/roll torque and causing immediate drone flip.
        # get_world_poses() returns the prim-origin world position, which differs
        # from the true COM by only ~4.4 cm, introducing a negligible torque error.
        ref_pos, _ = self.base_link.get_world_poses()    # world frame, (*shape, 3)
        rel = rotor_pos - ref_pos.unsqueeze(-2)   # (*shape, num_rotors, 3)
        pitch_roll = torch.cross(rel, thrust_world, dim=-1).sum(dim=-2)  # (*shape, 3)

        self.base_link.apply_forces_and_torques_at_pos(
            (self.forces + net_thrust).reshape(-1, 3),
            (self.torques + pitch_roll).reshape(-1, 3),
            is_global=True,
        )
        # ────────────────────────────────────────────────────────────────────

        self.throttle_difference[:] = torch.norm(self.throttle - last_throttle, dim=-1)
        return self.throttle.sum(-1)

    def apply_arm_motion(
        self,
        progress_buf: torch.Tensor,
        dt: float,
        active_joint: int = 0,
        amplitude: float = 0.3,
        frequency: float = 0.5,
    ) -> None:
        """Drive one arm joint with sinusoidal motion; hold all others at zero.

        Args:
            progress_buf: episode step counter, shape (num_envs,).
            dt: simulation timestep in seconds.
            active_joint: index into arm_joint_indices to animate.
            amplitude: peak displacement in radians (or metres for prismatic).
            frequency: oscillation frequency in Hz.
        """
        if self.arm_joint_indices is None or self.num_arm_joints == 0:
            return
        t = progress_buf.float() * dt  # (num_envs,)
        pos = torch.zeros(self._view.count, self.num_arm_joints, device=self.device)
        if active_joint < self.num_arm_joints:
            pos[:, active_joint] = amplitude * torch.sin(
                2.0 * math.pi * frequency * t
            )
        self._view.set_joint_position_targets(pos, joint_indices=self.arm_joint_indices)

