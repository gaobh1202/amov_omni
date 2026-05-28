"""AMOV hover readiness audit with Firefly-referenced checks.

Usage examples:
  python test_amov_thrust.py
  python test_amov_thrust.py --task HoverAmov --steps 600 --headless
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from omni_drones import init_simulation_app

_REPO_ROOT = Path(__file__).resolve().parent
_FIREFLY_YAML = _REPO_ROOT / "omni_drones/robots/assets/usd/firefly.yaml"


@dataclass
class CheckItem:
    name: str
    value: str
    status: str  # PASS | WARN | FAIL
    note: str


def _build_test_cfg(task_name: str, headless: bool, num_envs: int):
    os.environ.setdefault("OMNI_DRONES_HYDRA_SEARCHPATH", str(_REPO_ROOT / "cfg"))

    task_cfg = OmegaConf.load(str(_REPO_ROOT / f"cfg/task/{task_name}.yaml"))
    env_base = OmegaConf.load(str(_REPO_ROOT / "cfg/base/env_base.yaml"))
    sim_base = OmegaConf.load(str(_REPO_ROOT / "cfg/base/sim_base.yaml"))

    task_env_cfg = OmegaConf.select(task_cfg, "env", default={})
    task_sim_cfg = OmegaConf.select(task_cfg, "sim", default={})
    task_cfg.env = OmegaConf.merge(env_base.env, task_env_cfg)
    task_cfg.sim = OmegaConf.merge(sim_base.sim, task_sim_cfg)
    task_cfg.env.num_envs = num_envs

    cfg = OmegaConf.create(
        {
            "headless": headless,
            "task": task_cfg,
            "env": task_cfg.env,
            "sim": task_cfg.sim,
            "viewer": {
                "resolution": [960, 720],
                "eye": [8, 0.0, 6.0],
                "lookat": [0.0, 0.0, 1.0],
            },
        }
    )
    return cfg


def _firefly_reference() -> Dict[str, float]:
    ff = OmegaConf.to_container(OmegaConf.load(str(_FIREFLY_YAML)), resolve=True)
    n = ff["rotor_configuration"]["num_rotors"]
    kf = ff["rotor_configuration"]["force_constants"][0]
    w = ff["rotor_configuration"]["max_rotation_velocities"][0]
    mass = ff["mass"]
    t2w = n * kf * (w**2) / (mass * 9.81)
    hover_t = 1.0 / t2w
    return {
        "t2w": t2w,
        "hover_t": hover_t,
    }


def _alternate_direction_score(directions: torch.Tensor) -> float:
    if directions.numel() < 2:
        return 0.0
    sign = torch.sign(directions)
    alternations = (sign[:-1] * sign[1:] < 0).float()
    return alternations.mean().item()


def _check_runtime_health(env, drift_m: float, drift_slope: float) -> Tuple[List[CheckItem], Dict[str, float]]:
    drone = env.drone
    mass = float(drone.masses[0].item())
    gravity_force = float(drone.gravity[0].item())
    kf_values = drone.KF[0].reshape(-1).float()
    km_values = drone.KM[0].reshape(-1).float()
    max_omega = drone.MAX_ROT_VEL.reshape(-1).float()
    n_rotors = int(drone.num_rotors)

    total_max_thrust = float(kf_values.sum().item())
    t2w = total_max_thrust / gravity_force if gravity_force > 1e-6 else 0.0
    hover_t = gravity_force / total_max_thrust if total_max_thrust > 1e-6 else math.inf
    hover_cmd = 2.0 * hover_t - 1.0 if math.isfinite(hover_t) else math.inf
    force2moment = float((kf_values[0] / km_values[0]).item()) if km_values[0].abs().item() > 1e-12 else math.inf

    inertia_diag = drone.inertias[0, 0].float()
    inertia_norm = inertia_diag / max(mass, 1e-6)
    inertia_xy_ratio = float((inertia_diag[0] / max(inertia_diag[1], 1e-6)).item())

    arm_lengths = torch.as_tensor(drone.params["rotor_configuration"]["arm_lengths"]).float()
    directions = torch.as_tensor(drone.params["rotor_configuration"]["directions"]).float()
    alternation_score = _alternate_direction_score(directions)
    arm_spread = float((arm_lengths.max() - arm_lengths.min()).item())
    rotor_view_count = int(drone.rotors_view.count)

    ff_ref = _firefly_reference()
    items: List[CheckItem] = []

    def add(name: str, value: str, status: str, note: str):
        items.append(CheckItem(name=name, value=value, status=status, note=note))

    if t2w < 1.2:
        add("thrust_to_weight", f"{t2w:.3f}", "FAIL", "推重比过低，悬停与学习都很难稳定。")
    elif t2w < 1.8:
        add("thrust_to_weight", f"{t2w:.3f}", "WARN", "可飞但控制裕度偏低，收敛会慢。")
    else:
        add("thrust_to_weight", f"{t2w:.3f}", "PASS", "推重比处于可训练区间。")

    if not math.isfinite(hover_t) or hover_t <= 0 or hover_t >= 1:
        add("hover_throttle_ratio_t", str(hover_t), "FAIL", "悬停推力比例异常，通常是推力参数单位错误。")
    elif hover_t > 0.75:
        add("hover_throttle_ratio_t", f"{hover_t:.3f}", "WARN", "悬停时接近饱和，建议增加推力裕度。")
    else:
        add("hover_throttle_ratio_t", f"{hover_t:.3f}", "PASS", "悬停推力比例合理。")

    if alternation_score < 0.95:
        add("rotor_direction_alternation", f"{alternation_score:.2f}", "FAIL", "旋翼正反转序列非交替，偏航力矩可能错误。")
    else:
        add("rotor_direction_alternation", f"{alternation_score:.2f}", "PASS", "旋翼方向交替正常。")

    expected_rotor_prim_count = env.num_envs * n_rotors
    if rotor_view_count != expected_rotor_prim_count:
        add(
            "rotor_prim_binding",
            f"{rotor_view_count}/{expected_rotor_prim_count}",
            "FAIL",
            "rotors_view匹配数量异常，可能漏绑旋翼prim。",
        )
    else:
        add("rotor_prim_binding", f"{rotor_view_count}/{expected_rotor_prim_count}", "PASS", "rotor prim绑定数量正确。")

    if arm_spread > 0.08:
        add("arm_length_spread_m", f"{arm_spread:.3f}", "WARN", "臂长不对称较大，姿态轴耦合会增强。")
    else:
        add("arm_length_spread_m", f"{arm_spread:.3f}", "PASS", "臂长离散度可接受。")

    if inertia_norm.min().item() <= 0:
        add("inertia_positive", str(inertia_diag.tolist()), "FAIL", "转动惯量需要为正。")
    elif not (0.005 <= inertia_norm.mean().item() <= 0.08):
        add(
            "inertia_mass_scale",
            f"{inertia_norm.mean().item():.4f}",
            "WARN",
            "惯量-质量比例与常见飞行器偏差较大，请复核CAD/URDF惯量。",
        )
    else:
        add("inertia_mass_scale", f"{inertia_norm.mean().item():.4f}", "PASS", "惯量-质量比例在经验范围内。")

    if drift_m < -0.3:
        add("hover_drift_5to10s", f"{drift_m:+.3f} m", "FAIL", "固定悬停指令仍明显掉高，推力或动力映射有问题。")
    elif abs(drift_m) > 0.25:
        add("hover_drift_5to10s", f"{drift_m:+.3f} m", "WARN", "高度漂移偏大，建议先修模型/参数再训RL。")
    else:
        add("hover_drift_5to10s", f"{drift_m:+.3f} m", "PASS", "固定悬停指令下高度漂移可接受。")

    # contextual item against Firefly as reference, not hard requirement.
    if t2w < 0.75 * ff_ref["t2w"]:
        add("vs_firefly_t2w", f"{t2w/ff_ref['t2w']:.2f}x", "WARN", "相对Firefly控制裕度偏低，训练可能更慢。")
    else:
        add("vs_firefly_t2w", f"{t2w/ff_ref['t2w']:.2f}x", "PASS", "相对Firefly裕度可接受。")

    if hasattr(drone, "_amov_articulation_rel") and not drone._amov_articulation_rel:
        add("articulation_root_binding", "empty", "FAIL", "未缓存articulation root，关节链初始化可能异常。")
    else:
        add("articulation_root_binding", str(getattr(drone, "_amov_articulation_rel", "n/a")), "PASS", "articulation root已解析。")

    metrics = {
        "mass_kg": mass,
        "gravity_force_n": gravity_force,
        "num_rotors": float(n_rotors),
        "max_rot_vel_rad_s": float(max_omega[0].item()),
        "thrust_to_weight": t2w,
        "hover_t": hover_t,
        "hover_cmd": hover_cmd,
        "force_to_moment_ratio": force2moment,
        "inertia_xx": float(inertia_diag[0].item()),
        "inertia_yy": float(inertia_diag[1].item()),
        "inertia_zz": float(inertia_diag[2].item()),
        "inertia_xy_ratio": inertia_xy_ratio,
        "drift_m": drift_m,
        "drift_slope_m_per_s": drift_slope,
        "firefly_ref_t2w": ff_ref["t2w"],
        "firefly_ref_hover_t": ff_ref["hover_t"],
    }
    return items, metrics


def _run_hover_probe(env, steps: int, print_interval: int = 50) -> Tuple[float, float]:
    gravity_force = float(env.drone.gravity[0].item())
    total_max_thrust = float(env.drone.KF[0].reshape(-1).sum().item())
    hover_t = gravity_force / max(total_max_thrust, 1e-9)
    hover_cmd = 2.0 * hover_t - 1.0
    action = torch.ones(env.num_envs, 1, env.drone.num_rotors, device=env.device) * hover_cmd
    td = TensorDict(
        {"agents": {"action": action}},
        batch_size=[env.num_envs],
        device=env.device,
    )

    env.reset()
    zs = []
    for i in range(steps):
        env.step(td)
        z = float(env.drone.pos[..., 2].mean().item())
        zs.append(z)
        if i % print_interval == 0 or i == steps - 1:
            print(f"  Step {i:4d}: z_mean = {z:.3f} m")

    dt = env.dt * env.substeps
    drift_m = zs[-1] - zs[0]
    drift_slope = drift_m / max((steps - 1) * dt, 1e-9)
    return drift_m, drift_slope


def _render_report(items: List[CheckItem], metrics: Dict[str, float]) -> int:
    print("\n" + "=" * 84)
    print("AMOV HOVER READINESS REPORT (Firefly-referenced)")
    print("=" * 84)
    print("Core metrics:")
    for key in (
        "mass_kg",
        "gravity_force_n",
        "num_rotors",
        "max_rot_vel_rad_s",
        "thrust_to_weight",
        "hover_t",
        "hover_cmd",
        "force_to_moment_ratio",
        "inertia_xx",
        "inertia_yy",
        "inertia_zz",
        "drift_m",
        "drift_slope_m_per_s",
    ):
        print(f"  - {key:24s}: {metrics[key]:.6g}")

    print("\nChecks:")
    failed = 0
    warned = 0
    for item in items:
        if item.status == "FAIL":
            failed += 1
        elif item.status == "WARN":
            warned += 1
        print(f"  [{item.status:4s}] {item.name:28s} = {item.value:>14s} | {item.note}")

    print("\nSummary:")
    print(f"  FAIL: {failed}, WARN: {warned}, PASS: {len(items) - failed - warned}")
    if failed:
        print("  Verdict: NOT_READY_FOR_RL")
        return 2
    if warned:
        print("  Verdict: READY_WITH_WARNINGS")
        return 1
    print("  Verdict: READY_FOR_RL")
    return 0


def run_audit(task_name: str, headless: bool, num_envs: int, steps: int) -> int:
    cfg = _build_test_cfg(task_name=task_name, headless=headless, num_envs=num_envs)
    cfg.task.drone_model.name = "AmovDrone"

    print("=" * 84)
    print("Running AMOV hover physics audit")
    print(f"Task={task_name}, headless={headless}, num_envs={num_envs}, steps={steps}")
    print("=" * 84)

    simulation_app = init_simulation_app(cfg)
    _exit_code = 3
    try:
        from omni_drones.envs.isaac_env import IsaacEnv

        try:
            env = IsaacEnv.REGISTRY[cfg.task.name](cfg, headless=headless)
            drift_m, drift_slope = _run_hover_probe(env, steps=steps)
            items, metrics = _check_runtime_health(env, drift_m=drift_m, drift_slope=drift_slope)
            _exit_code = _render_report(items, metrics)
        except Exception:
            import traceback
            traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        simulation_app.close()
    return _exit_code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="HoverAmov", help="Task config name under cfg/task.")
    parser.add_argument("--steps", type=int, default=500, help="Hover probe simulation steps.")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel envs for probe.")
    parser.add_argument("--headless", action="store_true", help="Run without viewer.")
    args = parser.parse_args()
    raise SystemExit(run_audit(args.task, args.headless, args.num_envs, args.steps))


if __name__ == "__main__":
    main()
