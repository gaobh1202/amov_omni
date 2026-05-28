'''
python scripts/train.py algo=ppo headless=false \
  wandb.entity=benhua_g-the-hong-kong-polytechnic-university \
  wandb.project=omnidrone \
  task=HoverAmov
'''
import logging
import os
import time
from pathlib import Path

# Hydra 中 file://../cfg 会随当前工作目录变化；从仓库根目录执行 python scripts/train.py 时会失效。
_REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault(
    "OMNI_DRONES_HYDRA_SEARCHPATH",
    str(_REPO_ROOT / "cfg"),
)

import hydra
import torch
import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt

from torch.func import vmap
from tqdm import tqdm
from omegaconf import OmegaConf

from omni_drones import init_simulation_app
from torchrl.data import CompositeSpec
from torchrl.envs.utils import set_exploration_type, ExplorationType
from omni_drones.utils.torchrl import SyncDataCollector
from omni_drones.utils.torchrl.transforms import (
    FromMultiDiscreteAction,
    FromDiscreteAction,
    ravel_composite,
    AttitudeController,
    RateController,
)
from omni_drones.utils.wandb import init_wandb
from omni_drones.utils.torchrl import RenderCallback, EpisodeStats
from omni_drones.learning import ALGOS

from setproctitle import setproctitle
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose


def _run_preflight(base_env, cfg) -> None:
    preflight_cfg = cfg.task.get("preflight", None)
    if not preflight_cfg or not preflight_cfg.get("enabled", False):
        return

    drone = base_env.drone
    gravity = drone.gravity.reshape(-1).float()
    kf = drone.KF.reshape(-1, drone.num_rotors).float()[0]
    total_max_thrust = float(kf.sum().item())
    gravity_force = float(gravity[0].item())
    t2w = total_max_thrust / max(gravity_force, 1e-9)
    hover_t = gravity_force / max(total_max_thrust, 1e-9)
    rotor_view_count = int(getattr(drone.rotors_view, "count", 0))
    expected_rotor_count = int(base_env.num_envs * drone.num_rotors)
    articulation_rel = getattr(drone, "_amov_articulation_rel", None)

    checks = []
    checks.append(("thrust_to_weight", t2w, "FAIL" if t2w < 1.2 else ("WARN" if t2w < 1.8 else "PASS")))
    checks.append(("hover_ratio_t", hover_t, "FAIL" if not (0.0 < hover_t < 1.0) else ("WARN" if hover_t > 0.75 else "PASS")))
    checks.append((
        "rotor_prim_binding",
        f"{rotor_view_count}/{expected_rotor_count}",
        "FAIL" if rotor_view_count != expected_rotor_count else "PASS",
    ))
    checks.append((
        "articulation_binding",
        articulation_rel if articulation_rel else "missing",
        "FAIL" if articulation_rel in (None, "") else "PASS",
    ))

    logging.info("Preflight checks for %s", cfg.task.drone_model.name)
    fail_count = 0
    warn_count = 0
    for name, value, status in checks:
        logging.info("Preflight %-22s [%s] value=%s", name, status, value)
        if status == "FAIL":
            fail_count += 1
        elif status == "WARN":
            warn_count += 1

    if fail_count:
        raise RuntimeError(
            "Preflight failed with model-level issues. "
            "Run `python test_amov_thrust.py --task HoverAmov` for full diagnostics."
        )
    if warn_count:
        logging.warning("Preflight completed with %d warning(s).", warn_count)
    else:
        logging.info("Preflight passed without warnings.")


_THRUST_PRINT_COUNTER = 0
_THRUST_PRINT_INTERVAL = 50  # print every N policy updates

def _collect_thrust_metrics(base_env):
    global _THRUST_PRINT_COUNTER
    drone = getattr(base_env, "drone", None)
    if drone is None or not all(hasattr(drone, k) for k in ("thrusts", "KF", "gravity")):
        return {}

    # thrusts shape: (..., num_rotors, 3), z-axis stores per-rotor thrust scalar.
    thrust_z = drone.thrusts[..., 2]
    mean_total_thrust = thrust_z.sum(dim=-1).mean()
    max_total_thrust = drone.KF.sum(dim=-1).mean().clamp_min(1e-9)
    hover_force = drone.gravity.mean().clamp_min(1e-9)

    # Per-rotor throttle (0-1) and corresponding angular velocity (rad/s)
    mean_throttle = drone.throttle.mean().item()
    per_rotor_throttle = drone.throttle.reshape(-1, drone.num_rotors).mean(0)  # (num_rotors,)
    # angular velocity: throttle * MAX_ROT_VEL (rad/s)
    if hasattr(drone, "rotors") and hasattr(drone.rotors, "KF"):
        # MAX_ROT_VEL stored implicitly; derive from KF = max_omega^2 * force_constant
        # use drone.KF_0 and rotor config if available
        max_omega = getattr(drone, "_max_rot_vel", None)
        if max_omega is None and hasattr(drone, "KF_0") and hasattr(drone, "rotors"):
            # KF = omega_max^2 * kf_coef → omega_max = sqrt(KF / kf_coef) but we can't easily get kf_coef
            # fallback: just report throttle
            max_omega = None
    else:
        max_omega = None

    metrics = {
        "train/mean_thrust_n": mean_total_thrust.item(),
        "train/max_thrust_ratio": (mean_total_thrust / max_total_thrust).item(),
        "train/hover_ratio": (mean_total_thrust / hover_force).item(),
        "train/mean_throttle": mean_throttle,
    }

    # hover_cmd normalization params from env if available
    hover_cmd_offset = getattr(base_env, "_hover_cmd_offset", None)
    if hover_cmd_offset is not None:
        metrics["train/hover_cmd_offset"] = hover_cmd_offset
    hover_cmd_scale = getattr(base_env, "_hover_cmd_scale", None)
    if hover_cmd_scale is not None:
        metrics["train/hover_cmd_scale"] = hover_cmd_scale

    # Periodic console print for quick inspection
    _THRUST_PRINT_COUNTER += 1
    if _THRUST_PRINT_COUNTER % _THRUST_PRINT_INTERVAL == 0:
        thr_per_rotor_str = ", ".join(f"{v:.3f}" for v in per_rotor_throttle.tolist())
        thrust_per_rotor = (per_rotor_throttle ** 2) * drone.KF.reshape(-1, drone.num_rotors).mean(0)
        thrust_per_rotor_str = ", ".join(f"{v:.1f}N" for v in thrust_per_rotor.tolist())
        logging.info(
            "[Thrust] total=%.1fN  hover=%.1fN  ratio=%.2f  "
            "throttle/rotor=[%s]  thrust/rotor=[%s]",
            mean_total_thrust.item(),
            hover_force.item(),
            (mean_total_thrust / hover_force).item(),
            thr_per_rotor_str,
            thrust_per_rotor_str,
        )

    return metrics


@hydra.main(version_base=None, config_path=".", config_name="train")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)
    run = init_wandb(cfg)
    setproctitle(run.name)
    print(OmegaConf.to_yaml(cfg))

    from omni_drones.envs.isaac_env import IsaacEnv

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)
    _run_preflight(base_env, cfg)

    transforms = [InitTracker()]

    # a CompositeSpec is by default processed by a entity-based encoder
    # ravel it to use a MLP encoder instead
    if cfg.task.get("ravel_obs", False):
        transform = ravel_composite(base_env.observation_spec, ("agents", "observation"))
        transforms.append(transform)
    if cfg.task.get("ravel_obs_central", False):
        transform = ravel_composite(base_env.observation_spec, ("agents", "observation_central"))
        transforms.append(transform)

    # optionally discretize the action space or use a controller
    action_transform: str = cfg.task.get("action_transform", None)
    if action_transform is not None:
        if action_transform.startswith("multidiscrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromMultiDiscreteAction(nbins=nbins)
            transforms.append(transform)
        elif action_transform.startswith("discrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromDiscreteAction(nbins=nbins)
            transforms.append(transform)
        else:
            raise NotImplementedError(f"Unknown action transform: {action_transform}")

    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    try:
        policy = ALGOS[cfg.algo.name.lower()](
            cfg.algo,
            env.observation_spec,
            env.action_spec,
            env.reward_spec,
            device=base_env.device
        )
    except KeyError:
        raise NotImplementedError(f"Unknown algorithm: {cfg.algo.name}")

    frames_per_batch = env.num_envs * int(cfg.algo.train_every)
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    max_iters = cfg.get("max_iters", -1)
    eval_interval = cfg.get("eval_interval", -1)
    save_interval = cfg.get("save_interval", -1)

    stats_keys = [
        k for k in base_env.observation_spec.keys(True, True)
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(stats_keys)
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=cfg.sim.device,
        return_same_td=True,
    )

    @torch.no_grad()
    def evaluate(
        seed: int=0,
        exploration_type: ExplorationType=ExplorationType.MODE
    ):

        base_env.enable_render(True)
        base_env.eval()
        env.eval()
        env.set_seed(seed)

        render_callback = RenderCallback(interval=2)

        with set_exploration_type(exploration_type):
            trajs = env.rollout(
                max_steps=base_env.max_episode_length,
                policy=policy,
                callback=render_callback,
                auto_reset=True,
                break_when_any_done=False,
                return_contiguous=False,
            )
        base_env.enable_render(not cfg.headless)
        env.reset()

        done = trajs.get(("next", "done"))
        first_done = torch.argmax(done.long(), dim=1).cpu()

        def take_first_episode(tensor: torch.Tensor):
            indices = first_done.reshape(first_done.shape+(1,)*(tensor.ndim-2))
            return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

        traj_stats = {
            k: take_first_episode(v)
            for k, v in trajs[("next", "stats")].cpu().items()
        }

        info = {
            "eval/stats." + k: torch.mean(v.float()).item()
            for k, v in traj_stats.items()
        }

        # log video
        info["recording"] = wandb.Video(
            render_callback.get_video_array(axes="t c h w"),
            fps=0.5 / (cfg.sim.dt * cfg.sim.substeps),
            format="mp4"
        )

        # log distributions
        # df = pd.DataFrame(traj_stats)
        # table = wandb.Table(dataframe=df)
        # info["eval/return"] = wandb.plot.histogram(table, "return")
        # info["eval/episode_len"] = wandb.plot.histogram(table, "episode_len")

        return info

    pbar = tqdm(collector, total=total_frames//frames_per_batch)
    env.train()
    for i, data in enumerate(pbar):
        info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
        episode_stats.add(data.to_tensordict())
        info.update(_collect_thrust_metrics(base_env))

        if len(episode_stats) >= base_env.num_envs:
            stats = {
                "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item()
                for k, v in episode_stats.pop().items(True, True)
            }
            info.update(stats)

        info.update(policy.train_op(data.to_tensordict()))

        if eval_interval > 0 and i % eval_interval == 0:
            logging.info(f"Eval at {collector._frames} steps.")
            info.update(evaluate())
            env.train()
            base_env.train()

        if save_interval > 0 and i % save_interval == 0:
            try:
                ckpt_path = os.path.join(run.dir, "checkpoint_latest.pt")
                torch.save(policy.state_dict(), ckpt_path)
                logging.info(f"Saved checkpoint to {str(ckpt_path)} (frames={collector._frames})")
            except AttributeError:
                logging.warning(f"Policy {policy} does not implement `.state_dict()`")

        run.log(info)
        print(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, float)}))

        pbar.set_postfix({"rollout_fps": collector._fps, "frames": collector._frames})

        if max_iters > 0 and i >= max_iters - 1:
            break

    logging.info(f"Final Eval at {collector._frames} steps.")
    info = {"env_frames": collector._frames}
    info.update(evaluate())
    run.log(info)

    try:
        ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
        torch.save(policy.state_dict(), ckpt_path)

        model_artifact = wandb.Artifact(
            f"{cfg.task.name}-{cfg.algo.name.lower()}",
            type="model",
            description=f"{cfg.task.name}-{cfg.algo.name.lower()}",
            metadata=dict(cfg))

        model_artifact.add_file(ckpt_path)
        wandb.save(ckpt_path)
        run.log_artifact(model_artifact)

        logging.info(f"Saved checkpoint to {str(ckpt_path)}")
    except AttributeError:
        logging.warning(f"Policy {policy} does not implement `.state_dict()`")

    wandb.finish()

    simulation_app.close()


if __name__ == "__main__":
    main()
