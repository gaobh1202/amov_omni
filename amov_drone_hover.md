# AmovDrone 悬停实现总结

本文档记录将 `AmovDrone` 接入 OmniDrones `HoverAmov` 任务并成功训练悬停策略的关键修改，以及当前 `amov_drone.usd` 模型的结构特点。

---

## 运行方式

```bash
OMNIDRONES_AMOV_USD_PATH=/home/benhua/OmniDrones/omni_drones/robots/assets/usd/amov_drone.usd \
python scripts/train.py task=HoverAmov algo=ppo headless=false algo.actor_log_std_init=-1.0
```

- USD 路径可通过环境变量 `OMNIDRONES_AMOV_USD_PATH` 指定。
- 参数 YAML 自动匹配同名文件：`omni_drones/robots/assets/usd/amov_drone.yaml`（也可通过 `OMNIDRONES_AMOV_PARAM_PATH` 显式覆盖）。

---

## 一、成功实现悬停的关键修改

按重要性排序，大致分四层：**物理施力 → USD 模型 → 奖励/动作映射 → 训练初始化**。

### 1. 物理施力（最关键）

#### 问题 A：子连杆施力无效

Firefly 等模型可以对各 `rotor_link` 单独施力。AmovDrone 是 articulation 结构，PhysX 会**静默忽略**对关节链子连杆（`rotor_link_*`）通过 `RigidPrimView` 施加的力。

**修复**（`omni_drones/robots/drone/amov_drone.py` → `apply_action`）：

- 各旋翼推力仍按 `rotor_pos` 和世界坐标 thrust 方向计算。
- 汇总为 net force + pitch/roll torque。
- **统一施加到 `base_link`（ArticulationView 根连杆）**，而非各 rotor 子连杆。

#### 问题 B：力矩参考系错误

曾用 `get_coms()` 作力臂参考点。它返回的是**局部坐标偏移**（约 4 cm），而 `rotor_pos` 是**世界坐标**（128 个 env、间距 8 m 时可差几十米），产生数万 N·m 的虚假 pitch/roll 力矩，导致立刻侧翻。

**修复**：力臂参考点改为 `base_link.get_world_poses()`（世界坐标 prim 原点，与 COM 误差约 4 cm，可忽略）。

---

### 2. USD 模型参数（`omni_drones/robots/assets/usd/amov_drone.usd`）

| 修改项 | 原问题 | 现状态 |
|--------|--------|--------|
| 旋翼关节阻尼 | `damping = 10,000,000`，`set_joint_velocities` 产生巨大制动力矩，机体疯狂自旋 | `damping = 0` |
| 关节限位 | `±57.3°`，无法连续旋转，螺旋桨视觉上不转 | `±inf`（与 Firefly 一致） |
| 关节摩擦 | 有额外阻力 | `jointFriction = 0` |
| 旋翼质量 | 约 120 g，惯量过大 | 约 5 g |
| ArticulationRoot | 曾移到 Xform 根 → 旋翼变独立 `PxRigidDynamic`，触发 GPU API 冲突并崩溃 | **保持在 `base_link`** |

**视觉旋转**：重新启用 `set_joint_velocities` 驱动螺旋桨转动；物理推力仍通过 `base_link` 汇总施力。为避免高转速频闪，视觉角速度上限约 **15 rad/s**（`_VISUAL_MAX_RAD_S`），与真实 rotor 动力学解耦。

---

### 3. 控制参数（`omni_drones/robots/assets/usd/amov_drone.yaml`）

- **推重比**：`force_constants` 与 `max_rotation_velocities` 提高约 1.5 倍，目标 T/W ≈ **3.0**（原约 2.0）。
- **旋翼方向**：`directions: [-1, 1, -1, 1]`。
- **质量/惯量**：与 PhysX 实际值对齐（mass ≈ 10.33 kg）。
- **总质量计算**：`AmovDrone.initialize` 用所有 body 质量之和计算 hover throttle 和重力，而非仅 root link 质量。

---

### 4. 奖励与动作映射（`cfg/task/HoverAmov.yaml` + `omni_drones/envs/single/hover.py`）

| 配置/代码 | 作用 |
|-----------|------|
| `reward_effort_weight: 0.0` | 避免策略学「零推力省 effort」而自由落体 |
| `reward_alive` 高度比例 | 离地越高 reward 越大，提供悬停梯度 |
| `reward_up_weight: 0.5` | 独立姿态奖励，episode 初期就有「保持直立」信号 |
| `_hover_cmd_offset` / `_hover_cmd_scale` | action=0 对应 hover 推力，±1 对称探索 |
| `ground_reset_height: 0.45` | 触地即 reset，减少在地面「躺平」 |
| `min_up_z: -0.2` | 放宽侧翻终止阈值，给策略更多学习步数 |
| 缩小 `init_pos_range` | 初始离 target 更近，pose reward 更有效 |

**动作归一化逻辑**（`hover.py`）：

```python
_hover_frac = gravity / KF_sum          # 悬停所需 throttle 比例
_hover_cmd_offset = 2 * _hover_frac - 1 # action=0 → hover
_hover_cmd_scale = 1 - |offset|        # 对称映射到 [-1, 1]
actions = actions * scale + offset
```

---

### 5. PPO 训练初始化

| 项 | 作用 |
|----|------|
| `algo.actor_log_std_init: -1.0` | 初始探索 std ≈ 0.37，避免大随机力矩扰动 |
| Actor 最后一层 zero-init | 初始 policy 输出 ≈ hover 推力，而非随机大推力 |

---

### 6. 排查过程中已确认但非最终方案的内容

- **曾尝试**将 `ArticulationRootAPI` 移到 Xform 根以完全对齐 Firefly 结构 → 导致 `PxRigidDynamic` GPU API 错误，已回退。
- **曾怀疑** USD 中 `diagonalInertia = (0,0,0)` → 运行时 PhysX 会推导正确惯量，不是主要问题。
- **曾用** `amov_drone_arm_0.usd`（带机械臂）排查 → 无机械臂版 `amov_drone.usd` 可隔离机械臂惯量影响，当前悬停训练基于纯四旋翼模型。

---

## 二、当前 `amov_drone.usd` 结构特点

```
/amov_drone                    [Xform，spawn 根，无 ArticulationRoot]
└── base_link                  [RigidBody + MassAPI + ArticulationRootAPI]  ← 关节链根
    ├── rotor_joint_1 … 4      [RevoluteJoint]
    │   ├── damping = 0
    │   ├── lower/upperLimit = ±inf
    │   └── jointFriction = 0
    └── (通过 joint 连接)
        rotor_link_1 … 4       [RigidBody，质量约 5 g]
```

### 与 Firefly 的对比

| 特点 | AmovDrone | Firefly |
|------|-----------|---------|
| ArticulationRoot 位置 | `base_link` | Xform 根 |
| 旋翼是否为独立刚体 | 否（关节链子体） | 是 |
| 施力方式 | 汇总后施加到 `base_link` | 各 rotor 单独施力 |
| 视觉螺旋桨旋转 | `set_joint_velocities`（限速防频闪） | 同样机制 |
| 机械臂 | 无（纯四旋翼版） | 无 |

### 代码侧对应关系

- 机器人类：`omni_drones/robots/drone/amov_drone.py`（`AmovDrone`）
- 参数文件：`omni_drones/robots/assets/usd/amov_drone.yaml`
- 任务配置：`cfg/task/HoverAmov.yaml`
- 环境逻辑：`omni_drones/envs/single/hover.py`

---

## 三、训练与 Checkpoint

- 主训练配置：`scripts/train.yaml`（Hydra 实际读取此文件，非 `cfg/train.yaml`）。
- `save_interval: 200` 表示每 200 个 iteration 覆盖保存 `checkpoint_latest.pt`。
- Checkpoint 路径：当前 wandb run 目录下的 `files/checkpoint_latest.pt`。
- 训练结束另存：`checkpoint_final.pt`。

**Frames 含义**：`total_frames` 为所有并行 env 的 step 总数（非渲染帧）。对 HoverAmov（128 env × PPO `train_every=32`），每 iteration 收集 4096 frames。

---

## 四、一句话总结

悬停成功的核心是 **「汇总施力到 base_link + 世界坐标力矩参考 + 去掉旋翼关节超大阻尼」**；奖励和 PPO 初始化解决的是 **「策略愿不愿意、能不能从 hover 附近学起来」**。两者缺一不可——物理对了但 reward 不对会自由落体，reward 对了但施力错了会侧翻。
