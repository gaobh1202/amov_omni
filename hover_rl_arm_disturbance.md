# 无人机 Hover RL 策略输入、机械臂动态运动与扰动 force/wrench 获取

> 适用对象：OmniDrones 项目中的 `Hover` 环境 + `AmovDrone`（带 6-DOF airbot_play 机械臂）
> 代码路径：
> - 环境：`omni_drones/envs/single/hover.py`
> - 机器人：`omni_drones/robots/drone/amov_drone.py`、`omni_drones/robots/drone/multirotor.py`
> - 任务配置：`cfg/task/HoverAmovDynaArm.yaml`（动态臂）、`HoverAmovArm.yaml`（静止臂）
> - 评测/扰动测量：`scripts/eval_hover_robustness.py`

---

## 1. RL 策略的输入（Observation）

策略观测在 `Hover._compute_state_and_obs`（`hover.py:553`）中拼装，最终维度 **30**。

### 1.1 观测组成

```python
obs = [self.rpos, safe_drone_state[..., 3:], self.rheading]
if self.time_encoding:
    obs.append(time_encoding)          # 4 维
obs = torch.cat(obs, dim=-1)           # → (num_envs, 1, 30)
```

| 分量 | 维度 | 含义 | 来源 |
|------|------|------|------|
| `rpos` | 3 | 相对目标的位置误差 `target_pos - pos` | `hover.py:560` |
| `rot` | 4 | 机体姿态四元数 | `drone_state[3:7]` |
| `vel` | 6 | 线速度 + 角速度（6 维） | `drone_state[7:13]` |
| `heading` | 3 | 机头朝向向量（机体 x 轴在世界系） | `drone_state[13:16]` |
| `up` | 3 | 机体上向量（机体 z 轴在世界系） | `drone_state[16:19]` |
| `throttle` | 4 | 各旋翼当前油门，归一化到 `[-1,1]`（`throttle*2-1`） | `drone_state[19:23]` |
| `rheading` | 3 | 目标朝向与当前朝向之差 `target_heading - heading` | `hover.py:561` |
| `time_encoding` | 4 | 回合进度编码 `progress_buf / max_episode_length` 广播 4 维 | `hover.py:564` |

> 合计：3 + (4+6+3+3+4) + 3 + 4 = **30**。
> 注意 `safe_drone_state[..., 3:]` 丢弃了状态向量前 3 维（绝对位置 `pos`），改用 `rpos`（相对目标位置），从而让策略对目标位置平移不变。

### 1.2 底层 `drone_state` 的定义

`drone_state = self.drone.get_state()`（`MultirotorBase.get_state`, `multirotor.py:285`）：

```python
state = [self.pos, self.rot, self.vel, self.heading, self.up, self.throttle * 2 - 1]
state = torch.cat(state, dim=-1)   # 维度 = 3+4+6+3+3+num_rotors(4) = 23
```

- `pos`：世界系位置（`env_frame=True` 时已减去 env 原点偏移）。
- `rot`：姿态四元数（`get_world_poses`）。
- `vel`：6 维速度（线速度 3 + 角速度 3）。对 AmovDrone，`initialize` 中 `self.vel = self.vel_w`（别名同一张量，`amov_drone.py:246`），`get_state` 中 `self.vel_w[:] = vel_w`，因此该速度为**世界系**速度。
- `heading = quat_axis(rot, axis=0)`，`up = quat_axis(rot, axis=2)`。
- `throttle*2-1`：各旋翼油门，从一阶电机模型 `RotorGroup` 维护，归一化到 `[-1,1]`。

### 1.3 关键观察：观测里**不含机械臂状态**

策略观测**不包含任何机械臂关节角度/角速度**，也不包含臂端的 force/wrench。机械臂运动对策略而言是一种**外部扰动**：策略只能通过 `pos/vel/rot/角速度` 等本体状态的变化间接感知臂运动带来的重心偏移和反作用力矩，从而学习抗扰动悬停。这是该任务“抗机械臂扰动鲁棒 hover”设定的核心。

> 旁注：`info.drone_state`（`hover.py:577`）是给 `RateController`/`AttitudeController` 动作变换用的完整 23 维状态，**不进入策略输入**（ravel 只取 `("agents","observation")`）。详见 action-transform 流水线。

### 1.4 动作空间（对照）

- 默认：`num_rotors` 维（4），直接旋翼指令 `[-1,1]`，在 `_pre_sim_step` 做 hover 归一化（`action=0 → 悬停推力`）。
- `HoverAmovArmThrustRate.yaml`：4 维 `[roll_rate, pitch_rate, yaw_rate, thrust]`，经 `RateController` 变换，对应 PX4 rate 模式。

---

## 2. 机械臂动态运动的定义

`Hover` 支持三种机械臂控制模式，优先级 **arm_hold > arm_trajectory > arm_motion**（`hover.py:188-246` 解析，`_pre_sim_step` 应用）。动态运动主要由 **`arm_trajectory`** 模式实现。

### 2.1 多关节正弦轨迹（`arm_trajectory`，动态臂主模式）

每个机械臂关节独立按正弦轨迹运动：

```
posⱼ(t) = Aⱼ · sin(2π · fⱼ · t + φⱼ)
```

实现于 `Hover._apply_arm_trajectory`（`hover.py:537`）：

```python
t = self.progress_buf.float() * (self.dt * self.substeps)   # 每个 env 的仿真时间(s)
pos = amplitudes[None,:] * torch.sin(2π * frequencies[None,:] * t[:,None] + phases[None,:])
self.drone._view.set_joint_position_targets(pos, joint_indices=self.drone.arm_joint_indices)
```

- **时间基准** `t`：`progress_buf`（控制步计数）× `dt * substeps`，即真实秒数。
- **执行方式**：写入关节的**位置目标** `set_joint_position_targets`，由 PhysX 关节 PD 驱动器跟踪，而非直接施加力矩。
- **PD 增益**：`stiffness=500`、`damping=50`（`hover.py:232-242` 中 `set_gains`）。

每关节的 `amplitude / frequency / phase` 在配置中给出。`HoverAmovDynaArm.yaml` 的设定（airbot_play 6-DOF + 夹爪，共 8 DOF）：

| 关节 | amplitude (rad) | frequency (Hz) | phase (rad) | 说明 |
|------|-----------------|----------------|-------------|------|
| joint1 | 0.6 | 0.25 | 0.0 | 底座旋转 |
| joint2 | 0.4 | 0.30 | 1.047 | 肩部俯仰 |
| joint3 | 0.35 | 0.35 | 2.094 | 肘部俯仰 |
| joint4 | 0.25 | 0.40 | 0.0 | 前臂滚转 |
| joint5 | 0.25 | 0.45 | 1.047 | 腕部俯仰 |
| joint6 | 0.15 | 0.50 | 2.094 | 腕部滚转 |
| endleft/endright | 0.0 | 0.0 | 0.0 | 夹爪保持零位 |

不同的 amplitude/frequency/phase 组合使臂端轨迹形成不规则的复合运动，从而对机身产生时变的力/力矩扰动。

### 2.2 机械臂关节的识别

`AmovDrone.initialize`（`amov_drone.py:326`）：所有 DOF 名称中**不以 `"rotor"` 开头**的关节即为机械臂关节。

```
DOF 顺序: ['rotor_joint_1..4', 'joint1', 'joint2', ..., 'joint6', 'endleft', 'endright']
→ arm_joint_indices = 索引 4..11，num_arm_joints = 8
```

### 2.3 其它两种模式（对照）

- **`arm_hold`**（`HoverAmovArm.yaml`）：`hold_arm_joints` 把关节锁定在固定目标（默认全零），`zero_velocity_each_step` 可每步清零关节速度。臂作为**静止配重**。
- **`arm_motion`**（legacy）：`apply_arm_motion` 只驱动单个关节做正弦运动，其余关节保持零位。

### 2.4 复位（`_reset_idx`）

每回合复位时调用 `drone.reset_arm_joints`（`amov_drone.py:361`），将关节位置/速度/位置目标都置为目标值（动态/单关节模式为零），保证轨迹相位从 `progress_buf=0` 重新开始。

### 2.5 评测时的扰动扩展（`eval_hover_robustness.py`）

为测试泛化，评测脚本提供包装器在 `_reset_idx` 时重采样轨迹参数：
- `ArmPerturbWrapper`：在训练值附近 ±`perturb_scale` 扰动幅值/频率。
- `ArmRandomTrajWrapper`：每回合从独立均匀区间重采样 amplitude/frequency/phase（夹爪保持零位）。
- 对应 test 模式：`arm_perturb` / `arm_random` / `target_arm` / `target_arm_random`。

---

## 3. 机械臂对无人机扰动 force/wrench 的获取

### 3.1 物理耦合机制

机械臂作为同一 **articulation（关节链）**的子刚体，其运动通过关节约束自然地把反作用力/力矩传给 `base_link`——这是 PhysX 关节动力学自动求解的，**不需要手动注入扰动力**。旋翼推力则通过 `apply_action` 聚合后施加到 `base_link`（`amov_drone.py:391`）。因此机身感受到的臂扰动体现在其本体状态（位姿/速度）的变化上，被策略观测间接捕获（见 §1.3）。

### 3.2 显式测量臂–机身接头处的 wrench

要**定量获取**机械臂传给机身的 force/wrench，使用 Isaac Sim 的 articulation 接头力 API：

```python
jf = drone._view.get_measured_joint_forces()   # shape: (num_envs, num_bodies, 6)
```

- 返回每个 body 在其**父关节处**承受的 6 维广义力：`[:, body_idx, :3]` = 力 `Fx,Fy,Fz` (N)，`[:, body_idx, 3:]` = 力矩 `Tx,Ty,Tz` (N·m)。
- 取臂的**第一个连杆 `link1`** 对应的 `body_idx`，即可得到机械臂基座关节（臂↔机身连接处）传递的力/力矩——也就是机械臂对无人机施加的扰动 wrench。

`body_idx` 在运行时从 `_view.body_names` 解析（`eval_hover_robustness.py:449-470`）：

```python
_body_names = list(_view.body_names)
# 优先匹配 "joint1"/"link1"/...，回退到第一个非 rotor、非 base_link 的 body
_arm_body_idx = _body_names.index("link1")   # 示意
```

### 3.3 逐步记录与保存（评测脚本）

`eval_hover_robustness.py:509-527` 在每个仿真步读取并记录 env 0 的臂基座 wrench：

```python
jf = _view.get_measured_joint_forces()          # (num_envs, num_bodies, 6)
f_xyz = jf[0, _arm_body_idx, :3]                 # 力  (N)
t_xyz = jf[0, _arm_body_idx, 3:]                 # 力矩 (N·m)
_force_log.append((_global_step, f_xyz, t_xyz))
```

记录结果保存为 `joint_forces_{test_mode}.npy`（`eval_hover_robustness.py:558`），字段：`steps / forces(Fx,Fy,Fz) / torques(Tx,Ty,Tz) / body_name / test_mode`。

### 3.4 基线参考值

- **静止臂（`arm_hold`，HoverAmovArm）**：臂基座 `Fz ≈ -28 N`（对应臂自重 ~2.85 kg），`Ty` 标准差 ≈ 2.2 N·m。
- 动态臂模式下 force/wrench 随关节正弦运动而周期性时变，是策略需要抵抗的主要扰动来源。

### 3.5 在训练环境中也可记录（可选）

`hover.py` 中 `arm_log`（`hover.py:698-715`）会周期性打印臂关节角度（`get_joint_positions`），用于诊断；如需在训练侧记录 wrench，可同样调用 `drone._view.get_measured_joint_forces()`。

---

## 4. 小结

| 主题 | 关键点 |
|------|--------|
| **策略输入** | 30 维：`rpos(3)+rot(4)+vel(6)+heading(3)+up(3)+throttle(4)+rheading(3)+time(4)`；**不含臂状态**，臂扰动靠本体状态间接感知 |
| **臂动态运动** | 每关节独立正弦 `Aⱼsin(2πfⱼt+φⱼ)`，写入关节位置目标，PD 跟踪（stiffness=500, damping=50）；参数在 `HoverAmovDynaArm.yaml` |
| **扰动 wrench 获取** | 物理上由 articulation 关节自动耦合；定量测量用 `get_measured_joint_forces()` 取 `link1` 的 body 索引，得到臂基座 6 维力/力矩，逐步保存为 `.npy` |
```

