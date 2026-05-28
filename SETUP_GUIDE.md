# AMOV无人机仿真部署指南

**更新日期：** 2026-05-23  
**状态：** ✅ 真实参数已配置

---

## 📋 项目概述

**目标：** 在OmniDrones (Isaac Sim) 中实现AMOV无人机的Hover任务训练，为后续aerial manipulator系统奠定基础。

**关键成果：**
- ✅ 基于真实硬件参数配置无人机模型
- ✅ 推重比从0.01提升至1.81
- ✅ 创建优化的训练配置

---

## 🔧 硬件参数

### 真实硬件规格
```yaml
电机型号: X6 plus 6218-150KV
螺旋桨直径: 610mm
电池: 6S (22.2V)
机身质量: 10.21 kg
旋翼数量: 4
机臂长度: ~0.475m
```

### 仿真参数配置
```yaml
# amov_drone_real_params.yaml
mass: 10.211262 kg
force_constants: 5.155573e-04 N/(rad/s)²
max_rotation_velocities: 2830 RPM
moment_constants: 7.733359e-06 Nm/(rad/s)²
```

### 性能指标
```
单桨推力: 45.28 N
总推力: 181.12 N
推重比: 1.81
悬停推力占比: 55%
控制裕度: 45%
```

---

## 📊 与Firefly对比

| 参数 | Firefly | AMOV Real | 差异 |
|------|---------|-----------|------|
| **质量** | 1.57 kg | 10.21 kg | 6.5x |
| **旋翼数** | 6 | 4 | - |
| **单桨推力** | 0.07 N | 45.28 N | 688x |
| **总推力** | 0.39 N | 181.12 N | 459x |
| **推重比** | 0.03 ⚠️ | 1.81 ✓ | 70x |
| **机臂长** | 0.215 m | 0.475 m | 2.2x |

**关键发现：**
- Firefly原始配置推重比仅0.03，同样无法飞行
- AMOV配置为重型工业平台，适合大载荷应用
- 真实参数配置后具备良好的控制裕度

---

## 🚀 部署步骤

### 前置条件
```bash
# 确认OmniDrones已安装
cd /path/to/OmniDrones
conda activate omnidrones  # 或你的环境名
```

### Step 1: 复制文件

```bash
# 设置工作目录
AMOV_DIR="/path/to/aerial_manipulator"
OMNI_DIR="/path/to/OmniDrones"

# 1. 复制无人机类
cp $AMOV_DIR/amov_drone_omnidrones/amov_drone.py \
   $OMNI_DIR/omni_drones/robots/drone/

# 2. 复制USD模型和配置（使用真实参数版本）
cp $AMOV_DIR/amov_drone_omnidrones/amov_drone.usd \
   $OMNI_DIR/omni_drones/robots/assets/usd/
cp $AMOV_DIR/amov_drone_omnidrones/amov_drone_real_params.yaml \
   $OMNI_DIR/omni_drones/robots/assets/usd/amov_drone.yaml

# 3. 复制控制器配置
cp $AMOV_DIR/amov_drone_omnidrones/lee_controller_amov_drone.yaml \
   $OMNI_DIR/omni_drones/controllers/cfg/

# 4. 复制任务配置
cp $AMOV_DIR/AmovHover.yaml \
   $OMNI_DIR/cfg/task/
```

### Step 2: 注册无人机类

编辑 `$OMNI_DIR/omni_drones/robots/drone/__init__.py`：

```python
# 在文件末尾添加
from .amov_drone import AmovDrone
```

验证注册：
```bash
cd $OMNI_DIR
python -c "from omni_drones.robots.drone import AmovDrone; print('✓ AmovDrone registered')"
```

### Step 3: 验证配置

创建验证脚本 `verify_amov.py`：

```python
import yaml
import numpy as np

# 读取配置
with open('omni_drones/robots/assets/usd/amov_drone.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 提取参数
mass = config['mass']
num_rotors = config['rotor_configuration']['num_rotors']
k_f = config['rotor_configuration']['force_constants'][0]
max_rpm = config['rotor_configuration']['max_rotation_velocities'][0]

# 计算性能
max_omega = max_rpm * 2 * np.pi / 60
thrust_per_rotor = k_f * max_omega**2
total_thrust = num_rotors * thrust_per_rotor
thrust_to_weight = total_thrust / (mass * 9.81)

print("=" * 60)
print("AMOV无人机配置验证")
print("=" * 60)
print(f"质量: {mass:.2f} kg")
print(f"旋翼数: {num_rotors}")
print(f"推力系数: {k_f:.6e} N/(rad/s)²")
print(f"最大转速: {max_rpm} RPM ({max_omega:.1f} rad/s)")
print(f"\n单桨推力: {thrust_per_rotor:.2f} N")
print(f"总推力: {total_thrust:.2f} N")
print(f"推重比: {thrust_to_weight:.2f}")
print(f"悬停推力占比: {(mass * 9.81 / total_thrust * 100):.1f}%")
print("=" * 60)

if thrust_to_weight > 2.0:
    print("✓ 推重比优秀，可以开始训练")
elif thrust_to_weight > 1.5:
    print("✓ 推重比良好，可以训练")
else:
    print("⚠️ 推重比不足，需要调整参数")
```

运行验证：
```bash
cd $OMNI_DIR
python verify_amov.py
```

**预期输出：**
```
============================================================
AMOV无人机配置验证
============================================================
质量: 10.21 kg
旋翼数: 4
推力系数: 5.155573e-04 N/(rad/s)²
最大转速: 2830 RPM (296.4 rad/s)

单桨推力: 45.28 N
总推力: 181.12 N
推重比: 1.81
悬停推力占比: 55.3%
============================================================
✓ 推重比良好，可以训练
```

---

## 🎮 训练执行

### 可视化训练（调试）

```bash
cd $OMNI_DIR
python scripts/train.py task=AmovHover headless=false num_envs=16
```

**观察要点：**
- 无人机应能稳定悬停
- 初期会有震荡，逐渐收敛
- 不应出现立即坠落

### 正式训练

```bash
# 单GPU训练
python scripts/train.py task=AmovHover headless=true num_envs=512

# 多GPU训练（如果可用）
python scripts/train.py task=AmovHover headless=true num_envs=2048 \
    multi_gpu=true
```

### 训练配置说明

`AmovHover.yaml` 针对重型无人机优化：

```yaml
# 关键参数调整
reward_distance_scale: 0.5          # 降低距离敏感度（重型机响应慢）
reward_effort_weight: 0.01          # 降低能量惩罚（需要更大推力）
reward_action_smoothness_weight: 0.05  # 增加平滑性（避免震荡）

# 训练参数
num_envs: 512                       # 并行环境数
max_episode_length: 500             # episode最大长度
```

---

## 📈 监控指标

### 关键指标

| 指标 | 初始值 | 目标值 | 说明 |
|------|--------|--------|------|
| `pos_error` | ~2.5m | <0.5m | 位置误差 |
| `episode_len` | ~50 | 500 | episode长度 |
| `uprightness` | ~0.7 | >0.95 | 姿态正直性 |
| `return` | ~-10 | >100 | 累积奖励 |

### 训练曲线（预期）

```
Frames    | pos_error | episode_len | return | 状态
----------|-----------|-------------|--------|------
0         | 2.5       | 50          | -10    | 初始
500K      | 1.8       | 150         | 20     | 学习中
1M        | 1.2       | 300         | 50     | 改善
2M        | 0.8       | 500         | 80     | 接近收敛
5M        | 0.3       | 500         | 120    | 收敛 ✓
```

### 使用WandB监控

```bash
# 登录WandB（首次）
wandb login

# 训练时自动记录到WandB
python scripts/train.py task=AmovHover wandb.mode=online
```

---

## 🔧 故障排查

### 问题1: 无人机立即坠落

**症状：** episode_len < 20，无人机快速下坠

**原因：** 推力不足或控制器增益过低

**解决：**
```bash
# 1. 验证推重比
python verify_amov.py

# 2. 如果推重比<1.5，检查配置文件
# 确认使用的是 amov_drone_real_params.yaml

# 3. 增加控制器增益（如需要）
# 编辑 lee_controller_amov_drone.yaml
position_gain: [10, 10, 20]  # 增加20%
velocity_gain: [6, 6, 12]
```

### 问题2: 剧烈震荡

**症状：** 无人机快速震荡，pos_error波动大

**原因：** 控制器增益过大或奖励函数过敏感

**解决：**
```yaml
# 方案A: 降低控制器增益
# lee_controller_amov_drone.yaml
position_gain: [7, 7, 14]
velocity_gain: [5, 5, 9]

# 方案B: 降低奖励敏感度
# AmovHover.yaml
reward_distance_scale: 0.3  # 从0.5降到0.3
reward_action_smoothness_weight: 0.1  # 从0.05增到0.1
```

### 问题3: 训练不收敛

**症状：** 5M frames后指标仍无明显改善

**诊断步骤：**

1. **检查推重比**
```bash
python verify_amov.py
# 必须 > 1.5
```

2. **可视化观察**
```bash
python scripts/train.py task=AmovHover headless=false num_envs=16
# 观察行为模式
```

3. **调整学习率**
```yaml
# cfg/algo/mappo.yaml
lr: 1e-4  # 从3e-4降低
```

4. **增加训练时间**
```bash
# 重型无人机可能需要更长训练时间
python scripts/train.py task=AmovHover max_iterations=10000
```

### 问题4: ImportError

**症状：** `cannot import name 'AmovDrone'`

**解决：**
```bash
# 检查文件
ls $OMNI_DIR/omni_drones/robots/drone/amov_drone.py

# 检查注册
grep "AmovDrone" $OMNI_DIR/omni_drones/robots/drone/__init__.py

# 如果缺失，手动添加
echo "from .amov_drone import AmovDrone" >> \
    $OMNI_DIR/omni_drones/robots/drone/__init__.py
```

---

## 🎯 成功标准

训练成功的标志：

- ✅ `pos_error` < 0.5m
- ✅ `episode_len` = 500 (达到最大长度)
- ✅ `uprightness` > 0.95
- ✅ `return` > 100
- ✅ 可视化观察稳定悬停

达到这些标准后，可以进行下一步开发。

---

## 🚁 下一步：Aerial Manipulator

### 1. 准备机械臂模型

**要求：**
- USD格式，包含正确的关节配置
- 质量 < 2kg（保持推重比 > 1.5）
- 自由度：3-5 DOF

### 2. 集成机械臂

```python
# 在Isaac Sim中
# 1. 打开 amov_drone.usd
# 2. 添加机械臂为子prim
# 3. 创建固定关节连接到 base_link
# 4. 调整质量和惯性
# 5. 保存为 amov_manipulator.usd
```

### 3. 创建新环境

```python
# omni_drones/envs/single/aerial_manipulator.py
class AerialManipulator(IsaacEnv):
    def __init__(self, cfg):
        # 扩展观测：无人机状态 + 机械臂关节状态
        # 扩展动作：无人机控制 + 机械臂控制
        # 新奖励：末端位置误差 + 无人机稳定性
        ...
```

### 4. MPC+RL混合控制

**架构：**
```
RL Policy (高层)
    ↓ 输出末端期望轨迹
MPC Controller (底层)
    ↓ 优化联合控制
Aerial Manipulator System
```

**实现要点：**
- 使用CasADi实现非线性MPC
- 预测时域：20步 (0.4秒 @ 50Hz)
- 优化目标：末端跟踪 + 无人机稳定 + 能量 + 平滑性

---

## 📁 文件清单

### 必需文件
```
amov_drone_omnidrones/
├── amov_drone.py                      # 无人机类定义
├── amov_drone.usd                     # USD模型
├── amov_drone_real_params.yaml        # 真实参数配置 ✓
└── lee_controller_amov_drone.yaml     # 控制器增益

AmovHover.yaml                         # 优化的训练配置
```

### 辅助文件
```
verify_amov.py                         # 配置验证脚本
```

---

## 💡 关键经验

### 参数配置
1. **推重比是关键**：必须 > 1.5，理想 > 2.0
2. **真实参数优先**：基于实际硬件规格计算
3. **验证后训练**：先运行 verify_amov.py

### 训练策略
1. **先可视化**：观察行为再正式训练
2. **监控关键指标**：pos_error 最重要
3. **耐心等待**：重型机可能需要更长训练时间

### 调试技巧
1. **推力不足** → 检查配置文件
2. **震荡** → 降低增益或奖励敏感度
3. **不收敛** → 降低学习率，增加训练时间

---

## 📞 技术支持

### 常见问题
- 推重比计算：`T_total / (m * g)`
- 悬停推力：`m * g`
- 控制裕度：`(T_total - m*g) / T_total`

### 参数调整公式
```python
# 推力系数调整
k_f_new = k_f_old * (target_ratio / current_ratio)

# 最大转速调整
omega_new = omega_old * sqrt(target_ratio / current_ratio)
```

---

## 📝 更新日志

### 2026-05-23
- ✅ 基于真实硬件参数（X6 plus 6218-150KV, 610mm螺旋桨）
- ✅ 推重比从0.01提升至1.81
- ✅ 创建 amov_drone_real_params.yaml
- ✅ 整合所有文档为单一部署指南

### 2026-05-22
- 初始分析和配置
- 识别推重比不足问题
- 创建多个分析文档

---

**当前状态：** ✅ 参数已配置，可以开始训练  
**下一里程碑：** 完成Hover任务训练  
**预计时间：** 2-4小时（单GPU，512环境）
