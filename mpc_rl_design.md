# Aerial Manipulator MPC+RL混合控制方案

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    High-Level RL Policy                     │
│  输入：任务目标、环境感知、当前状态                          │
│  输出：机械臂末端期望轨迹 (x_ee_des, v_ee_des)              │
│  训练目标：学习复杂操作策略（抓取、避障、路径规划）          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   MPC Controller Layer                      │
│  输入：末端期望轨迹 + 当前系统状态                           │
│  优化目标：                                                  │
│    - 跟踪末端轨迹                                            │
│    - 保持无人机稳定                                          │
│    - 补偿机械臂扰动                                          │
│  输出：无人机推力 + 机械臂关节力矩                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Aerial Manipulator System                      │
│  - 四旋翼无人机（动力学耦合）                                │
│  - 机械臂（运动产生扰动）                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. 数学建模

### 1.1 系统动力学

#### 无人机动力学（考虑机械臂扰动）
```
m_total * a_base = F_thrust + m_total * g + F_arm_reaction
I_total * α = τ_thrust + τ_arm_reaction
```

其中：
- `m_total = m_drone + m_arm`：总质量
- `I_total`：总转动惯量（随机械臂姿态变化）
- `F_arm_reaction`：机械臂运动产生的反作用力
- `τ_arm_reaction`：机械臂运动产生的反作用力矩

#### 机械臂动力学
```
M(q) * q̈ + C(q, q̇) * q̇ + G(q) = τ + τ_base_motion
```

其中：
- `q`：关节角度
- `τ`：关节力矩
- `τ_base_motion`：无人机运动对机械臂的影响

#### 耦合项计算
```python
# 机械臂对无人机的扰动力
F_arm_reaction = -m_arm * (a_base + a_arm_rel)

# 机械臂对无人机的扰动力矩
τ_arm_reaction = -r_com × F_arm_reaction - I_arm × α_arm
```

### 1.2 末端位姿正运动学
```
x_ee = f_fk(q_base, q_arm)
v_ee = J(q) * [v_base; q̇_arm]
```

---

## 2. MPC控制器设计

### 2.1 优化问题formulation

```python
minimize:
    Σ_{k=0}^{N-1} [
        ||x_ee(k) - x_ee_des(k)||²_Q_ee +      # 末端跟踪误差
        ||x_base(k) - x_base_des(k)||²_Q_base + # 无人机位置误差
        ||θ_base(k)||²_Q_att +                  # 姿态误差
        ||u(k)||²_R +                           # 控制输入代价
        ||Δu(k)||²_R_smooth                     # 控制平滑性
    ]

subject to:
    # 系统动力学约束
    x(k+1) = f(x(k), u(k))
    
    # 无人机推力约束
    u_min ≤ u_thrust(k) ≤ u_max
    
    # 机械臂关节约束
    q_min ≤ q_arm(k) ≤ q_max
    τ_min ≤ τ_arm(k) ≤ τ_max
    
    # 安全约束
    z_base(k) ≥ z_min  # 最低高度
    ||θ_base(k)|| ≤ θ_max  # 最大倾角
```

### 2.2 关键参数

```python
# 预测时域
N = 20  # 0.4秒 @ 50Hz

# 权重矩阵
Q_ee = diag([100, 100, 100, 10, 10, 10])  # 末端位置+姿态
Q_base = diag([10, 10, 50, 1, 1, 1])      # 无人机位置+速度
Q_att = diag([50, 50, 20])                # roll, pitch, yaw
R = diag([0.1, 0.1, 0.1, 0.1, 1, 1, 1])   # 4电机 + 3关节
R_smooth = diag([1, 1, 1, 1, 5, 5, 5])    # 平滑性权重
```

### 2.3 实现方案

#### 方案A：基于CasADi的非线性MPC（推荐）

```python
import casadi as ca

class AerialManipulatorMPC:
    def __init__(self, dt=0.02, N=20):
        self.dt = dt
        self.N = N
        self._build_optimizer()
    
    def _build_optimizer(self):
        # 状态变量 [x, y, z, vx, vy, vz, qw, qx, qy, qz, ωx, ωy, ωz, q_arm, q̇_arm]
        nx = 13 + 2*n_joints
        # 控制变量 [f1, f2, f3, f4, τ_arm]
        nu = 4 + n_joints
        
        # 创建优化变量
        X = ca.MX.sym('X', nx, self.N+1)
        U = ca.MX.sym('U', nu, self.N)
        
        # 目标轨迹参数
        X_ee_ref = ca.MX.sym('X_ee_ref', 6, self.N)
        
        # 构建代价函数
        cost = 0
        for k in range(self.N):
            # 计算末端位姿
            x_ee_k = self.forward_kinematics(X[:, k])
            cost += ca.mtimes([
                (x_ee_k - X_ee_ref[:, k]).T,
                self.Q_ee,
                (x_ee_k - X_ee_ref[:, k])
            ])
            
            # 无人机稳定性代价
            cost += ca.mtimes([X[:3, k].T, self.Q_base, X[:3, k]])
            
            # 控制代价
            cost += ca.mtimes([U[:, k].T, self.R, U[:, k]])
            
            if k > 0:
                du = U[:, k] - U[:, k-1]
                cost += ca.mtimes([du.T, self.R_smooth, du])
        
        # 动力学约束
        g = []
        for k in range(self.N):
            x_next = self.dynamics(X[:, k], U[:, k])
            g.append(X[:, k+1] - x_next)
        
        # 构建NLP问题
        nlp = {
            'x': ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1)),
            'f': cost,
            'g': ca.vertcat(*g),
            'p': ca.reshape(X_ee_ref, -1, 1)
        }
        
        # 求解器选项
        opts = {
            'ipopt.print_level': 0,
            'ipopt.max_iter': 100,
            'ipopt.tol': 1e-4
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
    
    def dynamics(self, x, u):
        """耦合系统动力学"""
        # 提取状态
        pos = x[:3]
        vel = x[3:6]
        quat = x[6:10]
        omega = x[10:13]
        q_arm = x[13:13+n_joints]
        qd_arm = x[13+n_joints:]
        
        # 提取控制
        f_motors = u[:4]
        tau_arm = u[4:]
        
        # 计算机械臂对无人机的扰动
        F_disturbance, tau_disturbance = self.compute_arm_disturbance(
            q_arm, qd_arm, tau_arm
        )
        
        # 无人机动力学
        F_thrust = self.motor_to_thrust(f_motors)
        acc = (F_thrust + F_disturbance) / self.m_total + self.g
        alpha = ca.solve(self.I_total, tau_disturbance)
        
        # 机械臂动力学
        qdd_arm = self.arm_dynamics(q_arm, qd_arm, tau_arm, acc, alpha)
        
        # 状态导数
        xdot = ca.vertcat(
            vel,
            acc,
            self.quat_derivative(quat, omega),
            alpha,
            qd_arm,
            qdd_arm
        )
        
        return x + xdot * self.dt
    
    def solve(self, x0, x_ee_ref):
        """求解MPC问题"""
        # 初始猜测
        x_init = ca.repmat(x0, 1, self.N+1)
        u_init = ca.DM.zeros(self.nu, self.N)
        
        # 求解
        sol = self.solver(
            x0=ca.vertcat(ca.reshape(x_init, -1, 1), ca.reshape(u_init, -1, 1)),
            lbg=0, ubg=0,
            p=ca.reshape(x_ee_ref, -1, 1)
        )
        
        # 提取最优控制
        u_opt = ca.reshape(sol['x'][self.nx*(self.N+1):], self.nu, self.N)
        return u_opt[:, 0]  # 返回第一个控制输入
```

#### 方案B：基于线性化的快速MPC

如果计算资源有限，可以使用线性化MPC：

```python
import cvxpy as cp

class LinearizedMPC:
    def __init__(self, dt=0.02, N=20):
        self.dt = dt
        self.N = N
    
    def solve(self, x0, x_ee_ref, x_ref_traj):
        """在参考轨迹附近线性化"""
        # 在参考轨迹上计算雅可比矩阵
        A, B = self.linearize_dynamics(x_ref_traj)
        
        # 定义优化变量
        X = cp.Variable((self.nx, self.N+1))
        U = cp.Variable((self.nu, self.N))
        
        # 构建QP问题
        cost = 0
        constraints = [X[:, 0] == x0]
        
        for k in range(self.N):
            # 末端跟踪代价
            x_ee_k = self.C_ee @ X[:, k]
            cost += cp.quad_form(x_ee_k - x_ee_ref[:, k], self.Q_ee)
            
            # 控制代价
            cost += cp.quad_form(U[:, k], self.R)
            
            # 线性化动力学约束
            constraints += [
                X[:, k+1] == A[k] @ X[:, k] + B[k] @ U[:, k]
            ]
            
            # 输入约束
            constraints += [
                self.u_min <= U[:, k],
                U[:, k] <= self.u_max
            ]
        
        # 求解
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, warm_start=True)
        
        return U[:, 0].value
```

---

## 3. RL策略设计

### 3.1 观测空间

```python
obs = {
    # 任务相关
    "target_pos": [x, y, z],              # 目标位置
    "target_ori": [qw, qx, qy, qz],       # 目标姿态
    "ee_to_target": [dx, dy, dz, ...],    # 末端到目标的相对位姿
    
    # 系统状态
    "base_pos": [x, y, z],
    "base_vel": [vx, vy, vz],
    "base_ori": [qw, qx, qy, qz],
    "base_omega": [ωx, ωy, ωz],
    "arm_joint_pos": [q1, q2, ...],
    "arm_joint_vel": [q̇1, q̇2, ...],
    "ee_pos": [x, y, z],
    "ee_vel": [vx, vy, vz],
    
    # 环境感知（可选）
    "obstacles": [...],                    # 障碍物信息
    "lidar": [...],                        # 激光雷达
    
    # 时间信息
    "time_encoding": [sin(t), cos(t), ...]
}
```

### 3.2 动作空间

**方案1：直接输出末端期望轨迹**
```python
action = {
    "ee_pos_delta": [Δx, Δy, Δz],        # 末端位置增量
    "ee_ori_delta": [Δroll, Δpitch, Δyaw] # 末端姿态增量
}
```

**方案2：输出末端速度指令**
```python
action = {
    "ee_vel": [vx, vy, vz],               # 末端线速度
    "ee_omega": [ωx, ωy, ωz]              # 末端角速度
}
```

### 3.3 奖励函数

```python
def compute_reward(self, obs, action):
    # 末端跟踪奖励
    ee_error = torch.norm(obs["ee_to_target"], dim=-1)
    reward_tracking = torch.exp(-2.0 * ee_error)
    
    # 无人机稳定性奖励
    base_tilt = torch.acos(obs["base_ori"][:, 2].clamp(-1, 1))
    reward_stability = torch.exp(-5.0 * base_tilt)
    
    # 能量消耗惩罚
    reward_energy = -0.01 * torch.sum(torch.abs(action), dim=-1)
    
    # 平滑性奖励
    action_diff = action - self.last_action
    reward_smooth = -0.1 * torch.norm(action_diff, dim=-1)
    
    # 安全性惩罚
    penalty_collision = -10.0 * self.check_collision()
    penalty_height = -10.0 * (obs["base_pos"][:, 2] < 0.5)
    
    # 总奖励
    reward = (
        reward_tracking +
        0.5 * reward_stability +
        reward_energy +
        reward_smooth +
        penalty_collision +
        penalty_height
    )
    
    return reward
```

### 3.4 训练策略

使用PPO或SAC算法：

```python
# 在OmniDrones框架中实现
class AerialManipulatorEnv(IsaacEnv):
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self.mpc_controller = AerialManipulatorMPC()
    
    def _pre_sim_step(self, tensordict):
        # RL输出末端期望轨迹
        ee_des = tensordict[("agents", "action")]
        
        # MPC计算底层控制
        u_mpc = self.mpc_controller.solve(
            self.get_state(),
            ee_des
        )
        
        # 应用控制
        self.apply_control(u_mpc)
```

---

## 4. 在OmniDrones中的实现步骤

### Step 1: 创建Aerial Manipulator环境

```python
# omni_drones/envs/single/aerial_manipulator.py
class AerialManipulator(IsaacEnv):
    def _design_scene(self):
        # 加载无人机+机械臂USD模型
        self.system = ArticulationView(...)
        
    def _compute_state_and_obs(self):
        # 计算末端位姿
        ee_pos = self.compute_ee_position()
        
        obs = torch.cat([
            self.target_pos - ee_pos,
            self.system.get_joint_positions(),
            ...
        ], dim=-1)
        
        return obs
    
    def _compute_reward_and_done(self):
        # 实现上述奖励函数
        ...
```

### Step 2: 集成MPC控制器

```python
# omni_drones/controllers/mpc_controller.py
class MPCController:
    def __init__(self, cfg):
        # 初始化MPC求解器
        ...
    
    def compute_control(self, state, ee_ref):
        # 求解MPC问题
        return u_opt
```

### Step 3: 修改训练流程

```python
# scripts/train_aerial_manipulator.py
env = AerialManipulator(cfg, headless=cfg.headless)

# RL策略输出末端轨迹
policy = ALGOS["ppo"](...)

for step in range(total_steps):
    # RL决策
    ee_ref = policy(obs)
    
    # MPC执行
    action = env.mpc_controller.compute_control(state, ee_ref)
    
    # 环境步进
    obs, reward, done, info = env.step(action)
```

---

## 5. 预期效果与优势

### 优势
1. **RL负责高层决策**：学习复杂的操作策略，无需手工设计
2. **MPC保证底层稳定**：实时补偿扰动，保证安全性
3. **解耦复杂性**：避免端到端RL学习整个系统动力学
4. **可解释性强**：MPC层的行为可预测、可调试

### 预期性能
- 末端跟踪精度：< 5cm
- 无人机姿态偏差：< 10°
- 响应时间：< 0.5s
- 训练时间：~10M frames（约2-3天，8 GPU）

---

## 6. 下一步工作

1. **完成Hover任务调试**（当前阶段）
2. **添加机械臂模型到USD**
3. **实现MPC控制器**（先用简化模型测试）
4. **设计RL奖励函数**
5. **训练并评估性能**
6. **实机验证**（如果有硬件）

---

## 参考文献

1. Lee et al., "Geometric tracking control of a quadrotor UAV on SE(3)", CDC 2010
2. Ott et al., "Unified Impedance and Admittance Control", ICRA 2010
3. Neunert et al., "Fast Nonlinear Model Predictive Control for Unified Trajectory Optimization", ICRA 2016
4. Schulman et al., "Proximal Policy Optimization Algorithms", arXiv 2017
