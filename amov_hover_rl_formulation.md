# AmovDrone 悬停策略：观测空间、动作映射与奖励函数

> 任务：`HoverAmovDynaArm` — 机械臂按正弦轨迹运动条件下的鲁棒悬停  
> 机体：AmovDrone + Airbot Play 6-DOF 机械臂，总质量 $m = 13.332\ \mathrm{kg}$，4 旋翼

---

## 1. 观测空间

观测向量 $\mathbf{o}_t \in \mathbb{R}^{30}$ 在每个控制步由以下分量拼接构成：

$$
\mathbf{o}_t = \bigl[\,\Delta\mathbf{p},\;\mathbf{q},\;\mathbf{v}^w,\;\boldsymbol{\omega}^w,\;\mathbf{h},\;\mathbf{u},\;\tilde{\boldsymbol{\alpha}},\;\Delta\mathbf{h},\;\boldsymbol{\tau}_t\,\bigr]
$$

各分量说明：

| 符号 | 维度 | 含义 |
|------|------|------|
| $\Delta\mathbf{p} = \mathbf{p}_{\text{goal}} - \mathbf{p}$ | 3 | 目标位置与当前位置之差（环境坐标系） |
| $\mathbf{q} = [q_w, q_x, q_y, q_z]$ | 4 | 机体姿态四元数（世界系） |
| $\mathbf{v}^w = [v_x, v_y, v_z]$ | 3 | 世界系线速度 |
| $\boldsymbol{\omega}^w = [\omega_x, \omega_y, \omega_z]$ | 3 | 世界系角速度 |
| $\mathbf{h} = R\,\mathbf{e}_x$ | 3 | 机头朝向向量（世界系） |
| $\mathbf{u} = R\,\mathbf{e}_z$ | 3 | 机体竖直向量（世界系） |
| $\tilde{\boldsymbol{\alpha}} = 2\boldsymbol{\alpha} - \mathbf{1}$ | 4 | 归一化旋翼油门状态 $\in [-1,1]^4$ |
| $\Delta\mathbf{h} = \mathbf{h}_{\text{goal}} - \mathbf{h}$ | 3 | 期望朝向与当前朝向偏差 |
| $\boldsymbol{\tau}_t = [t/T, t/T, t/T, t/T]$ | 4 | 时间编码（$t$ 为当前步，$T$ 为最大步数） |

机械臂关节角度**不包含在观测中**。臂运动引起的质心偏移与惯量变化作为隐式外扰，策略通过无人机飞行状态反馈进行补偿。

---

## 2. 动作空间与推力映射

### 2.1 动作空间

策略输出 4 维连续动作 $\mathbf{a}_t \in [-1,1]^4$，每个分量对应一个旋翼的指令。

### 2.2 悬停工作点归一化

为使探索对称分布于悬停平衡点，先将策略输出线性映射至旋翼指令空间：

$$
c_i = a_i \cdot s_{\mathrm{h}} + \delta_{\mathrm{h}}, \qquad c_i \in [-1,\,1]
$$

悬停偏置和缩放因子由机体参数在线确定：

$$
\eta = \frac{mg}{\displaystyle\sum_{i=1}^{4} K_{F,i}}, \qquad
\delta_{\mathrm{h}} = 2\eta - 1, \qquad
s_{\mathrm{h}} = 1 - |\delta_{\mathrm{h}}|
$$

对 AmovDrone（$K_{F,i} \approx 126.58\ \mathrm{N}$）：

$$
\eta \approx 0.258, \quad \delta_{\mathrm{h}} \approx -0.484, \quad s_{\mathrm{h}} \approx 0.516
$$

映射效果：$a_i{=}0$ 对应悬停平衡；$a_i{=}{+}1$ 对应接近最大推力；$a_i{=}{-}1$ 对应零推力。

### 2.3 旋翼一阶动力学

指令 $c_i$ 首先转换为目标油门，再经一阶滞后滤波器更新实际油门：

$$
\hat{\alpha}_i = \sqrt{\mathrm{clamp}\!\left(\frac{c_i + 1}{2},\,0,\,1\right)}
$$

$$
\alpha_i \;\leftarrow\; \alpha_i + \tau\,(\hat{\alpha}_i - \alpha_i), \qquad
\tau = \begin{cases} \tau_{\uparrow} & \hat{\alpha}_i > \alpha_i \\ \tau_{\downarrow} & \text{otherwise} \end{cases}
$$

其中 $\tau_{\uparrow} = \tau_{\downarrow} = 0.43$（电机响应时间常数，无量纲，表示每控制步的跟随比例）。

### 2.4 推力与反扭矩

$$
F_i = K_{F,i}\,\alpha_i^2, \qquad M_i = -K_{M,i}\,\alpha_i^2\,d_i
$$

$d_i \in \{-1,+1\}$ 为旋翼转向，交替正反转以抵消偏航力矩。

旋翼参数由 YAML 配置文件给定：

$$
K_{F,i} = \Omega_{\max}^2 \cdot k_f \approx 126.58\ \mathrm{N}, \quad
K_{M,i} = \Omega_{\max}^2 \cdot k_m \approx 1.90\ \mathrm{N{\cdot}m}
$$

$$
\Omega_{\max} = 495.52\ \mathrm{rad/s}, \quad k_f = 5.156\times10^{-4}, \quad k_m = 7.733\times10^{-6}
$$

### 2.5 合力与合力矩

所有旋翼推力合并后施加到 base\_link：

$$
\mathbf{F}_{\mathrm{net}} = \sum_{i=1}^{4} F_i\,\hat{\mathbf{e}}_{z,i} + c_d\,m\,\mathbf{v}^w_{1:3}
$$

$$
\boldsymbol{\tau}_{\mathrm{net}} = \underbrace{\sum_{i=1}^{4}\!\left[(\mathbf{r}_i - \mathbf{r}_{\mathrm{ref}}) \times F_i\,\hat{\mathbf{e}}_{z,i}\right]}_{\text{pitch/roll}} + \underbrace{\sum_{i=1}^{4} M_i\,\hat{\mathbf{e}}_{z,i}}_{\text{yaw}}
$$

$\hat{\mathbf{e}}_{z,i}$ 为旋翼 z 轴世界系方向，$\mathbf{r}_i$ 为旋翼世界系位置，$\mathbf{r}_{\mathrm{ref}}$ 为 base\_link 质心位置，$c_d = 0.3$ 为线性气动阻力系数。

---

## 3. 机械臂轨迹

### 3.1 训练轨迹（固定正弦）

各关节由 PD 驱动器跟踪独立正弦参考轨迹（刚度 $K_s=500\ \mathrm{N{\cdot}m/rad}$，阻尼 $K_d=50\ \mathrm{N{\cdot}m{\cdot}s/rad}$）：

$$
q_j(t_k) = A_j\,\sin\!\bigl(2\pi f_j\,t_k + \varphi_j\bigr), \qquad t_k = k\,\Delta t_{\mathrm{ctrl}}
$$

$k$ 为控制步数，$\Delta t_{\mathrm{ctrl}} = \mathrm{dt} \times \mathrm{substeps}$。训练参数如下：

| 关节 | $A_j\ \mathrm{(rad)}$ | $f_j\ \mathrm{(Hz)}$ | $\varphi_j\ \mathrm{(rad)}$ |
|------|----------------------|--------------------|--------------------------|
| joint1（底座偏航） | 0.60 | 0.25 | 0.000 |
| joint2（肩部俯仰） | 0.40 | 0.30 | 1.047 |
| joint3（肘部俯仰） | 0.35 | 0.35 | 2.094 |
| joint4（前臂滚转） | 0.25 | 0.40 | 0.000 |
| joint5（腕部俯仰） | 0.25 | 0.45 | 1.047 |
| joint6（腕部滚转） | 0.15 | 0.50 | 2.094 |

### 3.2 评估扰动（`test=target_arm_random`）

每个回合重置时重新独立采样各关节轨迹参数：

$$
A_j \sim \mathcal{U}(0.1,\;0.5)\ \mathrm{rad}, \quad
f_j \sim \mathcal{U}(1,\;5)\ \mathrm{Hz}, \quad
\varphi_j \sim \mathcal{U}(0,\;2\pi)\ \mathrm{rad}
$$

同时目标悬停位置每 100 步随机变换：

$$
[x_g,\,y_g] \sim \mathcal{U}([-1.5,1.5]^2)\ \mathrm{m}, \quad z_g = 2.0\ \mathrm{m}
$$

---

## 4. 奖励函数

总奖励为以下各项之和：

$$
\boxed{r_t = r_{\mathrm{pose}} + r_{\mathrm{pose}}\,(r_{\mathrm{up}} + r_{\mathrm{spin}}) + r_{\mathrm{alive}} + r_{\mathrm{up,s}}}
$$

> 注：能耗项 $r_{\mathrm{effort}}$ 和平滑项 $r_{\mathrm{smooth}}$ 在本配置中权重为 0，已禁用。

### 4.1 位姿奖励

综合位置误差与朝向偏差，以反 Lorentz 函数形式给出：

$$
d = \bigl\|\bigl[\Delta\mathbf{p};\;\Delta\mathbf{h}\bigr]\bigr\|_2, \qquad
r_{\mathrm{pose}} = \frac{1}{1 + (s_d\,d)^2}, \quad s_d = 1.2
$$

### 4.2 竖直姿态奖励（位姿门控）

$$
r_{\mathrm{up}} = \left(\frac{u_z + 1}{2}\right)^{\!2}
$$

乘以 $r_{\mathrm{pose}}$ 形成门控，位置误差较大时压制姿态梯度，避免策略在高位置误差下仅优化姿态。

### 4.3 反自旋奖励（位姿门控）

$$
r_{\mathrm{spin}} = \frac{1}{1 + \omega_z^4}
$$

### 4.4 独立竖直奖励

$$
r_{\mathrm{up,s}} = w_{\mathrm{up}}\,\frac{u_z + 1}{2}, \quad w_{\mathrm{up}} = 0.5
$$

不依赖位置误差，训练早期（位置误差大、$r_{\mathrm{pose}}$ 接近 0 时）提供持续的保持直立信号。

### 4.5 生存高度奖励

从地面到目标高度提供连续上升梯度：

$$
r_{\mathrm{alive}} = \mathrm{clamp}\!\left(\frac{z - z_{\mathrm{reset}}}{z_{\mathrm{goal}} - z_{\mathrm{reset}}},\;0,\;1\right)
$$

其中 $z_{\mathrm{reset}} = 0.55\ \mathrm{m}$，$z_{\mathrm{goal}} = 2.0\ \mathrm{m}$。

### 4.6 各项汇总

| 奖励项 | 公式 | 说明 |
|--------|------|------|
| $r_{\mathrm{pose}}$ | $1/[1+(1.2\,d)^2]$ | 位置+朝向综合误差 |
| $r_{\mathrm{pose}} \cdot r_{\mathrm{up}}$ | $r_{\mathrm{pose}} \cdot ((u_z{+}1)/2)^2$ | 位姿门控竖直奖励 |
| $r_{\mathrm{pose}} \cdot r_{\mathrm{spin}}$ | $r_{\mathrm{pose}} / (1{+}\omega_z^4)$ | 位姿门控反自旋奖励 |
| $r_{\mathrm{alive}}$ | $\mathrm{clamp}((z{-}0.55)/1.45,\,0,\,1)$ | 高度生存梯度 |
| $r_{\mathrm{up,s}}$ | $0.5\,(u_z{+}1)/2$ | 独立竖直梯度 |

---

## 5. 回合终止条件

$$
\text{terminated} \;\iff\; z < 0.55\ \mathrm{m} \;\;\text{OR}\;\; d > 4\ \mathrm{m} \;\;\text{OR}\;\; u_z < -0.3 \;\;\text{OR}\;\; \text{state contains NaN}
$$

$$
\text{truncated} \;\iff\; k \geq T_{\max}
$$

训练时 $T_{\max}=500$，评估时 $T_{\max}=600$（命令行指定）；评估时关闭提前终止，确保完整回合运行。

---

## 6. 关键参数一览

| 参数 | 值 |
|------|----|
| 观测维度 | 30 |
| 动作维度 | 4（各旋翼独立） |
| 奖励距离缩放 $s_d$ | 1.2 |
| 独立竖直权重 $w_{\mathrm{up}}$ | 0.5 |
| 机械臂关节 PD 刚度 | $500\ \mathrm{N{\cdot}m/rad}$ |
| 机械臂关节 PD 阻尼 | $50\ \mathrm{N{\cdot}m{\cdot}s/rad}$ |
| 电机时间常数 $\tau_{\uparrow/\downarrow}$ | 0.43 |
| 气动阻力系数 $c_d$ | 0.3 |
| 机体总质量 | $13.332\ \mathrm{kg}$ |
| 单旋翼最大推力 $K_F$ | $\approx 126.58\ \mathrm{N}$ |
| 并行训练环境数 | 128 |
