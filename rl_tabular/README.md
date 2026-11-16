# RL-3：Tabular RL 与线性函数近似

本章把状态价值、动作价值和 Bellman 更新落实为可阅读的代码，并用一个线性模型过渡到参数学习。学习路线与待办统一见 [TASK](../TASK.md)。

| 内容 | 代码入口 |
| --- | --- |
| 环境的状态、动作、奖励与转移 | [envs/gridworld.py](envs/gridworld.py) |
| 3A：Bellman backup、Policy Evaluation、VI / PI | [model_based_planning.py](model_based_planning.py) |
| 3B：MC、TD(0)、ε-greedy、SARSA、Q-learning | [model_free_planning.py](model_free_planning.py) |
| 3C：一次线性函数近似更新 | [linear_approximation_demo.py](linear_approximation_demo.py) |
| 手算轨迹、终点与 Bellman 检查 | [tests/test_planning.py](tests/test_planning.py)、[tests/test_model_free_planning.py](tests/test_model_free_planning.py) |

## 状态价值、动作价值与回报

折扣回报表示从时刻 $t$ 开始，未来奖励的折扣和：

$$G_t=\sum_{k=0}^{T-t-1}\gamma^k R_{t+k+1}$$

其中 $\gamma$ 为 discount factor。状态价值与动作价值分别为：

$$V^\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s]$$

$$Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a]$$

Tabular 方法用表格保存每个状态或状态—动作对的估计。对某个表项赋值，不会直接改变其他表项；函数近似则可能通过共享参数影响多个预测。

## GridWorld 的固定 MDP 约定

```text
             col
          0     1     2
row 0    S0    S1    +10
row 1    S2   WALL   -10
row 2    S3    S4    S5
```

- 状态为 `(row, col)`；墙 `(1, 1)` 不属于状态集合。共 8 个状态，包含两个 terminal states。
- 动作顺序为 `UP, DOWN, LEFT, RIGHT`，位移分别是 `(-1,0), (1,0), (0,-1), (0,1)`。
- 当前为 deterministic MDP；越界或撞墙时原地不动。
- 进入 `(0, 2)` 获得 `+10.0`，进入 `(1, 2)` 获得 `-10.0`，其他转移奖励为 `-1.0`。终点奖励替代普通移动奖励，不再额外扣 1。
- 默认 `gamma=0.9`。终点未来价值固定为 0，奖励在进入终点的转移上领取，不能重复计算。
- `transitions(state, action)` 返回 `[(probability, next_state, reward)]`。已在终点时，合法动作返回 `[]`；非法状态或动作报错。

三个手算转移已包含在本章测试中：

| state | action | 输出 |
| --- | --- | --- |
| `(1, 0)` | `DOWN` | `[(1.0, (2, 0), -1.0)]` |
| `(2, 2)` | `UP` | `[(1.0, (1, 2), -10.0)]` |
| `(1, 0)` | `RIGHT` | `[(1.0, (1, 0), -1.0)]` |

## 3A：通过模型进行 Planning

### Bellman backup

给定旧 value table $V_k$，先对一个动作计算一步奖励与下一状态价值的期望：

$$Q_k(s,a)=\sum_{s'}T(s,a,s')[R(s,a,s')+\gamma V_k(s')]$$

`q_from_v` 通过 `env.transitions(state, action)` 获得概率、下一状态和奖励，对应代码：

```python
q += prob * (reward + env.gamma * V[next_state])
```

这里 $V_k$ 是读取的旧表，$Q_k$ 是本次 backup 的结果。进入终点时，$V_k(s')=0$，因此只留下进入终点的奖励。

### Synchronous Policy Evaluation

固定策略 $\pi$，按动作概率对 $Q_k$ 加权：

$$V_{k+1}(s)=\sum_a\pi(a\mid s)Q_k(s,a)$$

每轮都读取同一份旧 `V`，写入 `new_V`，整轮完成后再替换。不能在状态循环内覆盖旧表，否则会变成原地更新。

| 函数 | policy 格式 | 更新方式 |
| --- | --- | --- |
| `policy_evaluation` | `policy(env, state)` 返回 `{action: probability}` | 按动作概率加权 |
| `deterministic_policy_evaluation` | `{state: action}` | 直接取 $Q_k(s,\pi(s))$ |

`random_policy` 在非终点为每个动作分配 `0.25`，终点返回空字典。评估以 $\max_s\lvert V_{k+1}(s)-V_k(s)\rvert<\text{theta}$ 为停止条件，并受 `max_iterations` 限制；达到迭代上限的返回值不能自动称为已收敛。单轮 sweep 已由完整循环覆盖，不重复保存等价过渡代码。

### Policy Improvement 与 Policy Iteration

策略改进使用评估后的价值表选择最大 Q 值动作：

$$\pi_{\mathrm{new}}(s)\in\arg\max_a\sum_{s'}T(s,a,s')[R(s,a,s')+\gamma V^\pi(s')]$$

`policy_improvement` 返回新策略和 `policy_stable`，不修改传入的策略字典。当前 `greedy_action_from_v` 在并列时保留动作顺序中最先遇到的动作。

Policy Iteration 重复以下过程，直到策略稳定：

```text
固定当前 policy → Policy Evaluation → Policy Improvement → 检查 policy 是否变化
```

### Value Iteration

Value Iteration 每轮直接使用 Bellman optimality backup：

$$V_{k+1}(s)=\max_a\sum_{s'}T(s,a,s')[R(s,a,s')+\gamma V_k(s')]$$

`value_iteration` 同样采用 synchronous backup；得到 value table 后，`extract_policy` 再提取贪心策略。与 PI 相比，VI 的每轮更新不需要先把某个固定策略评估到停止条件。

文件末尾的 Step 8 在同一张地图、同一个 `gamma=0.9` 下比较 PI 和 VI，打印两张策略网格及 `Same policy: True`。当前六个非终点动作一致；一般情况下，存在多个最优动作时，策略字典是否相等也取决于并列选择规则。

### 保留的 V-table 练习

以下为已有练习值，终点未来价值仍为 0：

```text
V(0,0)=2   V(0,1)=5   V(0,2)=0
V(1,0)=1               V(1,2)=0
V(2,0)=0   V(2,1)=1   V(2,2)=2
```

取 `gamma=0.9`，练习计算 `Q((0,0), UP)`、`Q((0,1), RIGHT)`、`Q((2,2), UP)`。程序末尾还演示 `Q((0,0), RIGHT)`。保留题目供回顾，不据此判断用户是否已掌握。

## 3B：通过采样进行 Model-Free Learning

Model-free 算法使用采样经验更新，不能调用 `transitions()` 偷看模型期望。当前 MC、TD(0) 接收已有数据；SARSA、Q-learning 从环境采样。

### First-Visit Monte Carlo Prediction

MC 用完整 episode 的实际回报作为 target。先从后向前计算：

$$G_t=R_{t+1}+\gamma G_{t+1}$$

再按时间顺序，只更新每个状态在该 episode 中的首次出现：

$$V_{\mathrm{new}}(S_t)=V_{\mathrm{old}}(S_t)+\alpha[G_t-V_{\mathrm{old}}(S_t)]$$

`mc_prediction` 的数据是按时间排列的 `(state, reward)` 列表，其中 reward 是离开该 state 的当步奖励。当前实现使用固定 `alpha` 的增量更新，并非按访问次数求样本均值。

### TD(0)、SARSA 与 Q-learning

三者统一写成“旧值 + 学习率 × TD error”：

$$\delta_t=y_t-\text{old estimate}$$

$$\text{new estimate}=\text{old estimate}+\alpha\delta_t$$

非终止样本的 target 与被更新值如下：

| 算法 | target $y_t$ | 旧值 |
| --- | --- | --- |
| TD(0) | $R_{t+1}+\gamma V(S_{t+1})$ | $V(S_t)$ |
| SARSA | $R_{t+1}+\gamma Q(S_{t+1},A_{t+1})$ | $Q(S_t,A_t)$ |
| Q-learning | $R_{t+1}+\gamma\max_a Q(S_{t+1},a)$ | $Q(S_t,A_t)$ |

当 `done=True` 时，这三种算法的 target 都只取 $R_{t+1}$。TD(0) 的输入为 `(state, reward, next_state, done)` 序列。

SARSA 的 $A_{t+1}$ 是实际按当前策略选中的下一动作，因此是 on-policy 更新。Q-learning 的行为仍可采用 ε-greedy，但 target 使用最大 Q 值动作，因此是 off-policy 更新。

### ε-greedy

`choose_action` 以概率 $\epsilon$ 在全部动作中均匀探索，其余时间在贪心动作集合 $\mathcal{G}(s)$ 中均匀选择：

$$\pi(a\mid s)=\frac{\epsilon}{\lvert\mathcal{A}\rvert}+\begin{cases}\frac{1-\epsilon}{\lvert\mathcal{G}(s)\rvert},&a\in\mathcal{G}(s)\\0,&a\notin\mathcal{G}(s)\end{cases}$$

这里的并列处理与 3A 取第一个贪心动作的规则不同。调用者传入非空动作序列。

### 两类环境接口

| 使用位置 | 数据或接口 |
| --- | --- |
| DP / Planning | `transitions(state, action)` 返回模型转移 |
| MC Prediction | 完整的 `(state, reward)` episode |
| TD(0) Prediction | `(state, reward, next_state, done)` 序列 |
| SARSA / Q-learning | `reset()` 返回起始状态；`step(action)` 返回 `(next_state, reward, done)` |

当前 GridWorld 只实现 DP 的模型接口。控制算法的测试用 `TwoStepEnv` 验证采样和更新规则，未将 GridWorld 自动改造成采样环境。

## 3C：Linear Function Approximation

线性模型用特征向量 $\phi(s,a)$ 和共享参数 $\theta$ 表示 Q 值：

$$Q(s,a;\theta)=\theta^\top\phi(s,a)$$

把本次 target 视为常量，semi-gradient 更新为：

$$\delta=y-Q(s,a;\theta_{\mathrm{old}})$$

$$\theta_{\mathrm{new}}=\theta_{\mathrm{old}}+\alpha\delta\phi(s,a)$$

当前示例直接给定 `target=5.0`，使用 NumPy 的 `np.dot` 和向量运算；原标准库手写循环以注释保留。

| 项目 | 数值 |
| --- | --- |
| 旧参数 $\theta$ | `[2.0, -1.0]` |
| features | `[2.0, 1.0]` |
| 学习率 $\alpha$ | `0.1` |
| 旧预测、target、TD error | `3.0`、`5.0`、`2.0` |
| 新参数 | `[2.4, -0.8]` |
| 更新后预测 | `4.0` |

更新的是参数，新的 Q 值需要用新参数重新计算。它并不等于直接在原 Q 值上加 $\alpha\delta$；变化还取决于特征。

## 运行与测试

在仓库根目录运行，沿用已有 `.venv`。3A / 3B 只使用标准库，3C 使用已安装的 NumPy：

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m rl_tabular.model_based_planning
.\.venv\Scripts\python.exe -X utf8 -B -m rl_tabular.linear_approximation_demo
.\.venv\Scripts\python.exe -X utf8 -B -m unittest discover -s rl_tabular/tests -v
```

`-X utf8` 用于输出策略箭头。3B 模块提供函数，具体调用可参考手算测试；直接运行空的模块入口不代表完成了算法验证。

现有 10 项测试包括：GridWorld 转移、非法输入、terminal reward、同步更新、Bellman 固定点、策略评估与改进、PI / VI 对比，以及 MC / TD(0) 和 SARSA / Q-learning 的手算轨迹。学习示例通过打印结果对照手算，不添加数值接近断言。

## 来源

知识与练习沿用已读取的 [RL-3 分享快照](https://chatgpt.com/share/6ae896c1-274c-83ec-bc91-1c8d8a5fcbe9)。文档按当前实际实现整理；学习安排和历史验证结果见 [TASK](../TASK.md)。
