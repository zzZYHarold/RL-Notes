# RL-3：Tabular RL Lab 学习路线

目标：把已经学过的 tabular RL 公式落实为能解释、能验证的代码，再用小实验比较算法。以下沿用[分享会话](https://chatgpt.com/share/6ae896c1-274c-83ec-bc91-1c8d8a5fcbe9)与用户截图中的路线。

## 文件与阶段

| 阶段 | 内容 | 文件或位置 |
| --- | --- | --- |
| 3A | GridWorld MDP | `rl_tabular/envs/gridworld.py` |
| 3A | 从 V 计算 Q / Bellman backup | `rl_tabular/planning/bellman.py` |
| 3A | Policy Evaluation | `rl_tabular/planning/policy_evaluation.py` |
| 3A | Value Iteration | `rl_tabular/planning/value_iteration.py` |
| 3A | Policy Iteration | `rl_tabular/planning/policy_iteration.py` |
| 3A | CS188 Value Iteration | `cs188/value_iteration/` |
| 3B | MC Prediction | `rl_tabular/prediction/mc_prediction.py` |
| 3B | TD(0) Prediction | `rl_tabular/prediction/td0_prediction.py` |
| 3B | SARSA | `rl_tabular/control/sarsa.py` |
| 3B | Q-learning | `rl_tabular/control/q_learning.py` |
| 3B | CS188 Q-learning | `cs188/q_learning/` |
| 3C | VI vs Q-learning | `rl_tabular/experiments/compare_vi_q_learning.py` |
| 3C | MC vs TD(0) | `rl_tabular/experiments/compare_mc_td.py` |
| 3C | SARSA vs Q-learning | `rl_tabular/experiments/compare_sarsa_q_learning.py` |

`random_walk.py` 与 `cliff_walking.py` 也已在 `envs/` 中预留，目前不实现。3B 先沿用小 GridWorld，3C 再分别使用 GridWorld、Random Walk、CliffWalking。各实验的参数、预算和验收指标到对应小节再确定。

## 当前 GridWorld 的固定约定

```text
             col
          0     1     2
row 0    S0    S1    +10
row 1    S2   WALL   -10
row 2    S3    S4    S5
```

- 状态用 `(row, col)` 表示；`(1, 1)` 是墙，不属于状态集合。总计 8 个状态，包含两个 terminal states。
- 动作顺序：`UP, DOWN, LEFT, RIGHT`，分别对应 `(-1,0), (1,0), (0,-1), (0,1)`。
- 初始版本为 deterministic MDP，不加随机偏移。
- 越界或撞墙：原地不动。
- 进入 `(0, 2)` 获得 `+10.0`，进入 `(1, 2)` 获得 `-10.0`；其他转移奖励为 `-1.0`。终点奖励替代普通移动奖励，不再额外扣 1。
- `gamma = 0.9`。
- `transitions(state, action)` 返回 `[(probability, next_state, reward)]`。
- 已处于终点时，合法动作对应 `transitions(...) == []`；终点未来价值固定为 `0`。
- 非法状态或非法动作应报错；不要把非法输入当成撞墙。

会话中已完成的三个手算转移：

| 状态 | 动作 | 输出 |
| --- | --- | --- |
| `(1, 0)` | `DOWN` | `[(1.0, (2, 0), -1.0)]` |
| `(2, 2)` | `UP` | `[(1.0, (1, 2), -10.0)]` |
| `(1, 0)` | `RIGHT` | `[(1.0, (1, 0), -1.0)]` |

这些是后续实现的核对依据，尚未在本地执行验证。

## 接口逐步增加

当前先实现 `GridWorld`、状态集合、动作集合、`is_terminal(state)` 和 `transitions(state, action)`。

推进到采样交互时，再增加会话中建议的 `reset()`、`step(action)`、`get_states()`、`get_actions(state)`。返回值、起点、随机种子及终止后调用行为届时明确；当前不提前定义一套框架。

DP 能读取转移模型，MC、TD、SARSA、Q-learning 通过采样交互获得经验。统一环境概念不意味着 model-free 算法可以直接查询模型。

## 3A 的核心检查

Bellman backup 的公式为：

$$Q(s,a)=\sum_{s'}T(s,a,s')[R(s,a,s')+\gamma V(s')].$$

实现时逐项对应概率、下一状态、奖励和下一状态价值。进入终点的奖励只计算一次。Synchronous update 每一轮读取旧表、写入新表，完成整轮后再替换。

分享快照停在三个 Q 值练习题，使用以下 V-table 和 `gamma=0.9`：

```text
V(0,0)=2   V(0,1)=5   V(0,2)=0
V(1,0)=1               V(1,2)=0
V(2,0)=0   V(2,1)=1   V(2,2)=2
```

待回答：`Q((0,0), UP)`、`Q((0,1), RIGHT)`、`Q((2,2), UP)`。这里保留题目，不提前填写答案。

## 每个小节如何收尾

能解释核心更新语句对应的公式；能用至少一个手算例子核对输出；能运行相应检查并说明终点与边界处理；在 `TASK.md` 中分别记录实现与验证状态。

CS188 的具体课程版本、下载来源和运行要求，留到配套练习阶段再核实。
