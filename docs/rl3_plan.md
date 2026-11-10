# RL-3：Tabular RL Lab 学习路线

目标：把已经学过的 tabular RL 公式落实为能解释、能验证的代码，再用小实验比较算法。以下沿用[分享会话](https://chatgpt.com/share/6ae896c1-274c-83ec-bc91-1c8d8a5fcbe9)与用户截图中的路线。

## 学习阶段（不预先分文件）

| 阶段 | 学习顺序 |
| --- | --- |
| 3A Planning / DP | GridWorld → Bellman backup → Policy Evaluation → Value Iteration → Policy Iteration → CS188 Value Iteration |
| 3B Model-Free RL | MC Prediction → TD(0) → SARSA → Q-learning → CS188 Q-learning |
| 3C Experiments | GridWorld：VI vs Q-learning；Random Walk：MC vs TD(0)；CliffWalking：SARSA vs Q-learning |

学习时可以先集中写，用户要求整理时再按实际内容拆分。环境放在 `rl_tabular/envs/gridworld.py`，DP 函数与 Step 8 对比放在 `rl_tabular/planning.py`。各实验的参数、预算和验收指标到对应小节再确定。

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

这些手算转移已纳入 `tests/test_planning.py` 的检查。

## 接口逐步增加

`gridworld.py` 只保留 `GridWorld` 的状态、动作、终止和转移规则。Bellman backup、随机策略、策略评估、贪心动作提取与策略改进集中在 `planning.py`；原来的练习 V-table 放在该文件末尾示例中。

`policy_evaluation` 用策略函数给出的动作概率计算加权期望；`deterministic_policy_evaluation` 根据 `policy[state]` 选择一个动作，直接计算 Q。两者都用旧 V 读取、向新表写入，终点价值保持 0。确定性版本按用户补充的 P2 片段补齐了初始化、收敛判断及返回值；P1 单轮 sweep 已被完整循环覆盖，不另外保存。

Step 8 用同一个 `gamma=0.9` 的 GridWorld 求解两次。Policy Iteration 先评估当前策略，再按各状态的 Q 值更新动作，直到策略稳定。Value Iteration 每轮用旧价值表计算各状态的最大 Q 值，最后从求得的价值表提取策略。两种方法在当前地图的六个非终点状态都选了相同动作。存在多个最优动作时，策略字典是否相等还取决于并列时的选择。

推进到采样交互时，再增加会话中建议的 `reset()`、`step(action)`、`get_states()`、`get_actions(state)`。返回值、起点、随机种子及终止后调用行为届时明确；当前不提前定义一套框架。

DP 能读取转移模型，MC、TD、SARSA、Q-learning 通过采样交互获得经验。统一环境概念不意味着 model-free 算法可以直接查询模型。

## 3A 的核心检查

Bellman backup 的公式为：

$$Q(s,a)=\sum_{s'}T(s,a,s')[R(s,a,s')+\gamma V(s')].$$

实现时逐项对应概率、下一状态、奖励和下一状态价值。进入终点的奖励只计算一次。Synchronous update 每一轮读取旧表、写入新表，完成整轮后再替换。

此前分享快照的 Q 值练习使用以下 V-table 和 `gamma=0.9`：

```text
V(0,0)=2   V(0,1)=5   V(0,2)=0
V(1,0)=1               V(1,2)=0
V(2,0)=0   V(2,1)=1   V(2,2)=2
```

练习题为：`Q((0,0), UP)`、`Q((0,1), RIGHT)`、`Q((2,2), UP)`。保留题目供回顾，不依据旧快照判断当前是否已回答。

## 每个小节如何收尾

能解释核心更新语句对应的公式；能用至少一个手算例子核对输出；能运行相应检查并说明终点与边界处理；在 `TASK.md` 中分别记录实现与验证状态。

CS188 的具体课程版本、下载来源和运行要求，留到配套练习阶段再核实。
