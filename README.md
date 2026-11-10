# ReinforcementLearning

用于学习强化学习代码实操。目前已整理 **RL-3A Planning / DP** 和 **RL-3B Model-Free Implementation** 的本地实现。

采用 **先集中写，之后按需整理** 的学习方式。目前按环境、model-based 和 model-free 组织已有代码。

## 目录

```text
ReinforcementLearning/
├── AGENTS.md                 # 给代码助手的协作约定
├── TASK.md                   # 当前进度、下一步、验证记录
├── docs/
│   └── rl3_plan.md           # 学习路线和已约定的环境规则
├── rl_tabular/
│   ├── envs/
│   │   └── gridworld.py      # GridWorld 的状态、动作、转移与奖励
│   ├── model_based_planning.py  # 3A 的策略评估、PI / VI 与 Step 8 对比
│   └── model_free_planning.py   # 3B 的 MC、TD(0)、SARSA 和 Q-learning
└── tests/
    ├── test_planning.py            # 3A 检查
    └── test_model_free_planning.py # 3B 手算轨迹检查
```

`.venv`、`.idea` 和根目录的隐藏配置文件继续保留。

## 从哪里开始

1. 阅读 [当前任务](TASK.md)，确定这次只推进哪一步。
2. 运行 [3A 模块](rl_tabular/model_based_planning.py) 的 Step 8，比较 PI 和 VI 求得的策略。
3. 对照 [3B 模块](rl_tabular/model_free_planning.py) 的函数说明和[手算测试](tests/test_model_free_planning.py)，检查每种算法的更新目标。

`model_based_planning.py` 包含两种策略评估、策略改进、Policy Iteration 和 Value Iteration。Step 8 用 `print_policy()` 在同一个 GridWorld 上显示两种算法的动作、墙与终点奖励，并比较策略字典。

两种策略评估的输入不同：`policy_evaluation(env, policy)` 接收返回动作概率字典的函数；`deterministic_policy_evaluation(env, policy)` 接收 `{state: action}` 字典，可与 `policy_improvement` 配合。

示例 `V_table` 和 Bellman backup 的演示位于 `model_based_planning.py` 末尾。3B 的 MC 和 TD(0) 读取已采样的数据，SARSA 和 Q-learning 通过 `reset()` 和 `step(action)` 采样。当前 GridWorld 提供 `transitions()`，3B 的控制测试使用两步测试环境。

在项目根目录运行示例与测试：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m rl_tabular.model_based_planning
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

示例先演示 Bellman backup、策略评估和一次策略改进，最后打印 Step 8 的两张策略网格和 `Same policy: True`。`-X utf8` 让箭头以 UTF-8 输出。

等你明确说“帮我整理代码”时，助手再读取实际内容，决定哪些部分值得分开，并同步调用、验证和说明。不会仅因学到新算法就自动拆文件。

重复或已被完整实现覆盖的过渡片段，说明后可以不保存。之前的单轮 sweep 已包含在 `policy_evaluation` 中，没有重复添加。

`cs188`（课程练习）、`notes`（学习笔记）、`results`（实验输出）在有实际内容时再建；本次 `tests` 中已有实际检查，不放空占位文件。

已有 `.venv` 和 PyCharm 配置继续使用。需要 NumPy、绘图等库时，再随具体任务添加依赖清单和运行命令。

来源：[RL-3：Tabular RL Lab 分享会话](https://chatgpt.com/share/6ae896c1-274c-83ec-bc91-1c8d8a5fcbe9)，于 2026-11-02 读取。该链接是分享时的快照。
