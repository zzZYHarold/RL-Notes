# ReinforcementLearning

用于学习强化学习代码实操。目前进行 **RL-3：Tabular RL Lab**，处于 **3A Planning / DP**。

采用 **先在一个文件里写，之后按需整理** 的学习方式。你不用提前判断是否应该分文件。本轮按你的要求，把已有内容整理成环境与 DP 函数两份。

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
│   └── planning.py          # Bellman backup、策略评估、PI / VI 与 Step 8 对比
└── tests/
    └── test_planning.py     # 手算、同步更新、收敛和策略改进检查
```

`.venv`、`.idea` 和根目录的隐藏配置文件继续保留。

## 从哪里开始

1. 阅读 [当前任务](TASK.md)，确定这次只推进哪一步。
2. 在 [planning.py](rl_tabular/planning.py) 末尾运行 Step 8，对照 PI 和 VI 求得的策略。
3. 用手算例子和小测试验证，再记录进度。

`planning.py` 包含两种策略评估、策略改进、Policy Iteration 和 Value Iteration。Step 8 用 `print_policy()` 在同一个 GridWorld 上显示两种算法的动作、墙与终点奖励，并比较策略字典。

两种策略评估的输入不同：`policy_evaluation(env, policy)` 接收返回动作概率字典的函数；`deterministic_policy_evaluation(env, policy)` 接收 `{state: action}` 字典，可与 `policy_improvement` 配合。

辅助函数现在是普通函数，例如 `q_from_v(env, V, state, action)`，不再通过 `env.q_from_v(...)` 调用。原来的示例 `V_table` 移到了 `planning.py` 末尾的可运行示例中。

在项目根目录运行示例与测试：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m rl_tabular.planning
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

示例先演示 Bellman backup、策略评估和一次策略改进，最后打印 Step 8 的两张策略网格和 `Same policy: True`。`-X utf8` 让箭头以 UTF-8 输出。

等你明确说“帮我整理代码”时，助手再读取实际内容，决定哪些部分值得分开，并同步调用、验证和说明。不会仅因学到新算法就自动拆文件。

重复或已被完整实现覆盖的过渡片段，说明后可以不保存。之前的单轮 sweep 已包含在 `policy_evaluation` 中，没有重复添加。

`cs188`（课程练习）、`notes`（学习笔记）、`results`（实验输出）在有实际内容时再建；本次 `tests` 中已有实际检查，不放空占位文件。

已有 `.venv` 和 PyCharm 配置继续使用。需要 NumPy、绘图等库时，再随具体任务添加依赖清单和运行命令。

来源：[RL-3：Tabular RL Lab 分享会话](https://chatgpt.com/share/6ae896c1-274c-83ec-bc91-1c8d8a5fcbe9)，于 2026-11-02 读取。该链接是分享时的快照。
