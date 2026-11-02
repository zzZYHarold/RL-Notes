# ReinforcementLearning

用于学习强化学习代码实操。目前进行 **RL-3：Tabular RL Lab**，处于 **3A Planning / DP**。

先搭建框架，随着学习逐步填入代码。当前所有 `.py` 文件都是空占位文件，运行后没有输出，不代表算法已实现。

## 目录

```text
ReinforcementLearning/
├── AGENTS.md                 # 给代码助手的协作约定
├── TASK.md                   # 当前进度、下一步、验证记录
├── docs/
│   └── rl3_plan.md           # RL-3 路线、环境约定、公式与代码的对应
├── rl_tabular/               # 自己实现的 Tabular RL 代码
│   ├── envs/                 # GridWorld、Random Walk、CliffWalking
│   ├── planning/             # 3A：Bellman backup、PE、VI、PI
│   ├── prediction/           # 3B：MC Prediction、TD(0)
│   ├── control/              # 3B：SARSA、Q-learning
│   └── experiments/          # 3C：三个算法对比实验
├── cs188/                    # 课程配套练习，和自写代码分开
│   ├── value_iteration/
│   └── q_learning/
├── tests/                    # 实现时再添加对应检查
├── notes/                    # 自己的推导、疑问和复盘
└── results/                  # 后续实验的表格、图、运行记录
```

每个 Python 包中的 `__init__.py` 也是空文件，仅用于包组织。

## 从哪里开始

1. 阅读 [当前任务](TASK.md)，确定这次只推进哪一步。
2. 对照 [RL-3 学习路线](docs/rl3_plan.md)，在对应文件里实现。
3. 用手算例子和小测试验证，再记录进度。

当前首先填入的文件是 `rl_tabular/envs/gridworld.py`。会话中已经讨论过环境与 `transitions()`，但这些代码尚未落入本项目。

已有 `.venv` 和 PyCharm 配置继续使用。框架阶段无需安装依赖；需要 NumPy、绘图等库时，再随具体任务添加依赖清单和运行命令。

来源：[RL-3：Tabular RL Lab 分享会话](https://chatgpt.com/share/6ae896c1-274c-83ec-bc91-1c8d8a5fcbe9)，于 2026-11-02 读取。该链接是分享时的快照。
