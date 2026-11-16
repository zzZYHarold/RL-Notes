# ReinforcementLearning

强化学习代码实操仓库，目前已整理到 **RL-4B：Experience Replay 与 Target Network**。采用“先在一个文件中学习和实现，再按需整理”的方式。

## 学习入口

| 入口 | 内容 |
| --- | --- |
| [TASK](TASK.md) | 完整学习路线、当前进度、待办与验证记录 |
| [RL-3：Tabular RL](rl_tabular/README.md) | GridWorld、DP、MC / TD、SARSA / Q-learning 与线性函数近似 |
| [RL-4：Deep RL](rl_deep/README.md) | Q-network、Moving Target、Replay、Target Network 与 batch 更新 |

章节 README 结合实际代码讲解重点知识、公式与运行方法。以后新增有实际学习内容的章节时，再创建对应目录及 README；环境和测试说明归入所属章节。学习计划统一维护在 TASK。

## 目录

```text
ReinforcementLearning/
├── README.md
├── TASK.md
├── AGENTS.md
├── rl_tabular/
│   ├── README.md
│   ├── envs/
│   │   └── gridworld.py
│   ├── model_based_planning.py
│   ├── model_free_planning.py
│   ├── linear_approximation_demo.py
│   └── tests/
│       ├── test_planning.py
│       └── test_model_free_planning.py
└── rl_deep/
    ├── README.md
    └── q_network_demo.py
```

## 运行

复用已有 `.venv`。当前环境为 Python 3.14.2、NumPy 2.5.4、PyTorch 2.15.0+cpu；`.idea` 和隐藏配置继续保留。以下命令均在仓库根目录运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m rl_tabular.model_based_planning
.\.venv\Scripts\python.exe -X utf8 -B -m rl_tabular.linear_approximation_demo
.\.venv\Scripts\python.exe -X utf8 -B -m rl_deep.q_network_demo
.\.venv\Scripts\python.exe -X utf8 -B -m unittest discover -s rl_tabular/tests -v
```

数值示例通过打印结果对照手算；现有测试集中在 tabular 章节。具体输出和实现边界见各章节 README，实际验证结果见 TASK。
