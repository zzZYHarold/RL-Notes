# 当前任务与进度

更新日期：2026-11-12

## 当前状态

- 模块：RL-3 Tabular RL Lab。
- 阶段：3C Tabular 收尾 → Linear Function Approximation。
- 3A 代码位于 `rl_tabular/model_based_planning.py`。Step 8 在同一张 GridWorld 上比较 PI 和 VI。
- 3B 代码位于 `rl_tabular/model_free_planning.py`，已有 First-Visit MC、TD(0)、ε-greedy、SARSA 和 Q-learning。本轮为函数补充数据格式和采样接口说明，并增加手算测试。
- 环境 `rl_tabular/envs/gridworld.py` 提供 DP 用的 `transitions()`。控制算法通过两步测试环境验证 `reset()` 和 `step(action)` 的交互。
- 3C 代码位于 `rl_tabular/linear_approximation_demo.py`。用标准库演示一次线性 Q 值预测和 semi-gradient 参数更新，原 NumPy 代码完整保留为注释。

## 当前工作流

- 学习时先在一个文件里继续写，不预建后续算法文件或空目录。
- 按当前学习内容在已有模块中追加代码，不要求每学一个算法就新建文件。
- 用户明确要求整理时，助手先读实际代码，再按内容决定最小必要拆分，并同步调用与验证。
- 重复、等价或被完整实现覆盖的过渡片段，说明后可以省略，不逐段机械留存。

## 下一次从这里继续

当前 3C 记录用户提供的单次线性近似更新，后续按用户提供的学习内容推进。原计划中的 GridWorld、Random Walk 和 CliffWalking 对比实验保留为待安排项。路线里的 CS188 Q-learning 完成情况尚未核对。

完整学习顺序见 `docs/rl3_plan.md`，路线不再绑定预建文件。

## 验证记录

2026-11-12 用已有 `.venv` 运行 `python -B -m rl_tabular.linear_approximation_demo`，数值自检通过。旧 Q 值为 `3.0`，TD error 为 `2.0`，输出更新后参数 `[2.4, -0.8]` 和新 Q 值 `4.0`。已有 10 项单元测试、文档本地链接和 `git diff --check` 均通过。NumPy 注释版保留供阅读，本次执行的是标准库版本。

2026-11-10 整理 3B 后，用已有 `.venv` 运行 `python -B -m unittest discover -s tests -v`，10 项测试全部通过。新增的 3 项检查覆盖 First-Visit MC、TD(0) 的终点与非终点 target，以及同一两步环境下 SARSA 和 Q-learning 的不同 target。测试固定动作序列，验证范围为更新规则。

2026-11-10 用 `python -B -X utf8 -m rl_tabular.model_based_planning` 重跑 Step 8。PI 与 VI 在六个非终点状态的动作一致，程序输出 `Same policy: True`。

历史：首次初始化建立了预留骨架；用户使用后认为文件过多，现改为单文件学习、之后按需整理。

2026-11-02 清理：删除 24 个空占位文件、预留说明 `cs188/README.md`，并移除 10 个空目录；当时 `gridworld.py` 内容未改动。

2026-11-03 整理验证通过：

- 使用已有 `.venv` 运行 `python -B -m unittest discover -s tests -v`，6 项测试全部通过，覆盖手算转移、非法输入、terminal reward、单轮同步更新、随机策略 Bellman 固定点、确定性策略解析值及策略改进。
- 当时的 3A 模块示例运行成功，单独导入模块无输出。
- 对照整理前快照，环境部分 AST 未变，32 组状态—动作转移完全一致，原练习 V-table 数值保留。
- 文档本地链接、UTF-8、Python LF 行尾、无空 Python 占位文件及 `git diff --check` 检查通过。

新增确定性版本与现有随机版本在概率为 1 的同一策略上得到相同结果。保留原停止方式：达到 `theta` 或 `max_iterations` 即返回，不把任意迭代次数下的返回值都称为已收敛。

未新增依赖，未提交或推送改动。以上是软件检查，不自动判定用户已掌握。

## 上下文来源

原会话 ID：`6ae71731-78e8-83ee-9dae-6d86a0856d5f`。

已读取的[分享快照](https://chatgpt.com/share/6ae896c1-274c-83ec-bc91-1c8d8a5fcbe9)。本项目的建立范围由用户当前请求确定；快照用于恢复学习上下文。
