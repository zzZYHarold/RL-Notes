# 当前任务与进度

更新日期：2026-11-15

## 当前状态

- 当前模块为 RL-4 Deep RL，4B 聊天内容已整理成本地固定样本演示；聊天中已作答的练习与代码验证分别记录，不据此自动判定用户掌握。
- `rl_deep/q_network_demo.py` 保留 4A 单条 transition 更新，追加 Moving Target、三条样本的 batch 手算、Replay + Target Network 的四次梯度更新，以及独立的 Soft Update。使用固定权重线性网络和 SGD；按用户要求删除数值接近断言，保留计算与打印输出。
- 已按用户授权在项目现有 `.venv` 安装 NumPy 2.5.4 和 PyTorch 2.15.0+cpu 及必要依赖，解释器为 Python 3.14.2。
- 3A 代码位于 `rl_tabular/model_based_planning.py`。Step 8 在同一张 GridWorld 上比较 PI 和 VI。
- 3B 代码位于 `rl_tabular/model_free_planning.py`，已有 First-Visit MC、TD(0)、ε-greedy、SARSA 和 Q-learning，包含数据格式、采样接口说明和手算测试。
- 环境 `rl_tabular/envs/gridworld.py` 提供 DP 用的 `transitions()`。控制算法通过两步测试环境验证 `reset()` 和 `step(action)` 的交互。
- 3C 代码位于 `rl_tabular/linear_approximation_demo.py`。按用户本轮授权改用 NumPy 的点积和向量更新，原标准库手写循环与自检保留为注释，数值行为不变。
- 知识、公式、环境规则和运行方法分别见 [RL-3 README](rl_tabular/README.md) 与 [RL-4 README](rl_deep/README.md)。测试归入 `rl_tabular/tests/`；学习计划只在本文件维护。

## 完整学习路线

| 阶段 | 学习顺序 | 本地状态 |
| --- | --- | --- |
| 3A Planning / DP | GridWorld → Bellman backup → synchronous Policy Evaluation → Value Iteration → Policy Iteration → CS188 Value Iteration | 自写算法与 Step 8 已实现、验证；CS188 完成情况待核对 |
| 3B Model-Free RL | MC Prediction → TD(0) → ε-greedy → SARSA → Q-learning → CS188 Q-learning | 自写算法与手算轨迹测试已实现、验证；CS188 完成情况待核对 |
| 3C Function Approximation | Tabular 更新 → 线性表示 → 一次 semi-gradient 参数更新 | NumPy 示例已实现、运行；手写版本以注释保留 |
| 4A Q-network | 当前预测 → TD target → Loss → Backpropagation → SGD | 单条 transition 示例已实现、运行 |
| 4B DQN 稳定化机制 | 样本相关性与 Moving Target → Deadly Triad → Replay → Target Network → Hard / Soft Update → Batch Update | 聊天内容已整理，固定样本演示已运行；环境训练未实现 |
| 4C DQN 改进 | Double DQN → Dueling DQN → Prioritized Experience Replay | 分享聊天中提出的后续安排；本地未实现，按用户后续学习内容推进 |

路线表示学习次序，不自动认定用户已掌握，也不据此预建未来代码文件。

## 待安排事项

- 核对 CS188 Value Iteration 与 Q-learning 的课程版本、来源和完成情况；到对应阶段再安排，保留课程原接口。
- GridWorld：VI 与 Q-learning 对比；先明确采样接口，不能让 model-free 算法读取转移模型。
- Random Walk：MC 与 TD(0) 对比。
- CliffWalking：SARSA 与 Q-learning 对比。
- Deep RL 环境交互与朴素 DQN / Replay / Target 对比，等待对应学习任务，不把固定样本演示视为性能实验。

各实验的参数、预算和验收指标到对应任务再确定；实际运行时记录环境、随机种子、参数、命令和指标，不预先保证实验现象。

## 当前工作流

- 学习时先在一个文件里继续写，不预建后续算法文件或空目录。
- 按当前学习内容在已有模块中追加代码，不要求每学一个算法就新建文件。
- 用户明确要求整理时，助手先读实际代码，再按内容决定最小必要拆分，并同步调用与验证。
- 重复、等价或被完整实现覆盖的过渡片段，说明后可以省略，不逐段机械留存。
- 根 README 负责章节导航；每个有实际内容的学习章节维护自己的 README，解释知识、公式、代码对应和运行方法。环境和测试子目录不另设 README；计划、待办与验证历史集中在 TASK。

## 下一次从这里继续

4A / 4B 固定样本演示已运行验证，可对照输出阅读 `gather`、终止 target、batch loss 和目标网络同步。4B 分享聊天中用户表示准备在新会话开始 4C；本地尚未实现 Double DQN、Dueling DQN 或 Prioritized Replay，后续按用户提供的学习内容推进。

当前示例未接入环境交互，未做朴素 DQN 与 Replay / Target 的训练对比。原计划中的 GridWorld、Random Walk 和 CliffWalking 对比实验保留为待安排项。路线里的 CS188 Q-learning 完成情况尚未核对。

每个小节收尾时，对照公式与手算说明核心更新、终止和边界处理，并在本文件分别记录实现与验证状态；不自动勾选用户掌握。完整学习顺序见上面的路线表。

## 验证记录

2026-11-15 按章节整理文档与测试：根 README 改为导航，新增 RL-3 / RL-4 的章节 README；原独立路线文档的计划合并到本文件，环境规则、知识说明和练习迁入章节 README，删除原文档目录。两份测试原样移入 `rl_tabular/tests/`，同步 AGENTS 与运行说明。

- 迁移前从旧入口运行 10 项测试通过；迁移后用 `.\.venv\Scripts\python.exe -X utf8 -B -m unittest discover -s rl_tabular/tests -v` 仍发现并通过相同的 10 项测试。
- SHA-256 对照确认五份环境 / 算法 / 演示源码内容不变，两份测试搬迁前后内容相同；未改动已有暂存内容。
- 根 README 中三个模块运行命令均执行成功：PI / VI 输出 `Same policy: True`，3C 输出 `[2.4, -0.8]` 与 `4.0`，4A / 4B 打印结果与原例子一致。
- 文档 UTF-8、LF、代码围栏、本地链接、旧文档引用清理和 `git diff --check` 检查通过；原文档目录和根 tests 目录均已移除。未新增算法、数值接近断言或依赖，未提交或推送。

以下保留原有历史记录，其中旧命令和旧路径代表当时的目录布局。当前运行入口以章节 README 为准。

2026-11-15 按用户要求删除两个学习示例中的全部 `assert_close` / `assert_allclose`，并清理仅供这些断言使用的权重快照和预期梯度代码。以后不再写此类断言，也不替换为等价断言。保留普通检查、手写学习注释和既有测试；以下整理阶段的数值自检属于删除前的历史记录。

删除后用已有 `.venv` 重跑两个示例，均正常退出，打印输出与删除前完全一致；Python 源码搜索确认无此类断言，`git diff --check` 通过。本轮未重跑既有单元测试。

2026-11-15 整理 RL-4B，复用已有依赖，没有安装或升级库。CPU、Python 3.14.2、NumPy 2.5.4、PyTorch 2.15.0+cpu；固定 seed=7、buffer capacity=3、batch size=3、gamma=0.9、SGD lr=0.1、Hard Update C=2。三条人工经验重复两遍，仅演示机制，不是环境性能实验。

- `.\.venv\Scripts\python.exe -X utf8 -B -m rl_deep.q_network_demo`：4A 原自检保留通过；Moving Target 为 `1.5 -> 1.575`（浮点近似），聊天练习 Batch Loss 约 `0.19`。Replay 前两次跳过训练，完成四次更新；打印的更新前 loss 依次为 `2.245417`、`2.082395`、`1.971439`、`1.836515`，第 2、4 次同步 Target。目标在同步时变化，不能把这些输出当成固定目标下的收敛证据。
- 文件内断言检查缓冲区容量与淘汰顺序、终止样本只取奖励、目标无梯度、首个 batch 的手算梯度及更新后权重、同步间隔内参数不变和同步后参数相等。独立 Soft Update 的 `4 -> 4.06` 检查通过。
- `.\.venv\Scripts\python.exe -X utf8 -B -m rl_tabular.linear_approximation_demo`：NumPy 版本输出参数 `[2.4, -0.8]`、新 Q 值 `4.0`，自检通过。
- `.\.venv\Scripts\python.exe -X utf8 -B -m unittest discover -s tests -v`：已有 10 项测试全部通过。4A / 4B 的检查在示例内执行，未计入这 10 项。
- 两个示例的语法、LF 行尾、导入无输出，以及文档本地链接和 `git diff --check` 检查通过。本轮未提交或推送。

2026-11-15 安装与运行验证通过。NumPy 来自 PyPI，PyTorch 来自 `https://download.pytorch.org/whl/cpu`，安装版本为 NumPy 2.5.4 和 torch 2.15.0+cpu。首次安装被沙盒临时目录权限阻止，提升执行权限后完成安装，目标为本项目 `.venv`。

- `python -B -m rl_deep.q_network_demo` 的数值和梯度断言全部通过。输出为预测值 `3.0`、TD target `3.25`、loss `0.03125`、更新后 Q 值 `[3.125, 2.0]`。
- NumPy 数组转 PyTorch Tensor 的检查通过，`torch.version.cuda` 为 `None`，确认使用 CPU 构建。
- `python -B -m pip check` 报告依赖完整，已有 10 项单元测试全部通过。

2026-11-13 的 4A 示例已写入，Python 语法编译、文档本地链接和 `git diff --check` 检查通过，原有 10 项测试全部通过。当前 `.venv` 为 Python 3.14.2，PyTorch 和 NumPy 均未安装，4A 的前向、反向和数值自检待安装依赖后验证。

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

以上按日期记录软件验证结果，理解掌握情况由用户学习进度确认。

## 上下文来源

原会话 ID：`6ae71731-78e8-83ee-9dae-6d86a0856d5f`。

已读取的[分享快照](https://chatgpt.com/share/6ae896c1-274c-83ec-bc91-1c8d8a5fcbe9)。本项目的建立范围由用户当前请求确定；快照用于恢复学习上下文。

RL-4B 来源：[分享快照](https://chatgpt.com/share/6af8876a-d500-83e8-88c3-1abeba195ae0)，于 2026-11-15 读取。用户审阅整理方案后授权实现，并允许使用已安装的 NumPy / PyTorch、将有用的手写版本保留为注释。重复的概念伪代码合并为一个可运行示例与简短数据流说明，未逐段重复保存。
