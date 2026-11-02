# 当前任务与进度

更新日期：2026-11-02

## 当前状态

- 模块：RL-3 Tabular RL Lab。
- 阶段：3A Planning / DP。
- 本地：完成目录与空文件框架，尚无算法实现。
- 会话：已经讨论 GridWorld 表示和 `transitions()`；用户完成了三个转移手算题。最新教学内容为 `q_from_v` 和三个 Q 值手算题，尚未在分享快照中看到用户回答。
- 不将会话中的代码示例视为本地已实现，也不据此判定已掌握完整算法。

## 本次框架搭建

- [x] 读取分享会话并记录 RL-3 的约定。
- [x] 创建阶段目录、空 Python 文件和配套练习位置。
- [x] 写入 README、AGENTS 和学习路线。
- [x] 核对目录、空文件与文档链接。

## 下一次从这里继续

1. 将会话里已讨论的 GridWorld 表示与 `transitions()` 逐步落实到 `rl_tabular/envs/gridworld.py`。
2. 在 `tests/` 中补对应手算检查，验证墙、边界、终点、非法输入和转移输出。
3. 在 `rl_tabular/planning/bellman.py` 中实现 `q_from_v`，验证进入终点时不会重复计算奖励。
4. 再进入 synchronous Policy Evaluation，每次只推进一个小节。

这些是后续学习任务，本次不填入实现。

## 路线检查表

以下只记录本地实现及其验证；学习理解情况另行确认。

- [ ] 3A：GridWorld MDP
- [ ] 3A：Bellman backup / 从 V 计算 Q
- [ ] 3A：Policy Evaluation
- [ ] 3A：Value Iteration
- [ ] 3A：Policy Iteration
- [ ] 3A：CS188 Value Iteration
- [ ] 3B：MC Prediction
- [ ] 3B：TD(0) Prediction
- [ ] 3B：SARSA
- [ ] 3B：Q-learning
- [ ] 3B：CS188 Q-learning
- [ ] 3C：GridWorld，VI vs Q-learning
- [ ] 3C：Random Walk，MC vs TD(0)
- [ ] 3C：CliffWalking，SARSA vs Q-learning

## 验证记录

2026-11-02 框架检查通过：共 32 个项目文件（不含原有 `.idea/`、`.venv/`），其中 20 个 Python 文件均为 0 字节，5 个 `.gitkeep` 保留预留目录；已检查 UTF-8 编码、Markdown 本地链接及路线中的文件路径。已有 `.venv` 和 `.idea` 仍在原位置。

当前无可运行的算法测试；未安装依赖、未运行训练或实验。空文件检查只证明框架状态，不证明算法正确。

## 上下文来源

原会话 ID：`6ae71731-78e8-83ee-9dae-6d86a0856d5f`。

已读取的[分享快照](https://chatgpt.com/share/6ae896c1-274c-83ec-bc91-1c8d8a5fcbe9)。本项目的建立范围由用户当前请求确定；快照用于恢复学习上下文。
