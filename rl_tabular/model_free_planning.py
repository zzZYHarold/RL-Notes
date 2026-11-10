from collections import defaultdict
import random


def mc_prediction(episodes, gamma=0.9, alpha=0.1):
    """用完整 episode 做 First-Visit MC。每条轨迹按时间顺序存放 (state, reward)。"""
    V = defaultdict(float)

    for episode in episodes:
        returns = []
        G = 0.0

        # 先从后往前计算每个时刻的 return
        for state, reward in reversed(episode):
            G = reward + gamma * G
            returns.append((state, G))

        # 恢复正常时间顺序
        returns.reverse()

        # First-Visit MC
        visited = set()
        for state, G in returns:
            if state in visited:
                continue

            visited.add(state)
            V[state] += alpha * (G - V[state])

    return V


def td0_prediction(transitions, gamma=0.9, alpha=0.1):
    """按顺序读取 (state, reward, next_state, done)。done 时 target 只取 reward。"""
    V = defaultdict(float)

    for state, reward, next_state, done in transitions:
        if done:
            target = reward
        else:
            target = reward + gamma * V[next_state]

        td_error = target - V[state]
        V[state] += alpha * td_error

    return V


def choose_action(Q, state, actions, epsilon=0.1):
    """从非空动作序列中做 ε-greedy 选择，并列的贪心动作等概率抽取。"""
    if random.random() < epsilon:
        return random.choice(actions)

    best_value = max(Q[state][a] for a in actions)
    best_actions = [a for a in actions if Q[state][a] == best_value]

    return random.choice(best_actions)


def sarsa(env, actions, episodes, gamma=0.9, alpha=0.1, epsilon=0.1):
    """reset() 返回起始状态，step(action) 返回 (next_state, reward, done)。

    非终点 target 使用实际选中的 next_action。
    """
    Q = defaultdict(lambda: defaultdict(float))

    for _ in range(episodes):
        state = env.reset()
        action = choose_action(Q, state, actions, epsilon)

        done = False
        while not done:
            next_state, reward, done = env.step(action)

            if done:
                target = reward
            else:
                next_action = choose_action(Q, next_state, actions, epsilon)
                target = reward + gamma * Q[next_state][next_action]

            td_error = target - Q[state][action]
            Q[state][action] += alpha * td_error

            if not done:
                state = next_state
                action = next_action

    return Q


def q_learning(env, actions, episodes, gamma=0.9, alpha=0.1, epsilon=0.1):
    """reset() 返回起始状态，step(action) 返回 (next_state, reward, done)。

    非终点 target 使用下一状态的最大 Q 值。
    """
    Q = defaultdict(lambda: defaultdict(float))

    for _ in range(episodes):
        state = env.reset()

        done = False
        while not done:
            # behavior policy: ε-greedy
            action = choose_action(Q, state, actions, epsilon)

            next_state, reward, done = env.step(action)

            if done:
                target = reward
            else:
                best_next_value = max(Q[next_state][a] for a in actions)
                target = reward + gamma * best_next_value

            td_error = target - Q[state][action]
            Q[state][action] += alpha * td_error
            state = next_state

    return Q
