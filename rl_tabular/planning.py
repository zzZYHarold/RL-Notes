"""RL-3A 的 Bellman backup、Policy Evaluation、Policy Iteration 和 Value Iteration。"""


# RL-3A Step 4：Policy Evaluation 的第一次代码实现

def q_from_v(env, V, state, action):
    q = 0.0
    for prob, next_state, reward in env.transitions(state, action):
        q += prob * (reward + env.gamma * V[next_state])
    return q


def random_policy(env, state):
    if env.is_terminal(state):
        return {}

    prob = 1.0 / len(env.ACTIONS)
    return {action: prob for action in env.ACTIONS}


def state_value_from_v(env, V, policy, state):
    if env.is_terminal(state):
        return 0.0

    new_value = 0.0
    action_probs = policy(env, state)

    for action, action_prob in action_probs.items():
        q = q_from_v(env, V, state, action)
        new_value += action_prob * q

    return new_value


def policy_evaluation(env, policy, theta=1e-6, max_iterations=10_000):
    """policy(env, state) 返回动作到概率的字典。"""
    V = {state: 0.0 for state in env.states}

    for _ in range(max_iterations):
        new_V = {}
        delta = 0.0

        # 整轮读取旧 V，整轮结束后再替换，保持 synchronous backup。
        for state in env.states:
            new_value = state_value_from_v(env, V, policy, state)
            new_V[state] = new_value
            delta = max(delta, abs(new_value - V[state]))

        V = new_V
        if delta < theta:
            break

    return V


def deterministic_policy_evaluation(env, policy, theta=1e-6, max_iterations=10_000):
    """policy[state] 直接给出一个动作，对应 V(s) = Q(s, policy[s])。"""
    V = {state: 0.0 for state in env.states}

    for _ in range(max_iterations):
        new_V = {}
        delta = 0.0

        for state in env.states:
            if env.is_terminal(state):
                new_value = 0.0
            else:
                action = policy[state]
                new_value = q_from_v(env, V, state, action)

            new_V[state] = new_value
            delta = max(delta, abs(new_value - V[state]))

        V = new_V
        if delta < theta:
            break

    return V


# 下一步：Policy Improvement

def greedy_action_from_v(env, V, state):
    if env.is_terminal(state):
        return None

    best_action = None
    best_q = float("-inf")

    for action in env.ACTIONS:
        q = q_from_v(env, V, state, action)

        if q > best_q:
            best_q = q
            best_action = action

    return best_action


def policy_improvement(env, V, policy):
    new_policy = policy.copy()
    policy_stable = True

    for state in env.states:
        if env.is_terminal(state):
            continue

        old_action = policy[state]
        best_action = greedy_action_from_v(env, V, state)
        new_policy[state] = best_action

        if best_action != old_action:
            policy_stable = False

    return new_policy, policy_stable


# 进入 Value Iteration

def optimal_state_value(env, V, state):
    if env.is_terminal(state):
        return 0.0
    return max(q_from_v(env, V, state, action) for action in env.ACTIONS)

def value_iteration(env, theta=1e-6, max_iterations=10_000):
    V = {state: 0.0 for state in env.states }

    for _ in range(max_iterations):
        new_V = {}
        delta = 0.0

        for state in env.states:
            if env.is_terminal(state):
                new_value = 0.0
            else:
                new_value = optimal_state_value(env, V, state)

            new_V[state] = new_value
            delta = max(delta, abs(new_V[state] - V[state]))

        V = new_V

        if delta < theta:
            break

    return V

def extract_policy(env, V):
    policy = {}

    for state in env.states:
        if env.is_terminal(state):
            continue

        policy[state] = greedy_action_from_v(env, V, state)

    return policy


def policy_iteration(env):
    policy = {state: env.ACTIONS[0] for state in env.states if not env.is_terminal(state)}

    for _ in range(10_000):
        V = deterministic_policy_evaluation(env, policy)
        new_policy, policy_stable = policy_improvement(env, V, policy)
        if policy_stable:
            return V, policy
        policy = new_policy

    raise RuntimeError("Policy Iteration did not stabilize within 10000 iterations")


def print_policy(env, policy):
    symbols = {"UP": "↑", "DOWN": "↓", "LEFT": "←", "RIGHT": "→"}

    for r in range(env.rows):
        row = []
        for c in range(env.cols):
            state = (r, c)
            if state == env.wall:
                row.append("###")
            elif state in env.terminal_rewards:
                reward = env.terminal_rewards[state]
                row.append(f"{reward:+.0f}")
            else:
                row.append(symbols[policy[state]])

        print("  ".join(row))


if __name__ == "__main__":
    from rl_tabular.envs.gridworld import GridWorld

    env = GridWorld(gamma=0.9)
    V_table = {
        (0, 0): 2.0,
        (0, 1): 5.0,
        (0, 2): 0.0,

        (1, 0): 1.0,
        (1, 2): 0.0,

        (2, 0): 0.0,
        (2, 1): 1.0,
        (2, 2): 2.0,
    }
    print("Q((0, 0), RIGHT):", q_from_v(env, V_table, (0, 0), "RIGHT"))
    print("Random policy V:", policy_evaluation(env, random_policy))

    policy = {state: "UP" for state in env.states if not env.is_terminal(state)}
    V = deterministic_policy_evaluation(env, policy)
    new_policy, policy_stable = policy_improvement(env, V, policy)
    print("Deterministic UP policy V:", V)
    print("One policy improvement:", new_policy)
    print("Policy stable:", policy_stable)

    # RL-3A Step 8：在同一个 GridWorld 上比较 PI 和 VI。
    _, policy_pi = policy_iteration(env)
    V_vi = value_iteration(env)
    policy_vi = extract_policy(env, V_vi)

    print("\nPolicy Iteration:")
    print_policy(env, policy_pi)
    print("\nValue Iteration:")
    print_policy(env, policy_vi)
    print("\nSame policy:", policy_pi == policy_vi)
