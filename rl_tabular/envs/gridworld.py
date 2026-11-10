class GridWorld:
    ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")

    ACTION_DELTAS = {
        "UP":    (-1, 0),
        "DOWN":  (1, 0),
        "LEFT":  (0, -1),
        "RIGHT": (0, 1),
    }

    def __init__(self, gamma=0.9):
        self.rows = 3
        self.cols = 3

        self.wall = (1, 1)

        self.terminal_rewards = {
            (0, 2): 10.0,
            (1, 2): -10.0,
        }

        self.gamma = gamma

        self.states = [
            (r, c)
            for r in range(self.rows) for c in range(self.cols)
            if (r, c) != self.wall
        ]

    def is_terminal(self, state):
        return state in self.terminal_rewards

    def _in_bounds(self, state):
        r, c = state
        return 0 <= r < self.rows and 0 <= c < self.cols

    def transitions(self, state, action):
        if state not in self.states:
            raise ValueError(f"Invalid state: {state}")

        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action: {action}")

        # terminal state 已经结束，不再发生 transition
        if self.is_terminal(state):
            return []

        dr, dc = self.ACTION_DELTAS[action]
        candidate = (state[0] + dr, state[1] + dc)

        # 越界或者撞墙：原地不动
        if not self._in_bounds(candidate) or candidate == self.wall:
            next_state = state
        else:
            next_state = candidate

        # 进入 terminal 时得到 terminal reward
        if next_state in self.terminal_rewards:
            reward = self.terminal_rewards[next_state]
        else:
            reward = -1.0

        return [(1.0, next_state, reward)]
