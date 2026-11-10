import unittest
from contextlib import redirect_stdout
from io import StringIO

from rl_tabular.envs.gridworld import GridWorld
from rl_tabular.model_based_planning import (
    deterministic_policy_evaluation,
    extract_policy,
    greedy_action_from_v,
    policy_evaluation,
    policy_improvement,
    policy_iteration,
    print_policy,
    q_from_v,
    random_policy,
    state_value_from_v,
    value_iteration,
)


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.env = GridWorld()

    def test_transitions_match_hand_calculation_and_reject_invalid_inputs(self):
        cases = [
            ((1, 0), "DOWN", [(1.0, (2, 0), -1.0)]),
            ((2, 2), "UP", [(1.0, (1, 2), -10.0)]),
            ((1, 0), "RIGHT", [(1.0, (1, 0), -1.0)]),
            ((0, 0), "UP", [(1.0, (0, 0), -1.0)]),
            ((0, 1), "RIGHT", [(1.0, (0, 2), 10.0)]),
            ((0, 2), "LEFT", []),
            ((1, 2), "UP", []),
        ]
        for state, action, expected in cases:
            with self.subTest(state=state, action=action):
                self.assertEqual(self.env.transitions(state, action), expected)
        for state, action in [((1, 1), "UP"), ((-1, 0), "UP"), ((0, 0), "FLY")]:
            with self.subTest(state=state, action=action):
                with self.assertRaises(ValueError):
                    self.env.transitions(state, action)

    def test_bellman_backup_and_terminal_reward(self):
        V = {state: 0.0 for state in self.env.states}
        V[(0, 1)] = 5.0
        original = V.copy()
        self.assertEqual(q_from_v(self.env, V, (0, 0), "RIGHT"), 3.5)
        self.assertEqual(q_from_v(self.env, V, (0, 1), "RIGHT"), 10.0)
        self.assertEqual(q_from_v(self.env, V, (2, 2), "UP"), -10.0)
        self.assertEqual(state_value_from_v(self.env, V, random_policy, (0, 2)), 0.0)
        self.assertEqual(V, original)

    def test_one_sweep_uses_only_old_values(self):
        expected = {
            (0, 0): -1.0, (0, 1): 1.75, (0, 2): 0.0,
            (1, 0): -1.0, (1, 2): 0.0,
            (2, 0): -1.0, (2, 1): -1.0, (2, 2): -3.25,
        }
        V = policy_evaluation(self.env, random_policy, max_iterations=1)
        self.assertEqual(V, expected)
        self.env.states.reverse()
        self.assertEqual(policy_evaluation(self.env, random_policy, max_iterations=1), expected)

    def test_random_policy_converges_to_bellman_fixed_point(self):
        self.assertEqual(random_policy(self.env, (0, 0)), dict.fromkeys(self.env.ACTIONS, 0.25))
        self.assertEqual(random_policy(self.env, (0, 2)), {})
        V = policy_evaluation(self.env, random_policy)
        residual = max(
            abs(V[state] - state_value_from_v(self.env, V, random_policy, state))
            for state in self.env.states
        )
        self.assertLess(residual, 1e-6)
        self.assertEqual(V[(0, 2)], 0.0)
        self.assertEqual(V[(1, 2)], 0.0)

    def test_deterministic_evaluation_matches_analytic_values_and_probability_one(self):
        policy = {state: "RIGHT" for state in self.env.states if not self.env.is_terminal(state)}
        V = deterministic_policy_evaluation(self.env, policy)
        self.assertEqual(V[(0, 0)], 8.0)
        self.assertEqual(V[(0, 1)], 10.0)
        self.assertAlmostEqual(V[(1, 0)], -10.0, delta=1e-5)
        self.assertEqual(V[(0, 2)], 0.0)
        self.assertEqual(V[(1, 2)], 0.0)

        def probability_one_policy(env, state):
            return {} if env.is_terminal(state) else {policy[state]: 1.0}

        self.assertEqual(V, policy_evaluation(self.env, probability_one_policy))
        one_sweep = deterministic_policy_evaluation(self.env, policy, max_iterations=1)
        self.assertEqual(one_sweep[(0, 0)], -1.0)
        self.assertEqual(one_sweep[(0, 1)], 10.0)

    def test_improvement_preserves_input_and_reports_stability(self):
        policy = {state: "UP" for state in self.env.states if not self.env.is_terminal(state)}
        original = policy.copy()
        V = deterministic_policy_evaluation(self.env, policy)
        new_policy, stable = policy_improvement(self.env, V, policy)
        self.assertEqual(policy, original)
        self.assertFalse(stable)
        self.assertEqual(new_policy[(0, 1)], "RIGHT")
        zero_values = {state: 0.0 for state in self.env.states}
        self.assertEqual(greedy_action_from_v(self.env, zero_values, (2, 2)), "DOWN")
        self.assertIsNone(greedy_action_from_v(self.env, V, (0, 2)))
        _, stable = policy_improvement(self.env, V, new_policy)
        self.assertTrue(stable)
        improved_V = deterministic_policy_evaluation(self.env, new_policy)
        for state in self.env.states:
            self.assertGreaterEqual(improved_V[state] + 1e-5, V[state])

    def test_step_8_solvers_match_the_optimal_gridworld_policy(self):
        expected_policy = {
            (0, 0): "RIGHT", (0, 1): "RIGHT", (1, 0): "UP",
            (2, 0): "UP", (2, 1): "LEFT", (2, 2): "LEFT",
        }
        V_pi, policy_pi = policy_iteration(self.env)
        V_vi = value_iteration(self.env)
        self.assertEqual(policy_pi, expected_policy)
        self.assertEqual(extract_policy(self.env, V_vi), expected_policy)
        for V in (V_pi, V_vi):
            self.assertAlmostEqual(V[(0, 0)], 8.0)
            self.assertAlmostEqual(V[(2, 2)], 1.8098)
            self.assertEqual(V[(0, 2)], 0.0)
            self.assertEqual(V[(1, 2)], 0.0)

        output = StringIO()
        with redirect_stdout(output):
            print_policy(self.env, policy_pi)
        self.assertEqual(output.getvalue(), "→  →  +10\n↑  ###  -10\n↑  ←  ←\n")


if __name__ == "__main__":
    unittest.main()
