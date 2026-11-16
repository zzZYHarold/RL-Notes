import unittest
from unittest.mock import patch

from rl_tabular.model_free_planning import mc_prediction, q_learning, sarsa, td0_prediction


class TwoStepEnv:
    def reset(self):
        self.state = "start"
        return self.state

    def step(self, action):
        if self.state == "start":
            self.state = "next"
            return self.state, 0, False
        if self.state == "next":
            self.state = "terminal"
            reward = 2 if action == "good" else -1
            return self.state, reward, True
        raise AssertionError("step() called after the episode ended")


class ModelFreePlanningTests(unittest.TestCase):
    def test_mc_updates_only_first_visit_with_discounted_return(self):
        episodes = [[("a", 1), ("b", 2), ("a", 4)]]
        V = mc_prediction(episodes, gamma=0.5, alpha=0.5)
        self.assertEqual(V, {"a": 1.5, "b": 2.0})

    def test_td0_bootstraps_only_when_transition_is_nonterminal(self):
        transitions = [
            ("b", 4, "end", True),
            ("a", 1, "b", False),
            ("a", 2, "b", True),
        ]
        V = td0_prediction(transitions, gamma=0.5, alpha=0.5)
        self.assertEqual(V, {"a": 1.5, "b": 2.0})

    def test_sarsa_uses_sampled_action_and_q_learning_uses_maximum(self):
        actions = ("good", "other")
        # 第一回合学到 Q(next, good)=2。第二回合在 next 实际选 other。
        sampled_actions = ("other", "good", "other", "other")
        for learn, expected_start_value in ((sarsa, 0.0), (q_learning, 1.0)):
            with self.subTest(algorithm=learn.__name__):
                with patch("rl_tabular.model_free_planning.choose_action", side_effect=sampled_actions) as choose:
                    Q = learn(TwoStepEnv(), actions, episodes=2, gamma=0.5, alpha=1.0)
                self.assertEqual(choose.call_count, 4)
                self.assertEqual(Q["start"]["other"], expected_start_value)
                self.assertEqual(Q["next"], {"good": 2.0, "other": -1.0})
                self.assertNotIn("terminal", Q)


if __name__ == "__main__":
    unittest.main()
