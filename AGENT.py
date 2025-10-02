import random
import collections
import numpy as np

class SarsaAgent:
    def __init__(self, action_space, gamma=0.95, alpha=0.1, epsilon=0.1):
        """
        action_space: list of possible actions (e.g. [1,2,3,4,5])
        gamma: discount factor
        alpha: learning rate (if None, use 1/N update)
        epsilon: initial exploration prob
        """
        self.action_space = action_space
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

        # tabular Q and visit counts
        self.Q = {}   # dict with keys (state, action)
        self.N = collections.Counter()

    def _key(self, state, action):
        # state must be hashable: you already return tuple from env._build_state
        return (state, action)

    def get_Q(self, state, action):
        return self.Q.get(self._key(state, action), 0.0)

    def act(self, state, available_actions=None, epsilon=None):
        """
        epsilon-greedy action selection
        """
        if epsilon is None:
            epsilon = self.epsilon
        acts = available_actions if available_actions else self.action_space
        if random.random() < epsilon:
            return random.choice(acts)
        # greedy: pick argmax Q
        qs = [self.get_Q(state, a) for a in acts]
        max_q = max(qs)
        best_actions = [a for a, q in zip(acts, qs) if q == max_q]
        return random.choice(best_actions)

    def update(self, s, a, r, s_next, a_next, done):
        """
        Standard SARSA(0) update
        """
        key = self._key(s, a)
        q_sa = self.Q.get(key, 0.0)

        if done:
            target = r
        else:
            q_next = self.get_Q(s_next, a_next)
            target = r + self.gamma * q_next

        # step-size
        if self.alpha is None:
            self.N[key] += 1
            alpha = 1.0 / self.N[key]
        else:
            alpha = self.alpha

        self.Q[key] = q_sa + alpha * (target - q_sa)
