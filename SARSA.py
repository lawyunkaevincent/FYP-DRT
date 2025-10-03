# SARSA.py
class SarsaTrainer:
    def __init__(self, epsilon=0.1, log_file="reward.txt",
                 save_every=None, ckpt_path=None, start_episode=0,
                 best_ckpt_path=None,
                 epsilon_start=None, epsilon_end=None, epsilon_decay=None):
        """
        epsilon: fixed exploration if decay not used
        epsilon_start, epsilon_end, epsilon_decay:
            - if set, override epsilon each episode with decayed value
        """
        self.epsilon = epsilon
        self.log_file = log_file
        self.save_every = save_every
        self.ckpt_path = ckpt_path
        self.start_episode = start_episode

        self.best_ckpt_path = best_ckpt_path
        self.best_reward = float("-inf")

        # --- NEW: decay config ---
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay  # e.g. 0.99 for exponential, or int for linear steps

    def _get_epsilon(self, episode_idx):
        """Return epsilon for a given episode index."""
        if self.epsilon_start is None:
            return self.epsilon  # fixed

        # Exponential decay
        if isinstance(self.epsilon_decay, float):
            eps = self.epsilon_start * (self.epsilon_decay ** episode_idx)
            return max(self.epsilon_end, eps)

        # Linear decay (epsilon_decay = number of episodes over which to decay)
        elif isinstance(self.epsilon_decay, int):
            frac = min(1.0, episode_idx / self.epsilon_decay)
            eps = self.epsilon_start - frac * (self.epsilon_start - self.epsilon_end)
            return max(self.epsilon_end, eps)

        return self.epsilon_start

    def run_episode(self, agent, env, epsilon):
        state = env.reset()
        action = agent.act(state, epsilon=epsilon)
        total_reward = 0.0

        while True:
            next_state, reward, done = env.step(action)
            total_reward += reward
            next_action = None if done else agent.act(next_state, epsilon=epsilon)
            agent.update(state, action, reward, next_state, next_action, done)
            if done:
                break
            state, action = next_state, next_action
        return total_reward

    def train(self, agent, env, episodes=10):
        episodes_trained = getattr(agent, "_episodes_trained", 0)

        with open(self.log_file, "a") as f:
            for i in range(episodes):
                ep = self.start_episode + i
                eps = self._get_epsilon(ep)

                print(f"\n\n--- Episode {ep} (epsilon={eps:.4f}) ---\n\n")
                R = self.run_episode(agent, env, epsilon=eps)
                log_line = f"Episode {ep}, epsilon={eps:.4f}, total reward={R:.2f}\n"
                print(log_line.strip())
                f.write(log_line)
                f.flush()

                episodes_trained += 1

                if self.ckpt_path and self.save_every and (episodes_trained % self.save_every == 0):
                    agent.save(self.ckpt_path, episodes_trained=episodes_trained)

                if self.best_ckpt_path and R > self.best_reward:
                    self.best_reward = R
                    print(f"[INFO] New best reward {R:.2f}, saving best policy...")
                    agent.save(self.best_ckpt_path,
                               episodes_trained=episodes_trained,
                               extra_meta={"best_reward": R, "episode_index": ep})

        if self.ckpt_path:
            agent.save(self.ckpt_path, episodes_trained=episodes_trained)
