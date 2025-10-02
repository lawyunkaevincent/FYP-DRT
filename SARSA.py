# SARSA.py
class SarsaTrainer:
    def __init__(self, epsilon=0.1, log_file="reward.txt",
                 save_every=None, ckpt_path=None, start_episode=0,
                 best_ckpt_path=None):
        """
        epsilon: ε used for acting during training
        log_file: path to append episode rewards
        save_every: int or None, save 'ckpt_path' every N episodes
        ckpt_path: where to save the rolling/latest checkpoint
        start_episode: integer index to start logging from (useful when resuming)
        best_ckpt_path: if set, we save whenever we hit a new best episode reward
        """
        self.epsilon = epsilon
        self.log_file = log_file
        self.save_every = save_every
        self.ckpt_path = ckpt_path
        self.start_episode = start_episode

        # track best policy by single-episode return
        self.best_ckpt_path = best_ckpt_path
        self.best_reward = float("-inf")

    def run_episode(self, agent, env):
        state = env.reset()
        action = agent.act(state, epsilon=self.epsilon)  # ε-greedy
        total_reward = 0.0

        while True:
            next_state, reward, done = env.step(action)
            total_reward += reward
            next_action = None if done else agent.act(next_state, epsilon=self.epsilon)

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
                print(f"\n\n-------------------Currently At Episode EP {ep}------------------------\n\n")

                R = self.run_episode(agent, env)
                log_line = f"Episode {ep}, total reward={R:.2f}\n"
                print(log_line.strip())
                f.write(log_line)
                f.flush()

                episodes_trained += 1

                # Rolling/periodic checkpoint
                if self.ckpt_path and self.save_every and (episodes_trained % self.save_every == 0):
                    agent.save(self.ckpt_path, episodes_trained=episodes_trained)

                # Save best policy so far (by single-episode reward)
                if self.best_ckpt_path and R > self.best_reward:
                    self.best_reward = R
                    print(f"[INFO] New best reward {R:.2f}, saving best policy...")
                    agent.save(
                        self.best_ckpt_path,
                        episodes_trained=episodes_trained,
                        extra_meta={"best_reward": R, "episode_index": ep}
                    )

        # Final save of latest agent at the end (optional but handy)
        if self.ckpt_path:
            agent.save(self.ckpt_path, episodes_trained=episodes_trained)