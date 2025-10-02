class SarsaTrainer:
    def __init__(self, epsilon=0.1, log_file="reward.txt"):
        self.epsilon = epsilon
        self.log_file = log_file

    def run_episode(self, agent, env):
        state = env.reset()
        action = agent.act(state, epsilon=self.epsilon)   # ε-greedy
        total_reward = 0.0

        while True:
            next_state, reward, done = env.step(action)
            total_reward += reward
            next_action = None if done else agent.act(next_state, epsilon=self.epsilon)
            # SARSA update
            agent.update(state, action, reward, next_state, next_action, done)
            if done:
                break
            state, action = next_state, next_action
        return total_reward

    def train(self, agent, env, episodes=10):
        with open(self.log_file, "w") as f:   # open file once
            for ep in range(episodes):
                print(f"\n\n-------------------Currently At Episode EP {ep}------------------------\n\n")
                R = self.run_episode(agent, env)
                log_line = f"Episode {ep}, total reward={R:.2f}\n"
                print(log_line.strip())      # still print to console
                f.write(log_line)            # append to file
                f.flush()                    # ensure it’s written immediately
