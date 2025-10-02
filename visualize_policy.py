# visualize_policy.py
import argparse
from AGENT import SarsaAgent
from SUMOENV import SumoTaxiEnv

def main():
    parser = argparse.ArgumentParser(description="Visualize best SARSA policy in SUMO")
    parser.add_argument("--cfg", required=True, help="Path to .sumocfg file")
    parser.add_argument("--ckpt", required=True, help="Path to best checkpoint .pkl (e.g., checkpoints/best_sarsa.pkl)")
    parser.add_argument("--step-length", type=float, default=1.0, help="SUMO step length (s)")
    parser.add_argument("--no-gui", action="store_true", help="Run without SUMO GUI")

    args = parser.parse_args()

    # Always load SUMO with GUI unless explicitly disabled
    env = SumoTaxiEnv(cfg_path=args.cfg, step_length=args.step_length, use_gui=not args.no_gui)

    # Load best policy
    agent = SarsaAgent.load(args.ckpt)

    # Force greedy evaluation (ε=0)
    state = env.reset()
    total_reward = 0.0

    while True:
        action = agent.act(state, epsilon=0.0)  # greedy
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state
        if done:
            break

    print(f"[EVAL] Total reward from best policy = {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    main()
