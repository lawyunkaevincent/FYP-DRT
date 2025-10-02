import argparse
from SARSA import SarsaTrainer
from AGENT import SarsaAgent
from SUMOENV import SumoTaxiEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="Path to .sumocfg file")
    parser.add_argument("--gui", action="store_true", help="Use sumo-gui")
    parser.add_argument("--step-length", type=float, default=1.0, help="SUMO step length (s)")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Exploration epsilon")
    args = parser.parse_args()

    # Create environment
    env = SumoTaxiEnv(cfg_path=args.cfg, step_length=args.step_length, use_gui=args.gui)

    # Create agent and trainer
    agent = SarsaAgent(action_space=env.action_space, gamma=0.95, alpha=0.1, epsilon=args.epsilon)
    trainer = SarsaTrainer(epsilon=args.epsilon)

    # Train
    trainer.train(agent, env, episodes=args.episodes)

    # Close environment
    env.close()

if __name__ == "__main__":
    main()
