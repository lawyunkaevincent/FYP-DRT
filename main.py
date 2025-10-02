# main.py
import argparse
from SARSA import SarsaTrainer
from AGENT import SarsaAgent
from SUMOENV import SumoTaxiEnv  # your existing environment

def main():
    parser = argparse.ArgumentParser()
    # --- SUMO / env options ---
    parser.add_argument("--cfg", required=True, help="Path to .sumocfg file")
    parser.add_argument("--gui", action="store_true", help="Use sumo-gui")
    parser.add_argument("--step-length", type=float, default=1.0, help="SUMO step length (s)")

    # --- training options ---
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Exploration epsilon")

    # --- checkpointing options ---
    parser.add_argument("--load-ckpt", type=str, default=None,
                        help="Path to an existing checkpoint to resume from")
    parser.add_argument("--save-ckpt", type=str, default="checkpoints/latest_sarsa.pkl",
                        help="Where to save rolling/latest checkpoints")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save latest checkpoint every N episodes (0/None disables)")
    parser.add_argument("--best-ckpt", type=str, default="checkpoints/best_sarsa.pkl",
                        help="Where to save the best policy (highest episode reward)")

    args = parser.parse_args()

    # Build env first (to get its action_space)
    env = SumoTaxiEnv(cfg_path=args.cfg, step_length=args.step_length, use_gui=args.gui)

    # Create or load agent
    if args.load_ckpt:
        agent = SarsaAgent.load(args.load_ckpt)
        if list(agent.action_space) != list(env.action_space):
            print("[WARN] action_space mismatch between checkpoint and env.")
        start_episode = getattr(agent, "_episodes_trained", 0)
        print(f"[INFO] Resuming from checkpoint (episodes_trained={start_episode}).")
    else:
        agent = SarsaAgent(
            action_space=env.action_space,
            gamma=0.95,
            alpha=0.1,
            epsilon=args.epsilon
        )
        start_episode = 0

    # Configure trainer
    save_every = args.save_every if args.save_every and args.save_every > 0 else None
    trainer = SarsaTrainer(
        epsilon=args.epsilon,
        log_file="reward.txt",
        save_every=save_every,
        ckpt_path=args.save_ckpt if save_every else None,
        best_ckpt_path=args.best_ckpt,
        start_episode=start_episode
    )

    # Train
    trainer.train(agent, env, episodes=args.episodes)

    # Cleanup
    env.close()

if __name__ == "__main__":
    main()
