Run the code with the following command:

If start fresh:
python main.py --cfg .\RLTrainingMap1\map.sumocfg --episodes 30 --save-every 10 --save-ckpt checkpoints/latest.pkl --best-ckpt

If resume from checkpoint:
python main.py --cfg .\RLTrainingMap1\map.sumocfg --load-ckpt checkpoints/best_sarsa.pkl --episodes 30 --save-every 10 --save-ckpt checkpoints/latest.pkl --best-ckpt checkpoints/best_sarsa.pkl

To visualize the best policy:
python visualize_policy.py --cfg .\RLTrainingMap1\map.sumocfg --ckpt checkpoints/best_sarsa.pkl
