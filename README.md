Run the code with the following command:

If start fresh:
python main.py --cfg .\RLTrainingMap1\map.sumocfg --episodes 100 --save-every 10 --save-ckpt checkpoints/latest.pkl --best-ckpt checkpoints/best_sarsa.pkl --epsilon-start 1.0 --epsilon-end 0.05 --epsilon-decay 0.99 --reward-file reward.txt

If resume from checkpoint:
python main.py --cfg .\RLTrainingMap1\map.sumocfg --load-ckpt checkpoints/best_sarsa.pkl --episodes 30 --save-every 10 --save-ckpt checkpoints/latest.pkl --best-ckpt checkpoints/best_sarsa.pkl --epsilon-start 1.0 --epsilon-end 0.05 --epsilon-decay 0.99 --reward-file reward.txt

To visualize the best policy:
python visualize_policy.py --cfg .\RLTrainingMap1\map.sumocfg --ckpt checkpoints/best_sarsa.pkl

To run the dispactcher at the DQN folder:
python .\dispatcher.py --cfg ..\SmallTestingMap\map.sumocfg   

To generate the request: 
python .\request_chain_generator.py --report D:\FYP\FYP-DRT\SmallTestingMap\connectivity_report.json --taxi D:\FYP\FYP-DRT\SmallTestingMap\map.rou.xml --output D:\FYP\FYP-DRT\SmallTestingMap\persontrips_scale.rou.xml --num-requests 200 --depart-step 25 75 200 --max-random-deviation-pct 10