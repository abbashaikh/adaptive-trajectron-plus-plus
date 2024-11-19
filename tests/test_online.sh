python test_online.py --log_dir=experiments/pedestrians/kf_models --data_dir=experiments/processed --conf=config.json --eval_data_dict=eth_test.pkl

python test_online_trajdata.py --data_loc_dict=\{\"eupeds_eth\":\ \"~/Projects/RobotSocialNavigation/datasets/eth_ucy_peds\"\} --log_dir=experiments/pedestrians/kf_models --conf=config.json --eval_data=eupeds_eth-test --trajdata_cache_dir=~/Projects/RobotSocialNavigation/.unified_data_cache
