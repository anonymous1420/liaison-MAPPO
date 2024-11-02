#!/bin/sh
env="UAV"
scenario="confrontation"
num_agents=14
num_red=7
num_blue=7
algo="rmappo"
exp="check"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python ../train/train_uav.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 128 --num_mini_batch 1 --episode_length 25 --num_env_steps 2000000 \
    --ppo_epoch 15 --gain 0.01 --lr 7e-4 --critic_lr 7e-4 --wandb_name "UAV" --user_name "shuaixiangyu123-new-york-post" --share_policy
done
read -p "Press any key to continue..."