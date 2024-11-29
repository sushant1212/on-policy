env="SMACv2"
map="10gen_terran"
algo="rmappo"
units="5v5"

exp="Terran_5v5_noComms"
comms_experiment="NoComms"
seed_max=5

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python3 ../train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --map_name ${map} --seed ${seed} --units ${units} --n_training_threads 1 --n_rollout_threads 5 --num_mini_batch 1 --episode_length 100 \
    --num_env_steps 3000000 --ppo_epoch 15 --use_value_active_masks --use_comet  --share_policy --save_interval 1000000  \
    --comms_experiment ${comms_experiment} --log_interval 1
done
