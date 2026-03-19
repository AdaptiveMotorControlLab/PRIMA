exp_name_stage1=primaStage1
exp_name_stage2=primaStage2
experiment1=primaStage1
experiment2=primaStage2
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=1 python train.py exp_name=$exp_name_stage1 experiment=$experiment1 trainer=gpu launcher=local
cp -r ./logs/train/runs/$exp_name_stage1 ./logs/train/runs/$exp_name_stage2
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=1 python train.py exp_name=$exp_name_stage2 experiment=$experiment2 trainer=gpu launcher=local