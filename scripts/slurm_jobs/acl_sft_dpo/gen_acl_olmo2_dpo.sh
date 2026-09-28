#!/bin/bash -l
#SBATCH --job-name=acl_olmo2_dpo
#SBATCH --gres=gpu:a40:1
#SBATCH --time=04:00:00
#SBATCH --output=/home/hpc/<hpc_account>/<hpc_user>/thesis/scripts/jobs/acl_instruction_logs/acl_olmo2_dpo_%j.log
#SBATCH --error=/home/hpc/<hpc_account>/<hpc_user>/thesis/scripts/jobs/acl_instruction_logs/acl_olmo2_dpo_%j.err
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
module load python/3.12-base
source ~/thesis_env/bin/activate
export WORK=/home/atuin/<hpc_account>/<hpc_user>
export HF_HOME=$WORK/.cache/huggingface
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
python ~/thesis/scripts/generate_acl_instruction.py --model_id allenai/OLMo-2-1124-7B-DPO --family olmo2 --stage dpo
