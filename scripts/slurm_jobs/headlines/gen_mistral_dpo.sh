#!/bin/bash -l
#SBATCH --job-name=gen_mistral_dpo
#SBATCH --gres=gpu:a40:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/hpc/<hpc_account>/<hpc_user>/thesis/scripts/jobs/gen_mistral_dpo_%j.log
#SBATCH --error=/home/hpc/<hpc_account>/<hpc_user>/thesis/scripts/jobs/gen_mistral_dpo_%j.err
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
module load python/3.12-base
source ~/thesis_env/bin/activate
export WORK=/home/atuin/<hpc_account>/<hpc_user>
export HF_HOME=$WORK/.cache/huggingface
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
python ~/thesis/scripts/generate.py --model_id NousResearch/Nous-Hermes-2-Mistral-7B-DPO --family mistral --stage dpo
