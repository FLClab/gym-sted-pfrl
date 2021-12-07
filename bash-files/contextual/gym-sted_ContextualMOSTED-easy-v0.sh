#!/usr/bin/env bash
#
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=26
#SBATCH --array=2-5
#SBATCH --mem=10G
#SBATCH --partition=batch_48h
#SBATCH --output=logs/%x-%j.out
#

#### PARAMETERS

# Use this directory venv, reusable across RUNs
VENV_DIR=${HOME}/gym-sted

module load python/3.7

[ -d $VENV_DIR ] || virtualenv --no-download $VENV_DIR

source $VENV_DIR/bin/activate

#### RUNNING STUFF

# Moves to working folder
cd ~/Documents/gym-sted-pfrl

echo "**** STARTED TRAINING ****"

python main.py --env gym_sted:ContextualMOSTED-easy-v0 \
                --num-envs 25 --batchsize 64 --checkpoint-freq 5000 --steps 200000 \
                --gamma 0. --seed ${SLURM_ARRAY_TASK_ID}

echo "**** ENDED TRAINING ****"
