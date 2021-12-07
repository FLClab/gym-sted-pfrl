#!/usr/bin/env bash
#
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=11
#SBATCH --mem=16G
#SBATCH --partition=batch
#SBATCH --output=logs/%x-%j.out
#SBATCH --array=0-4
#

#### PARAMETERS

# Use this directory venv, reusable across RUNs
VENV_DIR=${HOME}/gym-sted

module load python/3.7

[ -d $VENV_DIR ] || virtualenv --no-download $VENV_DIR

source $VENV_DIR/bin/activate

#### RUNNING STUFF

SAMPLE_LIST=(
    5000
    50000
    100000
    150000
    200000
)
SAMPLE=${SAMPLE_LIST[${SLURM_ARRAY_TASK_ID}]}

# Moves to working folder
cd ~/Documents/gym-sted-pfrl/src/analysis

echo "**** STARTED EVALUATION ****"
echo "[----] Evaluating model " + ${1} + $SAMPLE

python run_gym1.py --model-name ${1} --num-envs 10 --eval-n-runs 100 --checkpoint $SAMPLE

echo "**** ENDED EVALUATION ****"
