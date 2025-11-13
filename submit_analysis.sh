#!/bin/bash
#SBATCH --job-name=high_priority_job
#SBATCH --output=job_output_%j.log
#SBATCH --error=job_error_%j.err
#SBATCH --partition=jag-hi
#SBATCH --account=nlp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --time=09:00:00 # Set for 9 hours

# Load your Conda environment
source /nlp/scr/busra/miniconda3/etc/profile.d/conda.sh
conda activate experiment2

# run the script
python advanced_nlp.py