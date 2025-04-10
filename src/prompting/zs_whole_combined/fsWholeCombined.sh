#!/bin/bash

#SBATCH --job-name=noCatzeroshot_whole
#SBATCH --account=st-wasserww-1-gpu
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/st-jzhu71-1/czhang/job/extraction/all_case_text_extraction/testset/fsWholeCombined/logging/output_%j.txt
#SBATCH --error=/scratch/st-jzhu71-1/czhang/job/extraction/all_case_text_extraction/testset/fsWholeCombined/logging/error_%j.txt
#SBATCH --mail-user=czhang@cmmt.ubc.ca
#SBATCH --mail-type=FAIL

module load gcc
module load openmpi
module load cuda
module load apptainer

# Define paths for caches
export APPTAINER_CACHEDIR=/scratch/st-jzhu71-1/czhang/iem/cache/apptainer
export HF_DATASETS_CACHE=/scratch/st-jzhu71-1/czhang/iem/cache/huggingface
export HF_HOME=/scratch/st-jzhu71-1/czhang/iem/cache/transformers/hub

# Start Ollama server within the container if needed
echo "Starting Ollama Server"
apptainer exec --nv \
  --home /scratch/st-jzhu71-1/czhang/iem \
  --env XDG_CACHE_HOME=/scratch/st-jzhu71-1/czhang/iem \
  /arc/project/st-jzhu71-1/czhang/container/dspy.sif bash -c 'source activate /arc/project/st-jzhu71-1/czhang/iem/ollama/ && export PATH=/scratch/st-jzhu71-1/czhang/iem/scratch/bin:$PATH && ollama serve &'

# Execute commands within the Apptainer container
echo "Running Script"
apptainer exec --nv \
  --home /scratch/st-jzhu71-1/czhang/iem \
  --env XDG_CACHE_HOME=/scratch/st-jzhu71-1/czhang/iem \
  /arc/project/st-jzhu71-1/czhang/container/dspy.sif bash -c 'source activate /arc/project/st-jzhu71-1/czhang/iem/ollama/ && python3 zsWholeCompiled.py /scratch/st-jzhu71-1/czhang/job/extraction/all_case_text_extraction/testset/testing_case_set.json'
