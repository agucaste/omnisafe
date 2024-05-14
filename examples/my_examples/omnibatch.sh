#!/bin/bash

#SBATCH --job-name="TRPO_BC_naive_both"
#SBATCH --output="TRPO_BC_naive.out"
#SBATCH --error="TRPO_BC_naive.err"
#SBATCH --partition=parallel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --time=16:00:00



# Change directory to the particular folder
cd /scratch4/emallad1/agu/omnisafe/examples/my_examples

# Load Anaconda module
ml anaconda

# Define the range of seed values
SEED_VALUES=(33 5 17 98)

# Loop over each seed value
for SEED in "${SEED_VALUES[@]}"; do
    # Activate your custom Conda environment
    conda activate omnisafe

    # Run your Python script with the seed value in the background
    python -i batch_run.py --seed $SEED &
done

# Wait for all background processes to finish
wait

# Deactivate the Conda environment
conda deactivate
