#!/bin/bash 
#SBATCH --job-name=TNG_MaNGA
#SBATCH --account=rrg-jfncc_cpu
#SBATCH --time=3:0:0
#SBATCH --mem-per-cpu=4000M
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END
#SBATCH --error='SLURM/JOBNAME-'%A_%a'.err' 
#SBATCH --output='SLURM/JOBNAME-'%A_%a'.out' 
#SBATCH --mail-user=cbottrel@uvic.ca
#SBATCH --array=0-207%16

source /home/bottrell/virtualenvs/p37/bin/activate

python generate_manga-tng_observations.py $SLURM_ARRAY_TASK_ID
