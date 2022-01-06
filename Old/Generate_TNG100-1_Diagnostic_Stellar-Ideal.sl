#!/bin/bash 
#SBATCH --job-name=Moment_Plots
#SBATCH --account=rrg-jfncc_cpu
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=4000M
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END
#SBATCH --mail-user=cbottrel@uvic.ca

source /home/bottrell/virtualenvs/p37/bin/activate
cd /home/bottrell/scratch/Merger_Kinematics/RealSim-IFS
python Generate_TNG100-1_Diagnostic_Stellar-Ideal.py 
