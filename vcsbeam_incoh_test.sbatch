#!/bin/bash -l
 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=370gb
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --account=mwavcs
#SBATCH --nice=0
#SBATCH --export=NONE
 
module use /astro/mwasci/jtian/software/mwa_sles12sp5/modulefiles
module load bifrost
module use /pawsey/mwa/software/python3/modulefiles
module load pymwalib/v0.13.0

python3 incoh_beamformingV3_1.py
