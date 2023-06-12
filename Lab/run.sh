#!/bin/bash
#SBATCH --job-name=lab
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --time=00:1:00

#SBATCH --partition=besteffort
# SBATCH --partition=normal

#SBATCH --qos=besteffort_gpu
# SBATCH --qos=gpu

#SBATCH --gres=gpu:p100:1
# #SBATCH --mail-type=ALL
#SBATCH --mail-user=mi@correo
#SBATCH -o salidaLab.out

export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

make
./sort

