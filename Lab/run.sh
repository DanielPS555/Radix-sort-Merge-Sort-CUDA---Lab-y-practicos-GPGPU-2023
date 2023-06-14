#!/bin/bash
#SBATCH --job-name=gpgpu10_lab
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --time=00:01:00

#SBATCH --partition=besteffort

#SBATCH --qos=besteffort_gpu

#SBATCH --gres=gpu:p100:1
#SBATCH -o lab.out

export PATH=$PATH:/usr/local/cuda-12/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12/lib64

make
./sort

