#!/bin/bash
#SBATCH --job-name=gpgpu10_practico2_ej1a
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --time=00:00:15

#SBATCH --partition=besteffort
# SBATCH --partition=normal

#SBATCH --qos=besteffort_gpu
# SBATCH --qos=gpu

#SBATCH --gres=gpu:p100:1
# #SBATCH --mail-type=ALL
#SBATCH --mail-user=mi@correo
#SBATCH -o salidaEj1c.out

export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
#cd /clusteruy/home/...

# cd anio2023
nvcc -lm -lineinfo ej1c.cu -o ej1c_sol
./ej1c_sol secreto.txt

# $1 $2 $3 $4 $5 $6 $7 $8 $9 $10 $11 $12 $13 $14 $15
