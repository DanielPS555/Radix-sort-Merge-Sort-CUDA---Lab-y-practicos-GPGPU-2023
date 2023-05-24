#!/bin/bash
#SBATCH --job-name=gpgpu10_practico4_ej1
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --time=00:01:00

#SBATCH --partition=besteffort
# SBATCH --partition=normal

#SBATCH --qos=besteffort_gpu
# SBATCH --qos=gpu

#SBATCH --gres=gpu:p100:1
# #SBATCH --mail-type=ALL
#SBATCH --mail-user=mi@correo
#SBATCH -o salidaEj1a.out

export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

make

## Variables

# Output CSV file name with execution times
file="test.csv"
# Algorithm to be executed
# 0: SIMPLE_TRANSPOSE
# 1: IMPROVED_TRANSPOSE
# 2: IMPROVED_TRANSPOSE_DUMMY

rm -f $file

echo "Algorithm,Ms" > $file

for algorithm in {0..0}
do
  ./histogram in/fing1.pgm $algorithm >> $file
done