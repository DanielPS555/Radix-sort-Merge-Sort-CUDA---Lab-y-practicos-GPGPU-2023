#!/bin/bash
#SBATCH --job-name=practico4ej3
#SBATCH --ntasks=1
#SBATCH --mem=4096
#SBATCH --time=00:02:00

#SBATCH --partition=besteffort

#SBATCH --qos=besteffort_gpu

#SBATCH --gres=gpu:p100:1
#SBATCH -o salida.out

export PATH=$PATH:/usr/local/cuda-12.1/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.1/lib64

source /etc/profile.d/modules.sh

make perf

## Variables

# Output CSV file name with execution times
file="test.csv"
# Algorithm to be executed
# 0: SIMPLE_TRANSPOSE
# 1: IMPROVED_TRANSPOSE
# 2: IMPROVED_TRANSPOSE_DUMMY

rm -f $file

echo "Algorithm,Ms" > $file

for algorithm in {2..2}
do
  compute-sanitizer ./histogram in/fing1.pgm $algorithm >> $file
done