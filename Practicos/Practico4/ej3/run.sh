#!/bin/bash
#SBATCH --job-name=gpgpu10_practico4_ej1
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --time=00:01:00

#SBATCH --partition=besteffort

#SBATCH --qos=besteffort_gpu

#SBATCH --gres=gpu:p100:1
#SBATCH -o ej3.out

export PATH=$PATH:/usr/local/cuda-12.1/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.1/lib64

make

images=("dwsample-pgm-640.pgm" "dwsample-pgm-1280.pgm" "dwsample-pgm-1920.pgm" "dwsample-pgm-4k.pgm")

## Variables

# Output CSV file name with execution times
file="test.csv"
# Algorithm to be executed
# 0: SIMPLE_TRANSPOSE
# 1: IMPROVED_TRANSPOSE
# 2: IMPROVED_TRANSPOSE_DUMMY

rm -f $file

echo "Image,Algorithm,Ms" > $file

for algorithm in {0..2}
do
  for image in "${images[@]}"
  do
    echo -n "$image", >> $file
    ./histogram in/"$image" "$algorithm" >> $file
  done
done