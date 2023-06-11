#!/bin/bash
#SBATCH --job-name=gpgpu10_practico4_ej2
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
#SBATCH -o salidaEj2.out

export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

make

## Variables

# Output CSV file name with execution times
file="test.csv"
# Algorithm to be executed
# 0: BLUR_WITH_SHARED
# 1: BLUR_WITHOUT_SHARED

rm -f $file

echo "Algorithm,Ms" > $file
echo "640p" > $file
for algorithm in {1..0}
do
  ./blur in/dwsample-pgm-640.pgm $algorithm >> $file
done

echo "1280p" >> $file
for algorithm in {1..0}
do
  ./blur in/dwsample-pgm-1280.pgm $algorithm >> $file
done

echo "1920p" >> $file
for algorithm in {1..0}
do
  ./blur in/dwsample-pgm-1920.pgm $algorithm >> $file
done

echo "4k" >> $file
for algorithm in {1..0}
do
  ./blur in/dwsample-pgm-4k.pgm $algorithm >> $file
done

