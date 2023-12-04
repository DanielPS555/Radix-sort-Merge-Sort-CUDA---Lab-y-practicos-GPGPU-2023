#!/bin/bash
#SBATCH --job-name=gpgpu10_practico2_ej2
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --time=00:03:00

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
#cd /clusteruy/home/...

make
sleep 2
declare -a algorithms=("MAIN_AJUSTE_BRILLO_CPU" "MAIN_AJUSTE_BRILLO_NO_COALESCED" "MAIN_AJUSTE_BRILLO_COALESCED" "MAIN_EFECTO_PAR_IMPAR_NO_DIVERGENTE" "MAIN_EFECTO_PAR_IMPAR_DIVERGENTE" "MAIN_BLUR_GPU" "MAIN_BLUR_CPU")

## Variables
file="test.csv"  # Output CSV file name with execution times
size=2           # How many times the image will be resized
                 # 0 -> original size,
                 # 1 -> original size and 1/2
                 # 2 -> original size, 1/2 and 1/4,
                 # 3 -> original size, 1/2, 1/4 and 1/8

rm -f $file
echo "Algorithm,Size,Ms" > $file


# Start the loop
for ((j=0; j<=$size; j++))
do
  for i in "${!algorithms[@]}"; do
    echo "Algorithm: ${algorithms[$i]}"
    ./blur img/fing1.pgm $j $i >> $file
  done
done



# $1 $2 $3 $4 $5 $6 $7 $8 $9 $10 $11 $12 $13 $14 $15
