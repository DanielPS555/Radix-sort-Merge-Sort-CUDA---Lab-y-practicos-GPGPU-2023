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

# cd anio2023
make

declare -a algorithms=("MAIN_AJUSTE_BRILLO_CPU" "MAIN_AJUSTE_BRILLO_NO_COALESCED" "MAIN_AJUSTE_BRILLO_COALESCED" "MAIN_EFECTO_PAR_IMPAR_NO_DIVERGENTE" "MAIN_EFECTO_PAR_IMPAR_DIVERGENTE" "MAIN_BLUR_GPU" "MAIN_BLUR_CPU")

file="test.csv"
algorithm="MAIN_EFECTO_PAR_IMPAR_DIVERGENTE"
rm -f $file
echo "Algorithm,Size,Ms" > $file

# Get the index of the algorithm
index=-1
for i in "${!algorithms[@]}"; do
  if [[ "${algorithms[$i]}" = "${algorithm}" ]]; then
    index=$i
    break
  fi
done

if [[ $index -eq -1 ]]; then
  echo "Algorithm not found in array"
  exit 1
fi

# Start the loop
for j in {1..2}
do
  #for i in "${!algorithms[@]}"; do
  #  echo "Algorithm: ${algorithms[$i]}"
  #  ./blur img/fing1.pgm $j $i >> $file
  #done
  echo "Algorithm: ${algorithm}"
  ./blur img/fing1.pgm $j $index >> $file
done


# $1 $2 $3 $4 $5 $6 $7 $8 $9 $10 $11 $12 $13 $14 $15
