#!/bin/bash
#BSUB -J train_multishot
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -q gpu_v100
#BSUB -gpu num=2:mode=exclusive_process:aff=yes

module load anaconda3

if [ -f "/seu_share/home/liwei01/220205671/miniconda3/etc/profile.d/conda.sh" ]; then
	. "/seu_share/home/liwei01/220205671/miniconda3/etc/profile.d/conda.sh"
else
	export PATH="/seu_share/home/liwei01/220205671/miniconda3/bin:$PATH"
fi

conda activate tf1.15

export PATH=$HOME/miniconda3/envs/tf1.15/bin:$PATH

module load cuda-11.02

cd $HOME/DL-reid-baseline/image_process

python3 background_removal_linux.py
