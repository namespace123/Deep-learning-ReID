#!/bin/bash
#BSUB -J transReID
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -q gpu_v100
#BSUB -gpu num=6:mode=exclusive_process:aff=yes


module load anaconda3

if [ -f "/seu_share/home/liwei01/220205671/miniconda3/etc/profile.d/conda.sh" ]; then
	. "/seu_share/home/liwei01/220205671/miniconda3/etc/profile.d/conda.sh"
else
	export PATH="/seu_share/home/liwei01/220205671/miniconda3/bin:$PATH"
fi

conda activate py37

export PATH=$HOME/miniconda3/envs/py37/bin:$PATH

module load cuda-11.02

cd $HOME/TransReID-main

python3 train_linux.py -g $CUDA_VISIBLE_DEVICES

