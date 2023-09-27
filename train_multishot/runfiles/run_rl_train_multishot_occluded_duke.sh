#!/bin/bash
#BSUB -J rl_train_multishot
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -q gpu_v100
#BSUB -gpu num=8:mode=exclusive_process:aff=yes


module load anaconda3

if [ -f "/seu_share/home/liwei01/220205671/miniconda3/etc/profile.d/conda.sh" ]; then
	. "/seu_share/home/liwei01/220205671/miniconda3/etc/profile.d/conda.sh"
else
	export PATH="/seu_share/home/liwei01/220205671/miniconda3/bin:$PATH"
fi

conda activate py37

export PATH=$HOME/miniconda3/envs/py37/bin:$PATH

module load cuda-11.02

cd $HOME/DL-reid-baseline/train_multishot

python3 rl_train_multishot_linux.py -j 8 -g $CUDA_VISIBLE_DEVICES -d occluded_duke --p 4 --k 4 --rl_k 4

#cd $HOME/DL-reid-baseline/train_multishot/results
#mv models/ models-4-occluded-duke-rl
#
#mkdir models
#cd $HOME/DL-reid-baseline/train_multishot
#
#python3 rl_train_multishot_linux.py -j 8 -g $CUDA_VISIBLE_DEVICES -d occluded_duke --p 4 --k 4 --rl_k 8

