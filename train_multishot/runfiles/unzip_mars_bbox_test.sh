#!/bin/bash
#BSUB -J unzip_mars_bbox_test
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -q normal
#BSUB -n 56

cd $HOME/dataset/MARS

unzip -d MARS2 MARS.zip

