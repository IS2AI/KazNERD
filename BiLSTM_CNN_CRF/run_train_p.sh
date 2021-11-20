#!/bin/bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
GPU=11
# device [cpu, cuda:0, cuda:1, ...]
python -u main.py --config ./Config/config.cfg --device cuda:$GPU --train -p
