#! /bin/bash

CUDA_VISIBLE_DEVICES=1

python trainer.py  -c config/config.yaml  -b /home/nas/user/kbh/VFWS/ -m v1
