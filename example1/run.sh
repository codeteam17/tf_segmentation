#!/bin/bash
rm -rf data
mkdir data
python generate_dataset.py
python train.py