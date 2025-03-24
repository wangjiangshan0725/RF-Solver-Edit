#!/bin/bash

python3 sample_video.py   \
        --target_prompt 'A cat is eating a watermelon'     \
        --infer-steps 25    \
        --source_prompt ""     \
        --flow-reverse   \
        --use-cpu-offload    \
        --save-path ./rabbit \
        --source_path './rabbit_watermelon.mp4' \
        --inject_step 1 \
        --embedded-cfg-scale 1 
