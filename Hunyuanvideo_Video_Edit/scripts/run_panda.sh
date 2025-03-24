#!/bin/bash


python3 sample_video.py   \
        --target_prompt 'A panda wearing a Crown walking in the snow'  \
        --infer-steps 25   \
        --source_prompt 'A panda walking in the snow'  \
        --flow-reverse  \
        --use-cpu-offload   \
        --save-path ./panda \
        --source_path './panda.mp4' \
        --inject_step 3 \
        --embedded-cfg-scale 7 
