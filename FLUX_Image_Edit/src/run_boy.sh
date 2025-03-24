python edit.py  --source_prompt "A young boy is playing with a toy airplane on the grassy front lawn of a suburban house, with a blue sky and fluffy clouds above." \
                --target_prompt "A young boy is playing with a toy airplane on the grassy front lawn of a suburban house, with a small brown dog playing beside him, and a blue sky with fluffy clouds above." \
                --guidance 2 \
                --source_img_dir 'examples/source/boy.jpg' \
                --num_steps 15 --offload \
                --inject 2 \
                --name 'flux-dev'  \
                --output_dir 'examples/edit-result/dog' 


