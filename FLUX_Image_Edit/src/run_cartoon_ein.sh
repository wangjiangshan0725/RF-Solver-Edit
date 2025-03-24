python edit.py  --source_prompt "" \
                --target_prompt "a cartoon style Albert Einstein raising his left hand " \
                --guidance 2 \
                --source_img_dir 'examples/source/cartoon.jpg' \
                --num_steps 25  \
                --inject 2 \
                --name 'flux-dev'  \
                --output_dir 'examples/edit-result/cartoon/' 