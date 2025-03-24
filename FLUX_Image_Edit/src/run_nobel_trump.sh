python edit.py  --source_prompt "" \
                --target_prompt "A minimalistic line-drawing portrait of Donald Trump with black lines and brown shadow" \
                --guidance 2.5 \
                --source_img_dir 'examples/source/nobel.jpg' \
                --num_steps 25  \
                --inject 3 \
                --name 'flux-dev'  \
                --output_dir 'examples/edit-result/nobel/' 
