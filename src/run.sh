

CUDA_VISIBLE_DEVICES=0 python edit.py   --source_prompt "A woman hiking on a trail with mountains in the distance, carrying a backpack." \
                --target_prompt "A woman hiking on a trail with mountains in the distance, carrying a backpack and holding a hiking stick." \
                --guidance 2 \
                --source_img_dir '/group/40034/hilljswang/flux_release/examples/hiking.jpg' \
                --num_steps 15  \
                --inject 2 --offload \
                --name 'flux-dev'  \
                --output_dir '/group/40034/hilljswang/flux_release/examples/edit-result/hiking/' &



CUDA_VISIBLE_DEVICES=1 python edit.py   --source_prompt "A young boy is riding a brown horse in a countryside field, with a large tree in the background." \
                --target_prompt "A young boy is riding a camel in a countryside field, with a large tree in the background." \
                --guidance 2 \
                --source_img_dir '/group/40034/hilljswang/flux_release/examples/horse.jpg' \
                --num_steps 15  \
                --inject 3 --offload \
                --name 'flux-dev'  \
                --output_dir '/group/40034/hilljswang/flux_release/examples/edit-result/horse_test2/' 




CUDA_VISIBLE_DEVICES=1 python edit.py   --source_prompt "A young boy is playing with a toy airplane on the grassy front lawn of a suburban house, with a blue sky and fluffy clouds above." \
                --target_prompt "A young boy is playing with a toy airplane on the grassy front lawn of a suburban house, with a small brown dog playing beside him, and a blue sky with fluffy clouds above." \
                --guidance 2 \
                --source_img_dir '/group/40034/hilljswang/flux_release/examples/boy.jpg' \
                --num_steps 15 --offload \
                --inject 2 \
                --name 'flux-dev'  \
                --output_dir '/group/40034/hilljswang/flux_release/examples/edit-result/adddog1' 
