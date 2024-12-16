<div align="center">
  
# Taming Rectified Flow for Inversion and Editing

[Jiangshan Wang](https://scholar.google.com/citations?user=HoKoCv0AAAAJ&hl=en)<sup>1,2</sup>, [Junfu Pu](https://pujunfu.github.io/)<sup>2</sup>, [Zhongang Qi](https://scholar.google.com/citations?hl=en&user=zJvrrusAAAAJ&view_op=list_works&sortby=pubdate)<sup>2</sup>, [Jiayi Guo](https://www.jiayiguo.net)<sup>1</sup>, [Yue Ma](https://mayuelala.github.io/)<sup>3</sup>, <br> [Nisha Huang](https://scholar.google.com/citations?user=wTmPkSsAAAAJ&hl=en)<sup>1</sup>, [Yuxin Chen](https://scholar.google.com/citations?hl=en&user=dEm4OKAAAAAJ)<sup>2</sup>, [Xiu Li](https://scholar.google.com/citations?user=Xrh1OIUAAAAJ&hl=en&oi=ao)<sup>1</sup>, [Ying Shan](https://scholar.google.com/citations?hl=en&user=4oXBp9UAAAAJ&view_op=list_works&sortby=pubdate)<sup>2</sup>

<sup>1</sup> Tsinghua University,  <sup>2</sup> Tencent ARC Lab,  <sup>3</sup> HKUST  

[![arXiv](https://img.shields.io/badge/arXiv-RFSolverEdit-b31b1b.svg)](https://arxiv.org/abs/2411.04746)
<a href='https://rf-solver-edit.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
[![Huggingface space](https://img.shields.io/badge/🤗-Huggingface%20Space-orange.svg)](https://huggingface.co/spaces/wjs0725/RF-Solver-Edit) 
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Demo-blue.svg)](https://github.com/logtd/ComfyUI-Fluxtapoz) 
</div>





<p>
We propose <strong>RF-Solver</strong> to solve the rectified flow ODE with less error, thus enhancing both sampling quality and inversion-reconstruction accuracy for rectified-flow-based generative models. Furthermore, we propose <strong>RF-Edit</strong> to leverage the <strong>RF-Solver</strong> for image and video editing tasks. Our methods achieve impressive performance on various tasks, including text-to-image generation, image/video inversion, and image/video editing. 
</p>



<p align="center">
<img src="assets/repo_figures/Picture1.jpg" width="1080px"/>
</p>

# 🔥 News
- [2024.11.30] Our demo is available on 🤗 [Huggingface Space](https://huggingface.co/spaces/wjs0725/RF-Solver-Edit)!
- [2024.11.18] More examples for style transfer are available!
- [2024.11.18] Gradio Demo for image editing is available!
- [2024.11.16] Thanks to @[logtd](https://github.com/logtd) for integrating RF-Solver into [ComfyUI](https://github.com/logtd/ComfyUI-Fluxtapoz)! 
- [2024.11.11] The [homepage](https://rf-solver-edit.github.io/) of the project is available!
- [2024.11.08] Code for image editing is released!
- [2024.11.08] Paper released!

# 👨‍💻 ToDo
- ☑️ Release the gradio demo
- ☑️ Release scripts for more image editing cases
- ☐ Release the code for video editing


# 📖 Method
## RF-Solver
<p>
<img src="assets/repo_figures/Picture2.jpg" width="1080px"/>
We derive the exact formulation of the solution for Rectified Flow ODE. The non-linear part in this solution is processed by Taylor Expansion. Through higher order expansion, the approximation error in the solution is significantly reduced, thus achieving impressive performance on both text-to-image sampling and image/video inversion.
</p>

## RF-Edit
<p>
<img src="assets/repo_figures/Picture3.jpg" width="1080px"/>
Based on RF-Solver, we further propose the RF-Edit for image and video editing. RF-Edit framework leverages the features from inversion in the denoising process, which enables high-quality editing while preserving the structural information of source image/video. RF-Edit contains two sub-modules, especially for image editing and video editing.
</p>

# 🛠️ Code Setup
The environment of our code is the same as FLUX, you can refer to the [official repo](https://github.com/black-forest-labs/flux/tree/main) of FLUX, or running the following command to construct the environment.
```
conda create --name RF-Solver-Edit python=3.10
conda activate RF-Solver-Edit
pip install -e ".[all]"
```
# 🚀 Examples for Image Editing
We have provided several scripts to reproduce the results in the paper, mainly including 3 types of editing: Stylization, Adding, Replacing. We suggest to run the experiment on a single A100 GPU.

## Stylization
<table class="center">
<tr>
  <td width=10% align="center">Ref Style</td>
  <td width=30% align="center"><img src="assets/repo_figures/examples/source/nobel.jpg" raw=true></td>
	<td width=30% align="center"><img src="assets/repo_figures/examples/source/art.jpg" raw=true></td>
  <td width=30% align="center"><img src="assets/repo_figures/examples/source/cartoon.jpg" raw=true></td>
</tr>
<tr>
  <td width="10%" align="center">Editing Scripts</td>
  <td width="30%" align="center"><a href="src/run_nobel_trump.sh">Trump</a></td>
  <td width="30%" align="center"><a href="src/run_art_mari.sh"> Marilyn Monroe</a></td>
  <td width="30%" align="center"><a href="src/run_cartoon_ein.sh">Einstein</a></td>
</tr>
<tr>
  <td width=10% align="center">Edtied image</td>
  <td width=30% align="center"><img src="assets/repo_figures/examples/edit/nobel_Trump.jpg" raw=true></td>
	<td width=30% align="center"><img src="assets/repo_figures/examples/edit/art_mari.jpg" raw=true></td>
  <td width=30% align="center"><img src="assets/repo_figures/examples/edit/cartoon_ein.jpg" raw=true></td>
</tr>

<tr>
  <td width="10%" align="center">Editing Scripts</td>
  <td width="30%" align="center"><a href="src/run_nobel_biden.sh">Biden</a></td>
  <td width="30%" align="center"><a href="src/run_art_batman.sh">Batman</a></td>
  <td width="30%" align="center"><a href="src/run_cartoon_herry.sh">Herry Potter</a></td>
</tr>
<tr>
  <td width=10% align="center">Edtied image</td>
  <td width=30% align="center"><img src="assets/repo_figures/examples/edit/nobel_Biden.jpg" raw=true></td>
	<td width=30% align="center"><img src="assets/repo_figures/examples/edit/art_batman.jpg" raw=true></td>
  <td width=30% align="center"><img src="assets/repo_figures/examples/edit/cartoon_herry.jpg" raw=true></td>
</tr>
</table>

## Adding & Replacing
<table class="center">
<tr>
  <td width=10% align="center">Source image</td>
  <td width=30% align="center"><img src="assets/repo_figures/examples/source/hiking.jpg" raw=true></td>
	<td width=30% align="center"><img src="assets/repo_figures/examples/source/horse.jpg" raw=true></td>
  <td width=30% align="center"><img src="assets/repo_figures/examples/source/boy.jpg" raw=true></td>
</tr>
<tr>
  <td width="10%" align="center">Editing Scripts</td>
  <td width="30%" align="center"><a href="src/run_hiking.sh">+ hiking stick</a></td>
  <td width="30%" align="center"><a href="src/run_horse.sh">horse -> camel</a></td>
  <td width="30%" align="center"><a href="src/run_boy.sh">+ dog</a></td>
</tr>
<tr>
  <td width=10% align="center">Edtied image</td>
  <td width=30% align="center"><img src="assets/repo_figures/examples/edit/hiking.jpg" raw=true></td>
	<td width=30% align="center"><img src="assets/repo_figures/examples/edit/horse.jpg" raw=true></td>
  <td width=30% align="center"><img src="assets/repo_figures/examples/edit/boy.jpg" raw=true></td>
</tr>

</table>


# 🪄 Edit Your Own Image

## Gradio Demo
We provide the gradio demo for image editing, which is also available on our 🤗 [Huggingface Space](https://huggingface.co/spaces/wjs0725/RF-Solver-Edit)! You can also run the gradio demo on your own device using the following command: 
```
cd src
python gradio_demo.py
```
Here is an example of using the gradio demo to edit an image! Note that here "Number of inject steps" means the steps of feature sharing in RF-Edit, which is highly related to the quality of edited results. We suggest tuning this parameter, and selecting the results with the best visual quality.
<div style="text-align: center;">
  <img src="assets/repo_figures/Picture7.jpg" style="width:100%; display: block; margin: 0 auto;" />
</div>


## Command Line
You can also run the following scripts to edit your own image. 
```
cd src
python edit.py  --source_prompt [describe the content of your image or leave it as null] \
                --target_prompt [describe your editing requirements] \
                --guidance 2 \
                --source_img_dir [the path of your source image] \
                --num_steps 30  \
                --inject [typically set to a number between 2 to 8] \
                --name 'flux-dev' --offload \
                --output_dir [output path] 
```
Similarly, The ```--inject``` refers to the steps of feature sharing in RF-Edit, which is highly related to the performance of editing. 



# 🖼️ Gallery
## Inversion and Reconstruction  

<p align="center">
<img src="assets/repo_figures/Picture4.jpg" width="1080px"/>
</p>

## Image Stylization

<p align="center">
<img src="assets/repo_figures/Picture8.jpg" width="1080px"/>
</p>

## Image Editing

<p align="center">
<img src="assets/repo_figures/Picture5.jpg" width="1080px"/>
</p>

## Video Editing

<p align="center">
<img src="assets/repo_figures/Picture6.jpg" width="1080px"/>
</p>

# 🖋️ Citation

If you find our work helpful, please **star 🌟** this repo and **cite 📑** our paper. Thanks for your support!

```
@article{wang2024taming,
  title={Taming Rectified Flow for Inversion and Editing},
  author={Wang, Jiangshan and Pu, Junfu and Qi, Zhongang and Guo, Jiayi and Ma, Yue and Huang, Nisha and Chen, Yuxin and Li, Xiu and Shan, Ying},
  journal={arXiv preprint arXiv:2411.04746},
  year={2024}
}
```

# Acknowledgements
We thank [FLUX](https://github.com/black-forest-labs/flux/tree/main) for their clean codebase.

# Contact
The code in this repository is still being reorganized. Errors that may arise during the organizing process could lead to code malfunctions or discrepancies from the original research results. If you have any questions or concerns, please send emails to wjs23@mails.tsinghua.edu.cn.
