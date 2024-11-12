<div align="center">
  
# Taming Rectified Flow for Inversion and Editing

[Jiangshan Wang](https://scholar.google.com/citations?user=HoKoCv0AAAAJ&hl=en), [Junfu Pu](https://pujunfu.github.io/), [Zhongang Qi](https://scholar.google.com/citations?hl=en&user=zJvrrusAAAAJ&view_op=list_works&sortby=pubdate), [Jiayi Guo](https://www.jiayiguo.net), [Yue Ma](https://mayuelala.github.io/), <br> [Nisha Huang](https://scholar.google.com/citations?user=wTmPkSsAAAAJ&hl=en), [Yuxin Chen](https://scholar.google.com/citations?hl=en&user=dEm4OKAAAAAJ), [Xiu Li](https://scholar.google.com/citations?user=Xrh1OIUAAAAJ&hl=en&oi=ao), [Ying Shan](https://scholar.google.com/citations?hl=en&user=4oXBp9UAAAAJ&view_op=list_works&sortby=pubdate)

[![arXiv](https://img.shields.io/badge/arXiv-RFSolverEdit-b31b1b.svg)](https://arxiv.org/abs/2411.04746)
<a href='https://rf-solver-edit.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>

</div>





<p>
We propose <strong>RF-Solver</strong> to solve the rectified flow ODE with less error, thus enhancing both sampling quality and inversion-reconstruction accuracy for rectified-flow-based generative models. Furthermore, we propose <strong>RF-Edit</strong> to leverage the <strong>RF-Solver</strong> for image and video editing tasks. Our methods achieve impressive performance on various tasks, including text-to-image generation, image/video inversion, and image/video editing. 
</p>



<p align="center">
<img src="assets/repo_figures/Picture1.jpg" width="1080px"/>
</p>

## 🔥 News
- [2024.11.11] The homepage of the project is avaible!
- [2024.11.08] Code for image editing is released!
- [2024.11.08] Paper released!

## 👨‍💻 ToDo
- ☐ Release the gradio demo
- ☐ Release scripts to for more image editing cases
- ☐ Release the code for video editing


## 📖 Method
### RF-Solver
<p>
<img src="assets/repo_figures/Picture2.jpg" width="1080px"/>
We derive the exact fomulation of the solution for Rectified Flow ODE. The non-linear part in this solution is processed by Taylor Expansion. Through higher order expansion, the approximation error in the solution is significantly reduced, thus achieving impressive performance on both text-to-image sampling and image/video inversion.
</p>

### RF-Edit
<p>
<img src="assets/repo_figures/Picture3.jpg" width="1080px"/>
Based on RF-Solver, we further propose the RF-Edit for image and video editing. RF-Edit framework leverages the features from inversion in the denoising process, which enables high-quality editing while preserving the structual information of source image/video. RF-Edit contains two sub-modules, espectively for image editing and video editing.
</p>

## 🔨 Code
### Setup
The environment of our code is the same as FLUX, you can refer to the [official repo](https://github.com/black-forest-labs/flux/tree/main) of FLUX, or running the following command to construct the environment.
```
conda create --name RF-Solver-Edit python=3.10
conda activate RF-Solver-Edit
pip install -e ".[all]"
```

### Image Editing
We have provided several scripts to reproduce the results in the paper. The resolution of following images is 1360*768. We suggest to run the experiment on a single A100 GPU
<table class="center">
<tr>
  <td width=10% align="center">Source image</td>
  <td width=30% align="center"><img src="assets/repo_figures/examples/source/hiking.jpg" raw=true></td>
	<td width=30% align="center"><img src="assets/repo_figures/examples/source/horse.jpg" raw=true></td>
  <td width=30% align="center"><img src="assets/repo_figures/examples/source/boy.jpg" raw=true></td>
</tr>
<tr>
  <td width="10%" align="center">Editing Scripts</td>
  <td width="30%" align="center"><a href="src/boy.sh">+ hiking stick</a></td>
  <td width="30%" align="center"><a href="src/horse.sh">horse -> camel</a></td>
  <td width="30%" align="center"><a href="src/boy.sh">+ dog</a></td>
</tr>
<tr>
  <td width=10% align="center">Edtied image</td>
  <td width=30% align="center"><img src="assets/repo_figures/examples/edit/hiking.jpg" raw=true></td>
	<td width=30% align="center"><img src="assets/repo_figures/examples/edit/horse.jpg" raw=true></td>
  <td width=30% align="center"><img src="assets/repo_figures/examples/edit/boy.jpg" raw=true></td>
</tr>

</table>

### Edit Your Own Image
You can run the following scripts to edit your own image. The ```--inject``` refers to the steps of feature sharing in RF-Edit, which is highly related to the performance of editing. We suggest to tune this hyper-parameter from 2 to 8, selecting the results with best visual quality.
```
python edit.py  --source_prompt [describe the content of your image or leaves it as null] \
                --target_prompt [describe your editing requirements] \
                --guidance 2 \
                --source_img_dir [the path of your source image] \
                --num_steps 30  \
                --inject [typically set to a number between 2 to 8] \
                --name 'flux-dev' --offload \
                --output_dir [output path] 
```


## 🖼️ Gallery
### Inversion and Reconstruction  

<p align="center">
<img src="assets/repo_figures/Picture4.jpg" width="1080px"/>
</p>

### Image Editing

<p align="center">
<img src="assets/repo_figures/Picture5.jpg" width="1080px"/>
</p>

### Video Editing

<p align="center">
<img src="assets/repo_figures/Picture6.jpg" width="1080px"/>
</p>

## 🖋️ Citation

If you find our work helpful, please **star 🌟** this repo and **cite 📑** our paper. Thanks for your support!

```
@misc{wang2024tamingrectifiedflowinversion,
      title={Taming Rectified Flow for Inversion and Editing}, 
      author={Jiangshan Wang and Junfu Pu and Zhongang Qi and Jiayi Guo and Yue Ma and Nisha Huang and Yuxin Chen and Xiu Li and Ying Shan},
      year={2024},
      eprint={2411.04746},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.04746}, 
}
```

## Acknowledgements
We thank [FLUX](https://github.com/black-forest-labs/flux/tree/main) for their clean codebase.

## Contact
The code in this repository is still being reorganized. Errors that may arise during the organizing process could lead to code malfunctions or discrepancies from the original research results. If you have any questions or concerns, please send email to wjs23@mails.tsinghua.edu.cn.