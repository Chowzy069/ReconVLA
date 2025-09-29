# ReconVLA: Reconstructive Vision-Language-Action Model as Effective Robot Perceiver
<a href="https://arxiv.org/abs/2508.10333" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/ReconVLA-red?label=arXiv&color=red" height="25" />
</a>
<a href="https://zionchow.github.io/ReconVLA/" target="_blank">
    <img alt="project" src="https://img.shields.io/badge/ReconVLA-blue?label=Project&color=blue" height="25" />
</a>
<a href="https://huggingface.co/zzyzyzy/ReconVLA" target="_blank">
    <img alt="HF Model: ReconVLA" src="https://img.shields.io/badge/ReconVLA-yellow?label=Model(consistent-training-free)&color=ffd400" height="25" />
</a>

We present ReconVLA, an implicit grounding paradigm for Vision-Language-Action models that reconstructs gaze regions to focus visual attention, achieving precise manipulation and strong generalization with only 100k+ trajectories. Key contributions include:
- **Implicit Grounding Architecture**: Reconstructive VLA paradigm that aligns gaze regions with manipulated targets, enforcing precise visual attention and fine-grained representation learning.
- **Large-scale Pretraining Foundation**:100k+ trajectory dataset (2M+ samples) boosting generalization of visual reconstruction capabilities.

## 📊 Overview
![teaser](./figs/arch.jpg)
Our model consists of a reconstructive part and an action part. The input includes multi-view images and a text instruction. For the action part, the model outputs discrete action tokens. For the reconstruction part, Reconvla is guided to output reconstructive tokens, which are conditions of the denoising process to reconstruct the scene tokens $z_0$ from noisy $z_t$. The scene tokens are tokenized images of gaze regions. This supervision enables Reconvla to enhance visual grounding and fine-grained comprehension capabilities, which contribute to precise manipulation.



## Getting Started <a name="installation"></a>


### Clone the Repository
```bash
git clone https://github.com/Chowzy069/Reconvla.git
cd ReconVLA
```


We use conda to manage the environment.

```bash
conda create -n reconvla python=3.10.16
conda activate reconvla
pip install -r recon_requirements.txt

cd reconvla
```


## Data Preparation
This project ships with no raw data.
Please download the three public datasets—[BridgeData V2](https://rail.eecs.berkeley.edu/datasets/bridge_release/data/), [LIBERO](https://libero-project.github.io/datasets), and [CALVIN](https://github.com/mees/calvin)—and preprocess them into the format described in the paper before running any training or evaluation scripts.
Please replace the folder `calvin_models/calvin_agent/evaluation` with `reconvla/evaluation`.

### CALVIN
#### Clone and install CALVIN
```bash
git clone --recurse-submodules https://github.com/mees/calvin.git
export CALVIN_ROOT=$(pwd)/calvin
cd $CALVIN_ROOT
conda create -n calvin_venv python=3.8  
conda activate calvin_venv
sh install.sh
```

#### Download CALVIN dataset
```bash
cd $CALVIN_ROOT/dataset
sh download_data.sh ABC
```
Please note that the `numpy` version = 1.23.5!

#### Preprocess CALVIN dataset. 
This step will output a JSON file formatted for VLA training and a processed folder containing stitched images. You can manually modify the save path, but please ensure to use the data from the correct path during training/testing.
You must have the preprocessed `target_image` ready in advance.

**Step 1: Extract tasks from CALVIN dataset**
```bash
cd reconvla/reconvla
python ./scripts/helper/calvin_extract_task.py \
    --ann_path /path/to/auto_lang_ann.npy \
    --npz_src_dir /path/to/training/ \
    --root_folder /output/path/
```
Below is an explanation of the parameters：
- `ann_path`: Path to auto_lang_ann.npy file.
- `npz_src_dir`: Source directory containing episode NPZ files.
- `root_folder`: Output root folder for extracted task.

**Input folder structure:**

**Output folder structure after extraction:**
```
output_folder/
├── 0_task_name_1/
│   ├── lang_ann/
│   │   └── lang_ann.yaml  # Task annotation and frame indices
│   └── img/
│       ├── frame_0000000.png
│       ├── frame_0000001.png
│       └── ...
├── 1_task_name_2/
│   └── ...
```
**Step 2: Generate target_image**

Generate target images using object detection and grounding methods such as GroundingDINO, YOLO, etc. These target images represent the gaze regions or objects of interest that the model should focus on during manipulation tasks.
```
output_folder/
├── 0_task_name_1/
│   ├── lang_ann/
│   │   └── lang_ann.yaml  
│   └── img/
│       ├── frame_0000000.png
│       └── ...
│   └── crop/
│       ├── frame_0000000.png
│       └── ...
```

**Step 3: Generate training JSON**
```bash
python ./scripts/helper/calvin_json.py \
    --calvin_original_data_path /path/to/original/calvin/ \
    --calvin_crop_data_path /path/to/extracted/tasks/ \
    --calvin_processed_directory /path/to/processed/images/ \
    --calvin_processed_json_path /path/to/output.json
```
Below is an explanation of the parameters：
- `calvin_original_data_path`: Path to the original calvin dataset directory.
- `calvin_crop_data_path`: Path to the crop dataset directory.
- `calvin_processed_directory`: Path to the calvin processed directory.
- `calvin_processed_json_path`: Path to the calvin processed json file.


## 📈 Training
Reconvla is trained on 8 A100 GPUs with 80GB memory. To train on fewer GPUs, you can reduce the per_device_train_batch_size and increase the gradient_accumulation_steps accordingly. If you want to train from the checkpoint, always keep the global batch size the same: per_device_train_batch_size x gradient_accumulation_steps x num_gpus.
If you have multiple GPUs and wish to use PyTorch's Distributed Data Parallel, simply set the number in the command below to match the number of available GPUs (CUDA_VISIBLE_DEVICES and localhost).

### Pretraining of Generalist Policy

```bash
conda activate reconvla
cd reconvla/reconvla
bash scripts/train_vla/pretrain.sh
```
### Fine-tune

```bash
conda activate reconvla
cd reconvla/reconvla
bash scripts/train_vla/finetune.sh
```

Below is an explanation of the most commonly adjusted training parameters：
- `model_name_or_path`: Path or name of the pre-trained language model.
- `data_path`: Path to the JSON file containing training data.
- `action_stat`: Path to action normalization statistics.
- `num_train_epochs`: Size of action discretization bins.
- `per_device_train_batch_size`: Training batch size per GPU.
- `image_aspect_ratio`: Image processing method.
- `num_train_epochs`: otal number of training rounds.
- `use_diffusion_head`:use difussion head for decode

## 🔬 Evaluation
First, run the Reconvla policy evaluation script:
```bash
conda activate reconvla
cd reconvla/reconvla
bash scripts/test_vla/start_multi_server.sh
```
Below is an explanation of the most commonly adjusted parameters:
- `dataset_path`: Path to the root directory of the dataset.
- `question_file`: Path to JSON file containing task descriptions or questions.
- `num_chunks`: Number of chunks to split tasks into for parallel processing.
- `chunk_idx`: Index of current chunk.
- `save_dir`: Directory to save inference results.
- `num_chunk`: Length of the action sequence generated per chunk.
- `conf_dir`: Directory containing configuration files.


In the second Terminal window,  run the robot server:
```
conda activate caivin_venv
cd reconvla/calvin/calvin_agent/evaluation
bash evaluate_policy_multiserver.sh
```
Start model server on you own port (here is 9097)，
CUDA_VISIBLE_DEVICES specifies the number of GPUs (e.g., if you have two GPUs, it would be 0,1).

Below is an explanation of the most commonly adjusted parameters:
- `model_path`: Path to the model checkpoint.
- `action_stat`: Action normalization stats.

## Contact

For further discussion and collaboration, please feel free to contact us via WeChat:

<div align="left">
<table>
<tr>
<td align="center">
<strong>Wenxuan Song</strong><br>
<img src="./figs/qr/wx.jpg" alt="Wenxuan Song WeChat QR Code" width="150"/>
</td>
<td align="center">
<strong>Ziyang Zhou</strong><br>
<img src="./figs/qr/zy.jpg" alt="Ziyang Zhou WeChat QR Code" width="154"/>
</td>
</tr>
</table>
</div>


## 📑Citation

If you find this work useful, please cite:

```bibtex
@article{song2025reconvla,
  title={ReconVLA: Reconstructive Vision-Language-Action Model as Effective Robot Perceiver},
  author={Song, Wenxuan and Zhou, Ziyang and Zhao, Han and Chen, Jiayi and Ding, Pengxiang and Yan, Haodong and Huang, Yuxin and Tang, Feilong and Wang, Donglin and Li, Haoang},
  journal={arXiv preprint arXiv:2508.10333},
  year={2025}
}
```
