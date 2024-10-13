# Causal Image Modeling for Efficient Visual Understanding
*Official PyTorch implementation of Adventurer, linear-time image models with causal modeling paradigm.*
*Arxiv: https://arxiv.org/pdf/2410.07599*

<img src="images\framework.png" width="80%" />

## Release
- [Oct.13.2024] 📢 Code and model weights released.

## Models
| Model              | Input Size | IN-1k Top-1 Acc. | Checkpoint                                                   |
| ------------------ | ---------- | ---------------- | ------------------------------------------------------------ |
| Adventurer-Tiny    | 224        | 78.2             | [Adventurer_tiny_patch16_224](https://huggingface.co/Wangf3014/Adventurer/resolve/main/adventurer_tiny_patch16_224.pth) |
| Adventurer-Small   | 224        | 81.8             | [Adventurer_small_patch16_224](https://huggingface.co/Wangf3014/Adventurer/resolve/main/adventurer_small_patch16_224.pth) |
| Adventurer-Base    | 224        | 82.6             | [Adventurer_base_patch16_224](https://huggingface.co/Wangf3014/Adventurer/resolve/main/adventurer_base_patch16_224.pth) |
| Adventurer-Large   | 224        | 83.4             | [Adventurer_large_patch16_224](https://huggingface.co/Wangf3014/Adventurer/resolve/main/adventurer_large_patch16_224.pth) |
| Adventurer-Base    | 384        | 83.9             | [Adventurer_base_patch16_384](https://huggingface.co/Wangf3014/Adventurer/resolve/main/adventurer_base_patch16_384.pth) |
| Adventurer-Base/P8 | 224        | 83.9             | [Adventurer_base_patch8_224](https://huggingface.co/Wangf3014/Adventurer/resolve/main/adventurer_base_patch8_224.pth) |
| Adventurer-Base/P8 | 448        | 84.8             | [Adventurer_base_patch8_448](https://huggingface.co/Wangf3014/Adventurer/resolve/main/adventurer_base_patch8_448.pth) |

- We retrained the Adventurer-Base/P8 model and got a slightly higher result than that in the paper (84.6 -> 84.8)
- We additionally provided a Base model at 384 input and a Base/P8 model at 224. Both have an 83.9 accuracy.

## Install
- Prepare your environment
```bash
conda create -n adventurer python=3.10
source activate adventurer
```
- Install Pytorch with CUDA version >= 11.8
```bash
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
```
- Other dependencies, wandb and submitit are optinal
```bash
pip install timm==0.4.12 mlflow==2.9.1 setuptools==69.5.1 wandb submitit
```
-Install causal-conv1d and mamba-2
```bash
pip install causal-conv1d==1.2.1
pip install mamba-ssm==2.0.4
```

## Evaluation
- Evaluate models with a 16*16 patch size:
```python
python -m torch.distributed.launch --nproc_per_node=1  --use_env main.py \
    --model adventurer_base_patch16 --input-szie 224 \
    --data-path /PATH/TO/IMAGENET --batch 128 \
    --resume /PATH/TO/CHECKPOINT \
    --eval --eval-crop-ratio 0.875
```
- Evaluate models with an 8*8 patch size:
```python
python -m torch.distributed.launch --nproc_per_node=1  --use_env main.py \
    --model adventurer_base_patch8 --input-szie 448 \
    --data-path /PATH/TO/IMAGENET --batch 128 \
    --resume /PATH/TO/CHECKPOINT \
    --eval --eval-crop-ratio 1.0
```

## Training
- Here we provide single-node, 8-GPU training scripts. For multi-node training, we have integerated the codebase with [submitit](https://github.com/facebookincubator/submitit), which allows conveniently launching distributed jobs on [Slurm](https://slurm.schedmd.com/quickstart.html) clusters. See [Multi-node training](README_MULTI_NODE.md) for more details.
- Stage one, pretrain with 128*128 inputs
```python
python -m torch.distributed.launch --nproc_per_node=8  --use_env main.py \
```
