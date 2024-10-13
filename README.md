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
| Adventurer-Base/P8 | 224        | 83.9             | [Adventurer_base_patch8_224](https://huggingface.co/Wangf3014/Adventurer/resolve/main/adventurer_base_patch8_224.pth) |
| Adventurer-Base    | 384        | 84.2             | [Adventurer_base_patch16_384](https://huggingface.co/Wangf3014/Adventurer/resolve/main/adventurer_base_patch16_384.pth) |
| Adventurer-Base    | 448        | 84.3             | [Adventurer_base_patch16_448](https://huggingface.co/Wangf3014/Adventurer/resolve/main/adventurer_base_patch16_448.pth) |
| Adventurer-Base/P8 | 448        | 84.8             | [Adventurer_base_patch8_448](https://huggingface.co/Wangf3014/Adventurer/resolve/main/adventurer_base_patch8_448.pth) |

- We retrained the Adventurer-Base/P8 model at 448 input and got a slightly higher result than that in the paper (84.6 -> 84.8).
- We additionally provided a Base/P8 model at 224, which obtains an 83.9 accuracy. The Base/P8-448 is fintuned from this model.
- We additionally provided a Base model at 384 input, which obtains an 84.2 accuracy.
- We retrained the Base model at 448 and also got a higher result (84.0 -> 84.3).
- **We actually have many different models scaling the patch and input sizes. We cannot upload all of them, but welcome to leave me a message so that I can send the checkpoints you are interested.**

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
- Evaluate models at a regular input and patch size, i.e., patch16 with input224:
```python
python -m torch.distributed.launch --nproc_per_node=1  --use_env main.py \
    --model adventurer_base_patch16 --input-szie 224 \
    --data-path /PATH/TO/IMAGENET --batch 128 \
    --resume /PATH/TO/CHECKPOINT \
    --eval --eval-crop-ratio 0.875
```
- For models with patch sizes<16 or input sizes>224, use --eval-crop-ratio 1.0:
```python
python -m torch.distributed.launch --nproc_per_node=1  --use_env main.py \
    --model adventurer_base_patch8 --input-szie 448 \
    --data-path /PATH/TO/IMAGENET --batch 128 \
    --resume /PATH/TO/CHECKPOINT \
    --eval --eval-crop-ratio 1.0
```

## Training
- We basically follow the multi-stage training strategy of [Mamba-Reg](https://github.com/wangf3014/Mamba-Reg).
- Here we provide single-node, 8-GPU training scripts. For multi-node training, we have integerated the codebase with [submitit](https://github.com/facebookincubator/submitit), which allows conveniently launching distributed jobs on [Slurm](https://slurm.schedmd.com/quickstart.html) clusters. See [Multi-node training](README_MULTI_NODE.md) for more details.
- Stage one, pretrain with 128*128 inputs
```python
python -m torch.distributed.launch --nproc_per_node=8  --use_env main.py \
    --model adventurer_base_patch16 \
    --data-path /PATH/TO/IMAGENET \
    --batch 128 --lr 5e-4 --weight-decay 0.05 \
    --output_dir ./output/adventurer_base_patch16_224/s1_128 \
    --reprob 0.0 --smoothing 0.0  --repeated-aug --ThreeAugment \
    --epochs 300 --input-size 128 --drop-path 0.1 --dist-eval
```
- Stage two, train with 224*224 inputs
```python
python -m torch.distributed.launch --nproc_per_node=8  --use_env main.py \
    --model adventurer_base_patch16 \
    --data-path /PATH/TO/IMAGENET \
    --batch 128 --lr 5e-4 --weight-decay 0.05 \
    --finetune ./output/adventurer_base_patch16_224/s1_128/checkpoint.pth
    --output_dir ./output/adventurer_base_patch16_224/s2_224 \
    --reprob 0.0 --smoothing 0.0  --repeated-aug --ThreeAugment \
    --epochs 100 --input-size 224 --drop-path 0.4 --dist-eval
```
- Stage three, finetune with 224*224 inputs
```python
python -m torch.distributed.launch --nproc_per_node=8  --use_env main.py \
    --model adventurer_base_patch16 \
    --data-path /PATH/TO/IMAGENET \
    --batch 64 --lr 1e-5 --weight-decay 0.1 --unscale-lr \
    --finetune ./output/adventurer_base_patch16_224/s2_224/checkpoint.pth
    --output_dir ./output/adventurer_base_patch16_224/s3_224 \
    --reprob 0.0 --smoothing 0.1 --no-repeated-aug --aa rand-m9-mstd0.5-inc1 \
    --epochs 20 --input-size 224 --drop-path 0.6 --dist-eval
```
