## Multi-node training

The simplest way to run multi-node jobs in [Slurm](https://slurm.schedmd.com/quickstart.html) is to use [submitit](https://github.com/facebookincubator/submitit). Below is an example to run a 2-node, 16-GPU job:
```bash
python run_with_submitit.py \
    --job_dir YOUR_JOB_DIR \
    --ngpus 8 --nodes 2 --partition YOUR_PARTITION --timeout 3000 \
    --model adventurer_base_patch16 \
    --data-path /PATH/TO/IMAGENET \
    --batch 128 --lr 5e-4 --weight-decay 0.05 \
    --reprob 0.0 --smoothing 0.0  --repeated-aug --ThreeAugment \
    --epochs 300 --input-size 128 --drop-path 0.1 --dist-eval  
```

Alternatively, you can launch N jobs on N machines respectively, with specified node ranks:
```bash
python -m torch.distributed.launch --nproc_per_node=8 --nnodes 2 --node_rank 0 --master_addr ADDR_OF_ONE_OF_THE_NODES --use_env main.py ...
```
