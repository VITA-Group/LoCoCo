# LoCoCo: Dropping In Convolutions for Long Context Compression
[Ruisi Cai](https://cairuisi.github.io/)<sup>1</sup>,
[Yuandong Tian](https://yuandong-tian.com/)<sup>2</sup>,
[Zhangyang Wang](https://vita-group.github.io/)<sup>1</sup>,
[Beidi Chen](https://www.andrew.cmu.edu/user/beidic/)<sup>3</sup>,

<sup>1</sup>University of Texas at Austin, <sup>2</sup>Meta AI (FAIR), <sup>3</sup>Carnegie Mellon University

## Usage 
```
python train.py \
    --dataset_name togethercomputer/RedPajama-Data-1T-Sample \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --block_size 512 \
    --clean_period 8 \
    --method conv \
    --kernel_size 21 \
    --n_convlayer 1 \
    --mem_size 512 \
    --max_train_steps 1000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 128 \
    --eval_iter 20 \
    --eval_interval 50 \
    --stream_tokenizer \
    --normalizer_init 0.5 \
    --memory_lr_scale 1000 \
    --norm_lr_scale 5 \
    --rope_change \
    --checkpointing_steps 100 \
    --output_dir output/no_extend/rp_{block_size}_{clean_period}_mem{mem_size}/{method}/ \
    --auto_resume 
```

The model checkpoints is coming soon!

## Citation
If you find this useful, please cite the following paper:
```
@article{cai2024lococo,
  title={LoCoCo: Dropping In Convolutions for Long Context Compression},
  author={Cai, Ruisi and Tian, Yuandong and Wang, Zhangyang and Chen, Beidi},
  journal={arXiv preprint arXiv:2406.05317},
  year={2024}
}
```
