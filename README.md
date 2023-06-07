# RWKV-v4neo Training Instructions

This README provides instructions on how to set up and run the RWKV-v4neo model, based on the source provided in the [BlinkDL's RWKV-LM repository](https://github.com/BlinkDL/RWKV-LM). The specific code was adapted from the [RWKV-v4neo](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v4neo) subdirectory.

## Environment Setup

Follow these steps to set up your environment:

1. Create a new conda environment with Python 3.10:
    ```shell
    conda create -n rwkv python=3.10
    ```
2. Activate the new conda environment:
    ```shell
    conda activate rwkv
    ```
3. Install the required dependencies:
    ```shell
    pip install -r requirements.txt
    ```
4. Login to Weights & Biases (if you want to use wandb):
    ```shell
    wandb login
    ```

## Training the Model

To train the model, run the following command:

```shell
python train.py \
    --load_model "" \
    --wandb "rwkv-v4neo" \
    --proj_dir "out" \
    --data_file "" \
    --data_type "dummy" \
    --vocab_size 0 \
    --ctx_len 128 \
    --epoch_steps 1000 \
    --epoch_count 20 \
    --epoch_begin 0 \
    --epoch_save 10 \
    --micro_bsz 16 \
    --n_layer 12 \
    --n_embd 768 \
    --pre_ffn 0 \
    --head_qk 0 \
    --lr_init 6e-4 \
    --lr_final 1e-5 \
    --warmup_steps 0 \
    --beta1 0.9 \
    --beta2 0.99 \
    --adam_eps 1e-8 \
    --accelerator gpu \
    --devices 1 \
    --precision bf16 \
    --strategy ddp_find_unused_parameters_false \
    --grad_cp 0
