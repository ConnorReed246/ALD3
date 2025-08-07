# Efficient Sampling of Diffusion Models through Adaptive Timesteps

Diffusion models are among the best-performing generative models but are constrained by high
computational costs. Previous work shows that optimizing timestep schedules with respect to factors
such as the dataset, solver, and diffusion model can improve sampling efficiency and output quality.
However, these schedules are static and do not adapt to individual diffusion processes.
This work empirically shows that per-sample timestep schedules exist that can improve FID scores
by up to 45.8% with an average of 31.4% over a globally optimal baseline on CIFAR-10. Computing
such schedules efficiently is non-trivial due to the problem’s non-convexity and a tendency to
converge to the global baseline. Despite this, up to a 12.1% FID reduction is achieved with a novel
one-shot Timestep Prediction Model, while a recursive extension yields improvements of up to
18.2%. Both approaches where trained on theoretically optimal schedules and incur negligible
computational overhead during sampling.

## Setup
```bash
conda env create -f requirements.yml
conda activate ld3
pip install -e ./src/clip/
pip install -e ./src/taming-transformers/
pip install omegaconf
pip install PyYAML
pip install requests
pip install scipy
pip install torchmetrics
```

## Download data
Notice that all data will be downloaded by the script, which might take time. Skip ones by commenting out.
```bash
bash scripts/download_model.sh
wget https://raw.githubusercontent.com/tylin/coco-caption/master/annotations/captions_val2014.json
``` 

## Running
### Generate training images 
```bash
CUDA_VISIBLE_DEVICES=0 python3 gen_data.py \
                    --all_config configs/cifar10.yml \
                    --total_samples 100 \
                    --sampling_batch_size 10 \
                    --steps 20 \
                    --solver_name uni_pc \
                    --skip_type edm \
                    --save_pt --save_png --data_dir train_data/train_data_cifar10 \
                    --low_gpu
```
### Training LD3*
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py \
--all_config configs/cifar10.yml \
--data_dir train_data/train_data_cifar10/uni_pc_NFE20_edm_seed0 \
--num_train 50 \
--num_valid 50 \
--main_train_batch_size 1 \
--main_valid_batch_size 10 \
--training_rounds_v1 2 \
--log_path logs/logs_cifar10 \
--force_train True \
--steps 10 
```

### RQ1: Find optimal timestep schedules
```bash
CUDA_VISIBLE_DEVICES=0 python3 gen_optimal_timesteps.py \
--all_config configs/cifar10.yml \
--data_dir train_data/train_data_cifar10/uni_pc_NFE20_edm_seed0 \
--num_train 20 \
--num_valid 0 \
--training_rounds_v1 100 \
--steps 5 \
--n_trials 1

```

### RQ2: Predicting adaptive timestep schedules



#### One-Shot
```bash
CUDA_VISIBLE_DEVICES=0 python3 -u train_evaluate_zeroshot_optimal_timesteps_strategy.py \
--all_config configs/cifar10.yml \
--num_train 100000 \
--num_valid 200 \
--main_train_batch_size 3 \
--main_valid_batch_size 200 \
--training_rounds_v1 1 \
--log_path logs/logs_cifar10 \
--force_train True \
--steps 3 \
--patient 30 \
--lr_time_1 0.0005 \
--mlp_dropout 0 \
--use_optimal_params False \
--log_suffix final_runs 
```
#### Delta

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u delta_timestep_training.py
--all_config configs/cifar10.yml \
--data_dir train_data/train_data_cifar10/uni_pc_NFE20_edm_seed0 \
--num_train 500000 \
--num_valid 200 \
--main_train_batch_size 3 \
--main_valid_batch_size 200 \
--training_rounds_v1 1 \
--log_path logs/logs_cifar10 \
--force_train True \
--steps 3 \
--lr_time_1 0.00005 \
--mlp_dropout 0.0 \
--log_suffix FinalDeltaFixedLastStep 
```



