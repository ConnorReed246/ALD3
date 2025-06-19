import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import lpips
import plotly.express as px
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import save_image



from dataset import LTTDataset, load_data_from_dir
from latent_to_timestep_model import Delta_LTT_model, LTT_model, Tiny_Delta_LTT_model, Medium_Delta_LTT_model, Large_Delta_LTT_model, Huge_Delta_LTT_model, Huge_Delta_LTT_CNN_Bigger, Huge_Delta_LTT_MLP_Bigger, Ginormous_Delta_LTT, Fancy_Delta_LTT_Model, Huge_Bottleneck_Delta, Delta_LTT_model_using_Bottleneck

from models import prepare_stuff
from trainer import LD3Trainer, ModelConfig, TrainingConfig, DiscretizeModelWrapper
from utils import (
    adjust_hyper,
    get_solvers,
    move_tensor_to_device,
    parse_arguments,
    set_seed_everything,
    visual
)


def setup(steps: int, use_optimal_params: bool = False, just_image: bool = False, model = None, return_bottleneck: bool = False, delta_model_size: str = None):
    
    args = parse_arguments([
        "--all_config", "configs/cifar10.yml",
        "--data_dir", "train_data/train_data_cifar10/uni_pc_NFE20_edm_seed0",
        "--num_train", "10000", #45000
        "--num_valid", "1000",
        "--main_train_batch_size", "200",
        "--main_valid_batch_size", "200",
        "--training_rounds_v1", "1",
        "--log_path", "logs/logs_cifar10",
        "--force_train", "True",
        "--lr_time_1", "0.00005",
        "--mlp_dropout", "0.0",
        "--log_suffix", "BiggerValidation_GroupNorm_EvalTrue",
        f"--use_optimal_params", "True" if use_optimal_params else "False",
        "--return_bottleneck", "True" if return_bottleneck else "False",
        "--delta_model_size", delta_model_size,
    ])

    set_seed_everything(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Dataset
    data_dir = 'train_data/train_data_cifar10/uni_pc_NFE20_edm_seed0'


    wrapped_model, _, decoding_fn, noise_schedule, latent_resolution, latent_channel, _, _ = prepare_stuff(args,return_bottleneck=return_bottleneck)
    solver, steps, solver_extra_params = get_solvers(
        args.solver_name,
        NFEs=steps,
        order=args.order,
        noise_schedule=noise_schedule,
        unipc_variant=args.unipc_variant,
    )

    valid_dataset = LTTDataset(dir=os.path.join(data_dir, "validation"), size=args.num_valid, train_flag=False, use_optimal_params=use_optimal_params, optimal_params_path="opt_t_clever_initialisation") 
    train_dataset = LTTDataset(dir=os.path.join(data_dir, "train"), size=args.num_train, train_flag=True, use_optimal_params=use_optimal_params, optimal_params_path="opt_t_clever_initialisation")

    if model is None:
        delta_ltt_model = Delta_LTT_model(steps = steps, mlp_dropout=args.mlp_dropout, just_image = just_image)
    else:
        delta_ltt_model = model(steps = steps, mlp_dropout=args.mlp_dropout)

    delta_ltt_model = delta_ltt_model.to(device)

    training_config = TrainingConfig(
        train_data=train_dataset,
        valid_data=valid_dataset,
        train_batch_size=args.main_train_batch_size,
        valid_batch_size=args.main_valid_batch_size,
        lr_time_1=args.lr_time_1,
        shift_lr=args.shift_lr,
        shift_lr_decay=args.shift_lr_decay,
        min_lr_time_1=args.min_lr_time_1,
        win_rate=args.win_rate,
        patient=args.patient,
        lr_time_decay=args.lr_time_decay,
        momentum_time_1=args.momentum_time_1,
        weight_decay_time_1=args.weight_decay_time_1,
        loss_type=args.loss_type,
        visualize=args.visualize,
        no_v1=args.no_v1,
        prior_timesteps=args.gits_ts,
        match_prior=args.match_prior,
    )
    model_config = ModelConfig(
        net=wrapped_model,
        decoding_fn=decoding_fn,
        noise_schedule=noise_schedule,
        solver=solver,
        solver_name=args.solver_name,
        order=args.order,
        steps=steps,
        prior_bound=args.prior_bound,
        resolution=latent_resolution,
        channels=latent_channel,
        time_mode=args.time_mode,
        solver_extra_params=solver_extra_params,
        device=device,
    )
    trainer = LD3Trainer(model_config, training_config)


    dis_model = DiscretizeModelWrapper( #Changed through LTT
            lambda_max=trainer.lambda_max,
            lambda_min=trainer.lambda_min,
            noise_schedule=trainer.noise_schedule,
            time_mode = trainer.time_mode,
        )

    return trainer, dis_model, delta_ltt_model, device, steps



if __name__ == "__main__":
    batch_size = 500
    number_of_fid_images = 50000
    shape = (batch_size, 3, 32, 32)

    # Global
    # n3_params = torch.tensor([0.3125, 0.1682, 0.1343, 0.3851], device='cuda:0')
    # n3_timestep = torch.tensor([[8.0000e+01, 5.9884e+00, 7.5587e-01, 2.0000e-03]], device='cuda:0')

    # n5_params = torch.tensor([0.2225, 0.1482, 0.1034, 0.0818, 0.0839, 0.3603], device='cuda:0')
    # n5_timestep = torch.tensor([[8.0000e+01, 1.0621e+01, 2.5949e+00, 8.5124e-01, 2.7130e-01, 2.0000e-03]],
    #     device='cuda:0')

    # n6_params = torch.tensor([0.1246, 0.1541, 0.1057, 0.0844, 0.0897, 0.0760, 0.3655],
    # device='cuda:0')
    # n6_timestep = torch.tensor([[8.0000e+01, 1.2389e+01, 3.4480e+00, 1.2409e+00, 4.1899e-01, 1.6694e-01,
    #         2.0000e-03]], device='cuda:0')
    
    # n7_params = torch.tensor([0.0437, 0.1651, 0.1108, 0.0790, 0.1370, 0.0390, 0.1000, 0.3254],
    # device='cuda:0')
    # n7_timestep = torch.tensor([[8.0000e+01, 1.2833e+01, 3.7612e+00, 1.5670e+00, 3.4353e-01, 2.2291e-01,
    #         7.3629e-02, 2.0000e-03]], device='cuda:0')

    # n10_params = torch.tensor([0.0994, 0.1309, 0.0971, 0.0447, 0.0592, 0.0616, 0.1001, 0.0391, 0.0728,
    #         0.0754, 0.2196], device='cuda:0')
    # n10_timestep =  torch.tensor([[8.0000e+01, 1.7152e+01, 5.4689e+00, 3.2309e+00, 1.6107e+00, 7.8026e-01,
    #         2.4016e-01, 1.5164e-01, 6.4386e-02, 2.6514e-02, 2.0000e-03]],
    #     device='cuda:0')

    # print("Global")
    # for n_steps, timestep in zip([3,5,6,7,10], [n3_timestep, n5_timestep, n6_timestep, n7_timestep, n10_timestep]):
    #     set_seed_everything(0)
    #     trainer, dis_model, delta_ltt_model, device, steps = setup(steps = n_steps)
    #     generator = torch.Generator(torch.device(device))
    #     timestep = timestep[0]
    #     generated_images = []
    #     with torch.no_grad():
    #         for i in tqdm(range(number_of_fid_images // batch_size), desc=f"Generating Images for {name}"):
    #             latent = torch.randn(shape, device=torch.device(device), generator=generator)
    #             x_next = trainer.noise_schedule.prior_transformation(latent)
    #             x_next = trainer.solver.sample_simple(
    #                 model_fn=trainer.net,
    #                 x=x_next,
    #                 timesteps=timestep,
    #                 order=trainer.order,
    #                 NFEs=trainer.steps,
    #                 **trainer.solver_extra_params,
    #             )
    #             x_next = trainer.decoding_fn(x_next)
    #             generated_images.append(x_next)

    #         generated_images = torch.cat(generated_images, dim=0)
    #         save_path = "/netpool/homes/connor/DiffusionModels/LD3_connor/fid-generated"
    #         dir_name = f"global_{name}"
    #         dir_path = os.path.join(save_path, dir_name)
    #         os.makedirs(dir_path, exist_ok=True)
    #         for i, img in enumerate(generated_images):
    #             save_image(img, os.path.join(dir_path, f"{i}.png"), normalize=True)
    #     torch.cuda.empty_cache()


    #Global Variable Last step


    # print("Global Variable Last Step")
    # for steps in [3, 5, 6, 7, 10]:
    #     set_seed_everything(0)
    #     trainer, dis_model, delta_ltt_model, device, steps = setup(steps = steps)
    #     model_path = f"/netpool/homes/connor/DiffusionModels/LD3_connor/runs_global_timesteps/models/model_N{steps}_lr0.001_batch3_nTrain200000_FinalGlobalVariableLastStep"
    #     timestep = torch.load(model_path, map_location=device)["timesteps"]
    #     generator = torch.Generator(torch.device(device))
    #     generated_images = []
    #     with torch.no_grad():
    #         for i in tqdm(range(number_of_fid_images // batch_size), desc=f"Generating Images for {steps}"):
    #             latent = torch.randn(shape, device=torch.device(device), generator=generator)
    #             x_next = trainer.noise_schedule.prior_transformation(latent)
    #             x_next = trainer.solver.sample_simple(
    #                 model_fn=trainer.net,
    #                 x=x_next,
    #                 timesteps=timestep,
    #                 order=trainer.order,
    #                 NFEs=trainer.steps,
    #                 **trainer.solver_extra_params,
    #             )
    #             x_next = trainer.decoding_fn(x_next)
    #             generated_images.append(x_next)

    #         generated_images = torch.cat(generated_images, dim=0)
    #         save_path = "/netpool/homes/connor/DiffusionModels/LD3_connor/fid-generated"
    #         dir_name = f"global_variable_last_step_n{steps}"
    #         dir_path = os.path.join(save_path, dir_name)
    #         os.makedirs(dir_path, exist_ok=True)
    #         for i, img in enumerate(generated_images):
    #             save_image(img, os.path.join(dir_path, f"{i}.png"), normalize=True)
    #     torch.cuda.empty_cache()
    
    #LD3
    # Load timesteps for n3
    # ld3_n3_path = "/netpool/homes/connor/LD3_main/logs/logs_cifar10/LD3_correctedLatents_N3-val200-train10000-rv11-seed0/best_v2.pt"
    # ld3_n3_dict = torch.load(ld3_n3_path, map_location=device)
    # ld3_n3_timestep = ld3_n3_dict['best_t_steps'][:len(ld3_n3_dict['best_t_steps']) // 2]

    # # Load timesteps for n5
    # ld3_n5_path = "/netpool/homes/connor/LD3_main/logs/logs_cifar10/LD3_correctedLatents_N5-val200-train10000-rv11-seed0/best_v2.pt"
    # ld3_n5_dict = torch.load(ld3_n5_path, map_location=device)
    # ld3_n5_timestep = ld3_n5_dict['best_t_steps'][:len(ld3_n5_dict['best_t_steps']) // 2]

    # # Load timesteps for n6
    # ld3_n6_path = "/netpool/homes/connor/LD3_main/logs/logs_cifar10/LD3_correctedLatents_N6-val200-train10000-rv11-seed0/best_v2.pt"
    # ld3_n6_dict = torch.load(ld3_n6_path, map_location=device)
    # ld3_n6_timestep = ld3_n6_dict['best_t_steps'][:len(ld3_n6_dict['best_t_steps']) // 2]

    # # Load timesteps for n7
    # ld3_n7_path = "/netpool/homes/connor/LD3_main/logs/logs_cifar10/LD3_correctedLatents_N7-val200-train10000-rv11-seed0/best_v2.pt"
    # ld3_n7_dict = torch.load(ld3_n7_path, map_location=device)
    # ld3_n7_timestep = ld3_n7_dict['best_t_steps'][:len(ld3_n7_dict['best_t_steps']) // 2]

    # # Load timesteps for n10
    # ld3_n10_path = "/netpool/homes/connor/LD3_main/logs/logs_cifar10/LD3_correctedLatents_N10-val200-train10000-rv11-seed0/best_v2.pt"
    # ld3_n10_dict = torch.load(ld3_n10_path, map_location=device)
    # ld3_n10_timestep = ld3_n10_dict['best_t_steps'][:len(ld3_n10_dict['best_t_steps']) // 2]

    # print("LD3")
    # for n_steps, timestep in zip(["3,5,6,7,10"], [ld3_n3_timestep, ld3_n5_timestep, ld3_n6_timestep, ld3_n7_timestep, ld3_n10_timestep]):
    #     trainer, dis_model, delta_ltt_model, device, steps = setup(steps = n_steps)
    #     set_seed_everything(0)
    #     generator = torch.Generator(torch.device(device))
    #     timestep = timestep
    #     generated_images = []
    #     with torch.no_grad():
    #         for i in tqdm(range(number_of_fid_images // batch_size), desc=f"Generating Images for {n_steps}"):
    #             latent = torch.randn(shape, device=torch.device(device), generator=generator)
    #             x_next = trainer.noise_schedule.prior_transformation(latent)
    #             x_next = trainer.solver.sample_simple(
    #                 model_fn=trainer.net,
    #                 x=x_next,
    #                 timesteps=timestep,
    #                 order=trainer.order,
    #                 NFEs=trainer.steps,
    #                 **trainer.solver_extra_params,
    #             )
    #             # x_next = trainer.decoding_fn(x_next) TODO test if this does anything
    #             generated_images.append(x_next)

    #         generated_images = torch.cat(generated_images, dim=0)
    #         save_path = "/netpool/homes/connor/DiffusionModels/LD3_connor/fid-generated"
    #         dir_name = f"ld3_n{name}"
    #         dir_path = os.path.join(save_path, dir_name)
    #         os.makedirs(dir_path, exist_ok=True)
    #         for i, img in enumerate(generated_images):
    #             save_image(img, os.path.join(dir_path, f"{i}.png"), normalize=True)
    #     torch.cuda.empty_cache()


    # print("Static Schedules")
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # vp_model_path = "/netpool/homes/connor/DiffusionModels/LD3_connor/runs_global_timesteps/models/vp_not_0.pt"
    # vp_state_dict = torch.load(vp_model_path, map_location=device)
    # ve_model_path = "/netpool/homes/connor/DiffusionModels/LD3_connor/runs_global_timesteps/models/ve_not_0.pt"
    # ve_state_dict = torch.load(ve_model_path, map_location=device)
    # iddpm_model_path = "/netpool/homes/connor/DiffusionModels/LD3_connor/runs_global_timesteps/models/iddpm_not_0.pt"
    # iddpm_state_dict = torch.load(iddpm_model_path, map_location=device)
    # edm_model_path = "/netpool/homes/connor/DiffusionModels/LD3_connor/runs_global_timesteps/models/edm_not_0.pt"
    # edm_state_dict = torch.load(edm_model_path, map_location=device)




    # for name, state_dict in zip(["vp", "ve", "iddpm", "edm"], [vp_state_dict, ve_state_dict, iddpm_state_dict, edm_state_dict]):
    #     for steps in [3, 5, 6, 7, 10]:
    #         timesteps = state_dict[steps]
    #         trainer, dis_model, delta_ltt_model, device, steps = setup(steps = steps)
    #         set_seed_everything(0)
    #         generator = torch.Generator(torch.device(device))
    #         generated_images = []
    #         with torch.no_grad():
    #             for i in tqdm(range(number_of_fid_images // batch_size), desc=f"Generating Images for {steps} with {name}"):
    #                 latent = torch.randn(shape, device=torch.device(device), generator=generator)
    #                 x_next = trainer.noise_schedule.prior_transformation(latent)
    #                 x_next = trainer.solver.sample_simple(
    #                     model_fn=trainer.net,
    #                     x=x_next,
    #                     timesteps=timesteps,
    #                     order=trainer.order,
    #                     NFEs=trainer.steps,
    #                     **trainer.solver_extra_params,
    #                 )
    #                 # x_next = trainer.decoding_fn(x_next) TODO test if this does anything
    #                 generated_images.append(x_next)

    #             generated_images = torch.cat(generated_images, dim=0)
    #             save_path = "/netpool/homes/connor/DiffusionModels/LD3_connor/fid-generated"
    #             dir_name = f"{name}_n{steps}"
    #             dir_path = os.path.join(save_path, dir_name)
    #             os.makedirs(dir_path, exist_ok=True)
    #             for i, img in enumerate(generated_images):
    #                 save_image(img, os.path.join(dir_path, f"{i}.png"), normalize=True)
    #         torch.cuda.empty_cache()


    print("Good vs Bad Schedules")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestep_large = torch.tensor([80.0, 60, 40, 30, 20, 10,
            5, 1, 0.5, 0.3, 0.1],
        device=device)

    timestep_small = torch.tensor([80.0, 30, 5,1, 0.1, 0.005, 0.001,
            0.0005, 0.0001, 0.00005, 0.00001],
        device=device)

    timestep_optimal = torch.tensor([8.0000e+01, 4.2390e+01, 3.3400e+00, 1.4845e+00, 6.6336e-01, 2.7701e-01,
            1.2638e-01, 6.4367e-02, 3.0538e-02, 9.6005e-03, 2.0000e-03],
        device=device)
    
    n_steps = 10

    for name, timestep in zip(["large", "small", "example_optimal"], [timestep_large, timestep_small, timestep_optimal]):
        trainer, dis_model, delta_ltt_model, device, steps = setup(steps = n_steps)
        set_seed_everything(0)
        generator = torch.Generator(torch.device(device))
        timestep = timestep
        generated_images = []
        with torch.no_grad():
            for i in tqdm(range(number_of_fid_images // batch_size), desc=f"Generating Images for {n_steps}"):
                latent = torch.randn(shape, device=torch.device(device), generator=generator)
                x_next = trainer.noise_schedule.prior_transformation(latent)
                x_next = trainer.solver.sample_simple(
                    model_fn=trainer.net,
                    x=x_next,
                    timesteps=timestep,
                    order=trainer.order,
                    NFEs=trainer.steps,
                    **trainer.solver_extra_params,
                )
                # x_next = trainer.decoding_fn(x_next) TODO test if this does anything
                generated_images.append(x_next)

            generated_images = torch.cat(generated_images, dim=0)
            save_path = "/netpool/homes/connor/DiffusionModels/LD3_connor/fid-generated"
            dir_name = f"{name}_n{n_steps}"
            dir_path = os.path.join(save_path, dir_name)
            os.makedirs(dir_path, exist_ok=True)
            for i, img in enumerate(generated_images):
                save_image(img, os.path.join(dir_path, f"{i}.png"), normalize=True)
        torch.cuda.empty_cache()


    # print("Delta Variable Last Step")
    # n_steps_list = [5]#3, 5, 6, 7, 10
    # model_path_list = [
    #     # "runs_delta_timesteps/models/N3_model_lr5e-05_batch3_nTrain500000_FinalDelta",
    #     # "runs_delta_timesteps/models/N5_model_lr5e-05_batch3_nTrain500000_FinalDelta",
    #     # "runs_delta_timesteps/models/N6_model_lr5e-05_batch3_nTrain500000_FinalDelta",   
    #     # "runs_delta_timesteps/models/N7_model_lr5e-05_batch3_nTrain500000_FinalDelta",
    #     # "runs_delta_timesteps/models/N10_model_lr5e-05_batch3_nTrain500000_FinalDelta"
    #     "/netpool/homes/connor/DiffusionModels/LD3_connor/runs_delta_timesteps/models/N5_model_lr5e-05_batch3_nTrain25000_FinalDeltaAgainstOverfitting",
    #     # "/netpool/homes/connor/DiffusionModels/LD3_connor/runs_delta_timesteps/models/N5_model_lr5e-05_batch3_nTrain25000_FinalDeltaFixedLastStep",
        
    # ]

    # for n_steps, path in zip(n_steps_list, model_path_list):
    #     trainer, dis_model, delta_ltt_model, device, steps = setup(steps = n_steps)
    #     set_seed_everything(0)
    #     generator = torch.Generator(torch.device(device))

    #     delta_ltt_model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    #     delta_ltt_model.eval()
    #     generated_images = []
    #     with torch.no_grad():
    #         for i in tqdm(range(number_of_fid_images // batch_size), desc=f"Generating Images with {n_steps} steps"):
    #             latent = torch.randn(shape, device=torch.device(device), generator=generator)
    #             x_next = trainer.noise_schedule.prior_transformation(latent)
    #             for x in x_next:
    #                 x, _, _ = trainer.solver.delta_sample_simple(
    #                     model_fn=trainer.net,
    #                     delta_ltt=delta_ltt_model,
    #                     x=x.unsqueeze(0),
    #                     order=trainer.order,
    #                     steps = trainer.steps,
    #                     start_timestep = 80,
    #                     NFEs=trainer.steps,
    #                     condition=None,
    #                     unconditional_condition=None,
    #                     **trainer.solver_extra_params,
    #                 )
    #                 generated_images.append(x)

    #         generated_images = torch.cat(generated_images, dim=0)
    #         save_path = "/netpool/homes/connor/DiffusionModels/LD3_connor/fid-generated"
    #         dir_name = f"delta_early_stop_n{n_steps}"
    #         dir_path = os.path.join(save_path, dir_name)
    #         os.makedirs(dir_path, exist_ok=True)
    #         for i, img in enumerate(generated_images):
    #             save_image(img, os.path.join(dir_path, f"{i}.png"), normalize=True)
    #     torch.cuda.empty_cache()


    # print("Delta Fixed Last Step")
    # n_steps_list = [3, 5, 6, 7, 10]
    # model_path_list = [
    # "runs_delta_timesteps/models/N3_model_lr5e-05_batch3_nTrain500000_FinalDeltaFixedLastStep",
    # "runs_delta_timesteps/models/N5_model_lr5e-05_batch3_nTrain500000_FinalDeltaFixedLastStep",
    # "runs_delta_timesteps/models/N6_model_lr5e-05_batch3_nTrain500000_FinalDeltaFixedLastStep",   
    # "runs_delta_timesteps/models/N7_model_lr5e-05_batch3_nTrain500000_FinalDeltaFixedLastStep",
    # "runs_delta_timesteps/models/N10_model_lr5e-05_batch3_nTrain500000_FinalDeltaFixedLastStep"
    # ]

    # for n_steps, path in zip(n_steps_list, model_path_list):
    #     trainer, dis_model, delta_ltt_model, device, steps = setup(steps = n_steps)
    #     set_seed_everything(0)
    #     generator = torch.Generator(torch.device(device))

    #     delta_ltt_model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    #     delta_ltt_model.eval()
    #     generated_images = []
    #     with torch.no_grad():
    #         for i in tqdm(range(number_of_fid_images // batch_size), desc=f"Generating Images with {n_steps} steps"):
    #             latent = torch.randn(shape, device=torch.device(device), generator=generator)
    #             x_next = trainer.noise_schedule.prior_transformation(latent)
    #             for x in x_next:
    #                 x, _, _ = trainer.solver.delta_sample_simple(
    #                     model_fn=trainer.net,
    #                     delta_ltt=delta_ltt_model,
    #                     x=x.unsqueeze(0),
    #                     order=trainer.order,
    #                     steps = trainer.steps,
    #                     start_timestep = 80,
    #                     NFEs=trainer.steps,
    #                     condition=None,
    #                     unconditional_condition=None,
    #                     fix_last_step=True,
    #                     **trainer.solver_extra_params,
    #                 )
    #                 generated_images.append(x)

    #         generated_images = torch.cat(generated_images, dim=0)
    #         save_path = "/netpool/homes/connor/DiffusionModels/LD3_connor/fid-generated"
    #         dir_name = f"delta_fixed_last_step_n{n_steps}"
    #         dir_path = os.path.join(save_path, dir_name)
    #         os.makedirs(dir_path, exist_ok=True)
    #         for i, img in enumerate(generated_images):
    #             save_image(img, os.path.join(dir_path, f"{i}.png"), normalize=True)
    #     torch.cuda.empty_cache()



    # print("Delta different Sizes")
    # n_steps_list = [
    #     # 5,
    #     # 5,
    #     # 5,
    #     # 5,
    #     # 5,
    #     # 5,
    #     # 5,
    #     # 6,
    #     # 6,
    #     # 6,
    #     # 6,
    #     # 3,
    #     # 7,
    #     # 10,
    #     # 3,
    #     # 5,
    #     # 6,
    #     # 7,
    #     # 10,
    #     3,
    #     5,
    #     6,
    #     7,
    #     10
    #     ]
    # model_path_list = [
    # # "runs_delta_timesteps/models/N5_model_lr5e-05_batch3_nTrain200000_FinalSizeTiny",
    # # "runs_delta_timesteps/models/N5_model_lr5e-05_batch3_nTrain200000_FinalSizeMedium",
    # # "runs_delta_timesteps/models/N5_model_lr5e-05_batch3_nTrain200000_FinalSizeLarge",
    # # "runs_delta_timesteps/models/N5_model_lr5e-05_batch3_nTrain200000_FinalSizeHuge",
    # # "runs_delta_timesteps/models/N5_model_lr5e-05_batch3_nTrain500000_FinalSizeHugeCNNBigger",
    # # "runs_delta_timesteps/models/N5_model_lr5e-05_batch3_nTrain500000_FinalSizeHugeMLPBigger",
    # # "runs_delta_timesteps/models/N5_model_lr5e-05_batch3_nTrain500000_FinalSizeHugeGinormous",
    # # "runs_delta_timesteps/models/N6_model_lr5e-05_batch3_nTrain500000_FinalSizeHuge",
    # # "runs_delta_timesteps/models/N6_model_lr5e-05_batch3_nTrain500000_FinalSizeHugeCNNBigger",
    # # "runs_delta_timesteps/models/N6_model_lr5e-05_batch3_nTrain500000_FinalSizeHugeMLPBigger",
    # # "runs_delta_timesteps/models/N6_model_lr5e-05_batch3_nTrain500000_FinalSizeHugeGinormous",
    # # "runs_delta_timesteps/models/N3_model_lr5e-05_batch3_nTrain500000_FinalSizeHuge",
    # # "runs_delta_timesteps/models/N7_model_lr5e-05_batch3_nTrain500000_FinalSizeHuge",
    # # "runs_delta_timesteps/models/N10_model_lr5e-05_batch3_nTrain500000_FinalSizeHuge",
    # # "runs_delta_timesteps/models/N3_model_lr5e-05_batch3_nTrain500000_FinalSizeFancy",
    # # "runs_delta_timesteps/models/N5_model_lr5e-05_batch3_nTrain500000_FinalSizeFancy",
    # # "runs_delta_timesteps/models/N6_model_lr5e-05_batch3_nTrain500000_FinalSizeFancy",
    # # "runs_delta_timesteps/models/N7_model_lr5e-05_batch3_nTrain500000_FinalSizeFancy",
    # # "runs_delta_timesteps/models/N10_model_lr5e-05_batch3_nTrain500000_FinalSizeFancy"
    # "runs_delta_timesteps/models/N3_model_lr5e-05_batch3_nTrain500000_FinalSizeHuge",
    # "runs_delta_timesteps/models/N5_model_lr5e-05_batch3_nTrain500000_FinalSizeHuge",
    # "runs_delta_timesteps/models/N6_model_lr5e-05_batch3_nTrain500000_FinalSizeHuge",
    # "runs_delta_timesteps/models/N7_model_lr5e-05_batch3_nTrain500000_FinalSizeHuge",
    # "runs_delta_timesteps/models/N10_model_lr5e-05_batch3_nTrain500000_FinalSizeHuge"

    # ]
    # model_list = [
    #     # Tiny_Delta_LTT_model,
    #     # Medium_Delta_LTT_model,
    #     # Large_Delta_LTT_model,
    #     # Huge_Delta_LTT_model,
    #     # Huge_Delta_LTT_CNN_Bigger,
    #     # Huge_Delta_LTT_MLP_Bigger,
    #     # Ginormous_Delta_LTT,
    #     # Huge_Delta_LTT_model,
    #     # Huge_Delta_LTT_CNN_Bigger,
    #     # Huge_Delta_LTT_MLP_Bigger,
    #     # Ginormous_Delta_LTT,
    #     # Huge_Delta_LTT_model,
    #     # Huge_Delta_LTT_model,
    #     # Huge_Delta_LTT_model,
    #     # Fancy_Delta_LTT_Model,
    #     # Fancy_Delta_LTT_Model,
    #     # Fancy_Delta_LTT_Model,
    #     # Fancy_Delta_LTT_Model,
    #     # Fancy_Delta_LTT_Model,
    #     Huge_Delta_LTT_model,
    #     Huge_Delta_LTT_model,
    #     Huge_Delta_LTT_model,
    #     Huge_Delta_LTT_model,
    #     Huge_Delta_LTT_model
    # ]

    # for n_steps, path, model in zip(n_steps_list, model_path_list,model_list):
    #     trainer, dis_model, delta_ltt_model, device, steps = setup(steps = n_steps, model= model)
    #     set_seed_everything(0)
    #     generator = torch.Generator(torch.device(device))

    #     delta_ltt_model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    #     delta_ltt_model.eval()
    #     generated_images = []
    #     with torch.no_grad():
    #         for i in tqdm(range(number_of_fid_images // batch_size), desc=f"Generating Images for {n_steps} steps with {model.__name__}"):
    #             latent = torch.randn(shape, device=torch.device(device), generator=generator)
    #             x_next = trainer.noise_schedule.prior_transformation(latent)
    #             for x in x_next:
    #                 x, _, _ = trainer.solver.delta_sample_simple(
    #                     model_fn=trainer.net,
    #                     delta_ltt=delta_ltt_model,
    #                     x=x.unsqueeze(0),
    #                     order=trainer.order,
    #                     steps = trainer.steps,
    #                     start_timestep = 80,
    #                     NFEs=trainer.steps,
    #                     condition=None,
    #                     unconditional_condition=None,
    #                     fix_last_step=True,
    #                     **trainer.solver_extra_params,
    #                 )
    #                 generated_images.append(x)

    #         generated_images = torch.cat(generated_images, dim=0)
    #         save_path = "/netpool/homes/connor/DiffusionModels/LD3_connor/fid-generated"
    #         dir_name = f"{model.__name__}_n{n_steps}"
    #         dir_path = os.path.join(save_path, dir_name)
    #         os.makedirs(dir_path, exist_ok=True)
    #         for i, img in enumerate(generated_images):
    #             save_image(img, os.path.join(dir_path, f"{i}.png"), normalize=True)
    #     torch.cuda.empty_cache()
    


    # print("Delta Just Image")
    # n_steps_list = [3, 5, 6, 7, 10]
    # model_path_list = [
    # "runs_delta_timesteps/models/N3_model_lr5e-05_batch3_nTrain500000_FinalDeltaJustImage",
    # "runs_delta_timesteps/models/N5_model_lr5e-05_batch3_nTrain500000_FinalDeltaJustImage",
    # "runs_delta_timesteps/models/N6_model_lr5e-05_batch3_nTrain500000_FinalDeltaJustImage",
    # "runs_delta_timesteps/models/N7_model_lr5e-05_batch3_nTrain500000_FinalDeltaJustImage",
    # "runs_delta_timesteps/models/N10_model_lr5e-05_batch3_nTrain500000_FinalDeltaJustImage"
    # ]

    # for n_steps, path in zip(n_steps_list, model_path_list):
    #     trainer, dis_model, delta_ltt_model, device, steps = setup(steps = n_steps, just_image=True)
    #     set_seed_everything(0)
    #     generator = torch.Generator(torch.device(device))

    #     delta_ltt_model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    #     delta_ltt_model.eval()
    #     generated_images = []
    #     with torch.no_grad():
    #         for i in tqdm(range(number_of_fid_images // batch_size), desc=f"Generating Images with {n_steps} steps"):
    #             latent = torch.randn(shape, device=torch.device(device), generator=generator)
    #             x_next = trainer.noise_schedule.prior_transformation(latent)
    #             for x in x_next:
    #                 x, _, _ = trainer.solver.delta_sample_simple(
    #                     model_fn=trainer.net,
    #                     delta_ltt=delta_ltt_model,
    #                     x=x.unsqueeze(0),
    #                     order=trainer.order,
    #                     steps = trainer.steps,
    #                     start_timestep = 80,
    #                     NFEs=trainer.steps,
    #                     condition=None,
    #                     unconditional_condition=None,
    #                     fix_last_step=True,
    #                     **trainer.solver_extra_params,
    #                 )
    #                 generated_images.append(x)

    #         generated_images = torch.cat(generated_images, dim=0)
    #         save_path = "/netpool/homes/connor/DiffusionModels/LD3_connor/fid-generated"
    #         dir_name = f"delta_just_image_n{n_steps}"
    #         dir_path = os.path.join(save_path, dir_name)
    #         os.makedirs(dir_path, exist_ok=True)
    #         for i, img in enumerate(generated_images):
    #             save_image(img, os.path.join(dir_path, f"{i}.png"), normalize=True)
    #     torch.cuda.empty_cache()


    # print("Bottleneck Huge")
    # for n_steps in [3, 5, 6, 7, 10]:
    #     model_path = f"runs_delta_timesteps/models/N{n_steps}_model_lr5e-05_batch3_nTrain500000_FinalBottleneckSizeHuge"
    #     trainer, dis_model, delta_ltt_model, device, steps = setup(steps = n_steps, model= Huge_Bottleneck_Delta, return_bottleneck=True, delta_model_size="huge")
    #     set_seed_everything(0)
    #     generator = torch.Generator(torch.device(device))

    #     delta_ltt_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    #     delta_ltt_model.eval()
    #     generated_images = []
    #     with torch.no_grad():
    #         for i in tqdm(range(number_of_fid_images // batch_size), desc=f"Generating Images for {n_steps} steps with Huge_Bottleneck_Delta"):
    #             latent = torch.randn(shape, device=torch.device(device), generator=generator)
    #             x_next = trainer.noise_schedule.prior_transformation(latent)
    #             for x in x_next:
    #                 x, _, _ = trainer.solver.delta_sample_simple(
    #                     model_fn=trainer.net,
    #                     delta_ltt=delta_ltt_model,
    #                     x=x.unsqueeze(0),
    #                     order=trainer.order,
    #                     steps = trainer.steps,
    #                     start_timestep = 80,
    #                     NFEs=trainer.steps,
    #                     condition=None,
    #                     unconditional_condition=None,
    #                     fix_last_step=True,
    #                     return_bottleneck = True, 
    #                     **trainer.solver_extra_params,
    #                 )
    #                 generated_images.append(x)

    #         generated_images = torch.cat(generated_images, dim=0)
    #         save_path = "/netpool/homes/connor/DiffusionModels/LD3_connor/fid-generated"
    #         dir_name = f"Huge_Bottleneck_Delta_n{n_steps}"
    #         dir_path = os.path.join(save_path, dir_name)
    #         os.makedirs(dir_path, exist_ok=True)
    #         for i, img in enumerate(generated_images):
    #             save_image(img, os.path.join(dir_path, f"{i}.png"), normalize=True)
    #     torch.cuda.empty_cache()


    # print("Bottleneck")
    # for n_steps in [3, 5, 6, 7, 10]:
    #     model_path = f"runs_delta_timesteps/models/N{n_steps}_model_lr5e-05_batch3_nTrain500000_FinalBottleneck"
    #     trainer, dis_model, delta_ltt_model, device, steps = setup(steps = n_steps, model= Delta_LTT_model_using_Bottleneck, return_bottleneck=True)
    #     set_seed_everything(0)
    #     generator = torch.Generator(torch.device(device))

    #     delta_ltt_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    #     delta_ltt_model.eval()
    #     generated_images = []
    #     with torch.no_grad():
    #         for i in tqdm(range(number_of_fid_images // batch_size), desc=f"Generating Images for {n_steps} steps with Bottleneck_Delta"):
    #             latent = torch.randn(shape, device=torch.device(device), generator=generator)
    #             x_next = trainer.noise_schedule.prior_transformation(latent)
    #             for x in x_next:
    #                 x, _, _ = trainer.solver.delta_sample_simple(
    #                     model_fn=trainer.net,
    #                     delta_ltt=delta_ltt_model,
    #                     x=x.unsqueeze(0),
    #                     order=trainer.order,
    #                     steps = trainer.steps,
    #                     start_timestep = 80,
    #                     NFEs=trainer.steps,
    #                     condition=None,
    #                     unconditional_condition=None,
    #                     fix_last_step=True,
    #                     return_bottleneck = True, 
    #                     **trainer.solver_extra_params,
    #                 )
    #                 generated_images.append(x)

    #         generated_images = torch.cat(generated_images, dim=0)
    #         save_path = "/netpool/homes/connor/DiffusionModels/LD3_connor/fid-generated"
    #         dir_name = f"Bottleneck_Delta_n{n_steps}"
    #         dir_path = os.path.join(save_path, dir_name)
    #         os.makedirs(dir_path, exist_ok=True)
    #         for i, img in enumerate(generated_images):
    #             save_image(img, os.path.join(dir_path, f"{i}.png"), normalize=True)
    #     torch.cuda.empty_cache()







    # print("Optimal Timesteps")
    # n_steps = 5
    # trainer, dis_model, delta_ltt_model, device, steps = setup(steps = n_steps, use_optimal_params=True)
    # generated_images = []
    # with torch.no_grad():
    #     for batch in tqdm(trainer.train_loader, desc="Generating Images with Optimal Timesteps"):
    #         _, latent, optimal_params = batch
    #         latent = latent.to(device)
    #         optimal_params = optimal_params.to(device)
    #         timesteps = dis_model.convert(optimal_params)
    #         for x, timestep in zip(latent, timesteps):
    #             x_next = trainer.noise_schedule.prior_transformation(x)
    #             x_next = trainer.solver.sample_simple(
    #                 model_fn=trainer.net,
    #                 x=x_next.unsqueeze(0),
    #                 timesteps=timestep,
    #                 order=trainer.order,
    #                 NFEs=trainer.steps,
    #                 **trainer.solver_extra_params,
    #             )
    #         # x_next = trainer.decoding_fn(x_next) TODO test if this does anything
    #             generated_images.append(x_next)

    #     generated_images = torch.cat(generated_images, dim=0)
    #     save_path = "/netpool/homes/connor/DiffusionModels/LD3_connor/fid-generated"
    #     dir_name = f"optimal_n{n_steps}"
    #     dir_path = os.path.join(save_path, dir_name)
    #     os.makedirs(dir_path, exist_ok=True)
    #     for i, img in enumerate(generated_images):
    #         save_image(img, os.path.join(dir_path, f"{i}.png"), normalize=True)
    # torch.cuda.empty_cache()


    # print("zero-shot")
    # for n_step in [3, 5, 6, 7, 10]:
    #     # Generate 50k images for FID/IS
    #     print(f"Generating 50k images for zero-shot n={n_step}")
    #     set_seed_everything(0)
    #     trainer, dis_model, _, device, steps = setup(steps=n_step)
    #     batch_size = 500
    #     number_of_fid_images = 50000
    #     shape = (batch_size, 3, 32, 32)
    #     generator = torch.Generator(torch.device(device))
    #     ltt_model = LTT_model(steps=n_step)
    #     state_dict = torch.load(
    #         f"/netpool/homes/connor/DiffusionModels/LD3_connor/runs_zeroshot_timesteps/models/steps_{n_step}_model_lr0.0005_batch3_final_runs",
    #         weights_only=True
    #     )
    #     ltt_model.load_state_dict(state_dict)
    #     ltt_model = ltt_model.to(device)
    #     ltt_model.eval()
    #     generated_images = []
    #     with torch.no_grad():
    #         for i in tqdm(range(number_of_fid_images // batch_size), desc=f"Zero-shot FID images n={n_step}"):
    #             latent = torch.randn(shape, device=device, generator=generator)
    #             timesteps = dis_model.convert(ltt_model(latent))
    #             x_next_finished = []
    #             for timestep, single_latent in zip(timesteps, latent):
    #                 x_next = trainer.noise_schedule.prior_transformation(single_latent)
    #                 x_next = trainer.solver.sample_simple(
    #                     model_fn=trainer.net,
    #                     x=x_next.unsqueeze(0),
    #                     timesteps=timestep,
    #                     order=trainer.order,
    #                     NFEs=n_step,
    #                     **trainer.solver_extra_params,
    #                 )
    #                 x_next = trainer.decoding_fn(x_next)
    #                 x_next_finished.append(x_next)
    #             x_next = torch.cat(x_next_finished, dim=0)
    #             generated_images.append(x_next)
    #         generated_images = torch.cat(generated_images, dim=0)
    #         save_path = "/netpool/homes/connor/DiffusionModels/LD3_connor/fid-generated"
    #         dir_name = f"zeroshot_n{n_step}"
    #         dir_path = os.path.join(save_path, dir_name)
    #         os.makedirs(dir_path, exist_ok=True)
    #         for i, img in enumerate(generated_images):
    #             save_image(img, os.path.join(dir_path, f"{i}.png"), normalize=True)
    #     torch.cuda.empty_cache()

    
                
        
