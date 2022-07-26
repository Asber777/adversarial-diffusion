import argparse
import os
import shutil
import numpy as np
import torch as th
import os.path as osp
import datetime
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn
from torch import clamp
from torchvision import transforms
import lpips
from pytorch_msssim import ssim, ms_ssim
from robustbench.utils import load_model
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
    get_idex2name_map, 
    save_args, 
    arr2pic_save, 
)

hidden_index = [1,2,3,4,5,6,7]
GUIDE_Y = [583, 445, 703, 546, 254]
INDEX2NAME_MAP_PATH = "/root/hhtpro/123/guided-diffusion/scripts/image_label_map.txt"
DT = lambda :datetime.datetime.now().strftime("adv-%Y-%m-%d-%H-%M-%S-%f")

def create_argparser():
    defaults = dict(
        result_dir='/root/hhtpro/123/result', # where to store experiments
        describe="default_desc",
        guide_exp="256_guide_pic", # where to load original picture
        clip_denoised=True,
        num_samples=5,
        batch_size=5,
        model_path="/root/hhtpro/123/256x256_diffusion.pt",
        classifier_path="/root/hhtpro/123/256x256_classifier.pt",
        splitT=200, 
        generate_scale=10.0,
        guide_scale=0.0, 
        hidden_scale=1.0,
        lpips_scale=10.0, 
        ssim_scale=10.0, 
        use_lpips=True, 
        use_mse=False,
        use_ssim=True, 
        get_hidden=False,
        get_middle=False, 
        guide_as_generate=False, 
        attack_model_name="Salman2020Do_50_2",
        attack_model_type='Linf',
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    # MODEL_FLAGS
    model_flags = dict(
        timestep_respacing='ddim25', 
        use_ddim=True,
        image_size=256, 
        attention_resolutions="32,16,8",
        class_cond=True, 
        diffusion_steps=1000,
        learn_sigma = True, 
        noise_schedule='linear', 
        num_channels=256,
        num_head_channels=64,
        num_res_blocks=2,
        resblock_updown = True, 
        use_fp16 = True,
        use_scale_shift_norm = True
    )
    defaults.update(model_flags)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()

    dir = osp.join(args.result_dir, args.describe, DT(),)
    logger.configure(dir)

    save_args(logger.get_dir(), args)
    shutil.copy(os.path.realpath(__file__), logger.get_dir())

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16: model.convert_to_fp16()
    model.eval()

    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16: classifier.convert_to_fp16()
    classifier.eval()

    attack_model = load_model(model_name=args.attack_model_name, 
        dataset='imagenet', threat_model=args.attack_model_type)
    attack_model = attack_model.to(dist_util.dev())

    loss_fn_alex = lpips.LPIPS(net='alex')
    loss_fn_alex = loss_fn_alex.to(dist_util.dev())
    
    if args.guide_exp: # 之后要改
        logger.log("This experimen t is guide_exp")
        guide_path = osp.join(args.result_dir, args.guide_exp, "samples_5x256x256x3.npz")
        guide_np = np.load(guide_path)
        guide_x_np, generate_y_np = guide_np['arr_0'], guide_np['arr_1']
        assert args.batch_size <= len(generate_y_np) # num_samples
        guide_x = th.from_numpy(guide_x_np[:args.batch_size]).to(dist_util.dev())
        guide_x = (guide_x*2. -1.).clamp(-1., 1.)
        generate_y = th.from_numpy(generate_y_np[:args.batch_size]).to(dist_util.dev())
        guide_y_np = np.array(GUIDE_Y)
        guide_y = th.from_numpy(guide_y_np[:args.batch_size]).to(dist_util.dev())
        if args.guide_as_generate:
            guide_y = generate_y.detach().clone()
            guide_y_np = guide_y.cpu().numpy()
    else: 
        guide_x_np = np.array([])
        logger.log("This experiment is not guide_exp")
        guide_y = generate_y = th.randint(
            low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
        )

    # modify conf_fn and model_fn to get adv
    CenterCrop = lambda x: x[:, :, 16:240, 16:240]
    def cond_fn(x, t, y=None, **kwargs):
        assert y is not None
        time = t[0].detach().clone().cpu().item()
        with th.enable_grad():
            generate_loss = 0
            hidden_loss = 0
            lpips_loss = 0
            ssim_loss = 0
            adv_loss = 0

            x_in = x.detach().requires_grad_(True)
            if args.use_lpips == True:
                # input need to be range from [-1, 1]
                lpips_loss += loss_fn_alex.forward(x_in, guide_x).sum()
            
            if args.use_ssim == True: 
                # ssim need input range from [0, 1]
                ssim_loss = ssim((guide_x+1)/2, (x_in+1)/2, data_range=1, size_average=True)
            
            if args.use_mse == True:
                _, guide_hidden = classifier(guide_x, t, \
                    args.get_hidden, args.get_middle, hidden_index)
                _, hidden = classifier(x_in, t, \
                    args.get_hidden, args.get_middle, hidden_index)
                for h, gh in zip(hidden, guide_hidden):
                    hidden_loss -= ((h - gh)**2).mean()
            
            if time > args.splitT:
                logits = classifier(x_in, t)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs[range(len(logits)), generate_y.view(-1)] 
                generate_loss = selected.sum()

            if time < args.splitT:
                attack_logits = attack_model(((CenterCrop(x_in)+1)/2.))
                attack_log_probs = F.log_softmax(attack_logits, dim=-1)
                attack_selected = attack_log_probs[range(len(logits)), guide_y.view(-1)] 
                adv_loss = attack_selected.sum()

            loss = generate_loss* args.generate_scale
            loss += adv_loss * args.guide_scale
            loss += ssim_loss * args.ssim_scale
            loss += hidden_loss * args.hidden_scale
            loss -= lpips_loss * args.lpips_scale
            gradient = th.autograd.grad(loss, x_in)[0]
        logger.log("t:{}, generate: {}, guide:{}, ssim:{}, lpips:{}".format(time,selected.sum(),adv_loss,ssim_loss, lpips_loss))
        return gradient

    def model_fn(x, t, y=None, **kwargs):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("sampling...", DT())
    all_images = []
    all_labels = []
    all_predict = []
    model_kwargs = {"y": generate_y}
    sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
    while len(all_images) * args.batch_size < args.num_samples:
        sample = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
        )
        # log_probs = classifier(sample, th.zeros_like(generate_y))
        log_probs = attack_model(((CenterCrop(sample)+1)/2.))
        predict = log_probs.argmax(dim=-1)

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        # collect sample from each process

        def gether(sample):
            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            return [sample.cpu().numpy() for sample in gathered_samples]

        all_images.extend(gether(sample))
        all_labels.extend(gether(generate_y))
        all_predict.extend(gether(predict))
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    
    arr = np.concatenate(all_images, axis=0)[: args.num_samples] # arr: genrate result
    label_arr = np.concatenate(all_labels, axis=0)[: args.num_samples] # label_arr: generate_y
    predict_arr = np.concatenate(all_predict, axis=0)[: args.num_samples] # predict result
    
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        arr2pic_save(arr, logger.get_dir())
        # print label and save data
        map_i_s = get_idex2name_map(INDEX2NAME_MAP_PATH)
        for i, (y_, p_) in enumerate(zip(label_arr, predict_arr)):
            g_y = guide_y_np[i%args.batch_size] if args.guide_exp else guide_y_np[i]
            logger.log(f"original_label_{i}:{y_}, guide_y_{i%args.batch_size}:{g_y}, predict{i}:{p_}")
            logger.log(f"{map_i_s[y_]} ; {map_i_s[g_y]}; {map_i_s[p_]}")
        np.savez(out_path, arr, label_arr, guide_y_np, predict_arr, guide_x_np)
    
    dist.barrier()
    logger.log("sampling complete", DT())

if __name__ == "__main__":
    main()
