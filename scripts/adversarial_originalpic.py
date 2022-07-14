import argparse
from ast import arg
from asyncore import write
import os

import shutil
import json
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
from robustbench.utils import load_model
from tensorboardX import SummaryWriter
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
)

SHOW_PIC_N = 5
ORGINAL_PIC_PATH = "generate_original/classifier_scale10/adv-2022-06-29-00-04-25-887867"
INDEX2NAME_MAP_PATH = "/root/hhtpro/123/guided-diffusion/scripts/image_label_map.txt"
'''
This file can generate new adversarial images that look like guide_x and 
were classified as guide_y which is asign by us. 
Use following command to do so:
SAMPLE_FLAGS="--batch_size 5 --num_samples 5 --timestep_respacing 250"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 --learn_sigma True \
--noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_new_attention_order True \
--use_fp16 True --use_scale_shift_norm True"
python guided-diffusion/scripts/adversarial_originalpic.py $MODEL_FLAGS --classifier_path 64x64_classifier.pt --classifier_depth 4 \
--model_path 64x64_diffusion.pt $SAMPLE_FLAGS  --generate_scale 13.0 --guide_scale 13.0 --hidden_scale 20.0 --describe "pic_and_adv_guide"
'''

def get_idex2name_map():
    result_dict = {}
    with open(INDEX2NAME_MAP_PATH) as fp:
        sample = fp.readlines()
        for line in sample:
            sample_ = line.split('\t',maxsplit=1)
            result_dict[int(sample_[0])]=sample_[1].split('\n')[0]
    return result_dict

def create_argparser():
    defaults = dict(
        result_dir='/root/hhtpro/123/result', # where to store experiments
        guide_exp=ORGINAL_PIC_PATH, # where to load original picture
        splitT=400, 
        clip_denoised=True,
        num_samples=5,
        batch_size=5,
        use_ddim=False,
        model_path="",
        classifier_path="",
        generate_scale=1.0,
        guide_scale=1.0, 
        hidden_scale=1.0,
        use_lpips=True, 
        use_pgd=False, 
        get_hidden=True,
        get_middle=True, 
        describe="default_desc",
        guide_as_generate=False, 
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()

    dir = osp.join(args.result_dir, args.describe,
        datetime.datetime.now().strftime("adv-%Y-%m-%d-%H-%M-%S-%f"),
    )
    logger.configure(dir)

    logger.log("creating model and diffusion...")
    logger.log(
        "current config is splitT: {}, guide_exp: {}, hidden_scale: {}, generate_scale: {}, guide_scale: {}".format(
                 args.splitT, args.guide_exp, args.hidden_scale,
                  args.generate_scale, args.guide_scale))
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    loss_fn_alex = lpips.LPIPS(net='alex')
    loss_fn_alex = loss_fn_alex.to(dist_util.dev())

    writer = SummaryWriter(logger.get_dir()+'/'+args.describe)
    
    # get guide picture and guide_y generate y
    if args.guide_exp is not None:
        logger.log("This experiment is guide_exp")
        guide_path = osp.join(args.result_dir, args.guide_exp, "samples_5x64x64x3.npz")
        guide_np = np.load(guide_path)
        guide_x_np = guide_np['arr_0']
        generate_y_np = guide_np['arr_1'] 
        assert args.batch_size <= len(generate_y_np)
        guide_x = th.from_numpy(guide_x_np[:args.batch_size]).to(dist_util.dev())
        guide_x = guide_x.permute(0, 3, 1, 2)
        # guide_x is orginal natural picture
        guide_x = ((guide_x/127.5) -1.).clamp(-1., 1.)
        writer.add_image('guide_x', make_grid(guide_x, args.batch_size, normalize=True), 0)
        # generate_y means origianl label of guide_x
        generate_y = th.from_numpy(generate_y_np[:args.batch_size]).to(dist_util.dev())
        # guide_y means guide_y in cond_fn to make it classified as guide_y
        # this is dummpy to do so ⬇, we need to guide it to ... 
        guide_y_np = np.array([583, 445, 703, 546, 254])
        guide_y = th.from_numpy(guide_y_np[:args.batch_size]).to(dist_util.dev())
        if args.guide_as_generate:
            guide_y = generate_y.detach().clone()
            guide_y_np = guide_y.cpu().numpy()
    else: 
        logger.log("This experiment is not guide_exp")
        guide_y = generate_y = th.randint(
            low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
        )
    if args.use_pgd: 
        predict = load_model(model_name='Standard_R50', dataset='imagenet', threat_model='Linf')
        predict = predict.to(dist_util.dev())
        def pgd(x, y, nb_iter=10, eps=8./255, eps_iter=2./255, 
                clip_min=0.0, clip_max=1.0, random_init = True):
            x, y = x.detach().clone(), y.detach().clone()
            # init delta randomly
            delta = th.zeros_like(x)
            delta = nn.Parameter(delta)
            if random_init:
                delta.data.uniform_(-1, 1)
                delta.data *= eps
                delta.data = clamp(x + delta.data, min=clip_min, max=clip_max) - x
            # begin PGD
            with th.enable_grad():
                delta.requires_grad_()
                for _ in range(nb_iter):
                    outputs = predict(x + delta)
                    loss = nn.CrossEntropyLoss(reduction="sum")(outputs, y)
                    loss.backward()
                    # first limit delta in [-eps,eps] then limit data in [clip_min,clip_max](inf_ord)
                    grad_sign = delta.grad.data.sign()
                    grad_sign *= eps_iter
                    delta.data = delta.data + grad_sign
                    delta.data = clamp(delta.data, -eps, eps)
                    delta.data = clamp(x.data + delta.data, clip_min, clip_max) - x.data
                    delta.grad.data.zero_()
                x_adv = clamp(x + delta, clip_min, clip_max)
            return x_adv.data

    # modify conf_fn and model_fn to get adv
    def cond_fn(x, t, y=None, mean=None, variance=None, **kwargs):
        assert y is not None
        assert mean is not None
        assert variance is not None
        time = t[0].detach().cpu().item()
        print(time)
        with th.enable_grad():
            hidden_loss = 0
            x_in = x.detach().requires_grad_(True)
            if args.use_lpips == True:
                hidden_loss -= loss_fn_alex.forward(x_in, guide_x).sum()
                logits = classifier(x_in, t)
            else:
                _, guide_hidden = classifier(guide_x, t, args.get_hidden, args.get_middle)
                logits, hidden = classifier(x_in, t, args.get_hidden, args.get_middle)
                for h, gh in zip(hidden, guide_hidden):
                    hidden_loss -= ((h - gh)**2).mean()
            # when t>splitT, use generate_y else guide_y
            # I think it's dummy to do so, we should optimize pic fid 
            # as use genrate_y when it's classified as guide_y, 
            # and use guide_y when it's classified as other label. 
            log_probs = F.log_softmax(logits, dim=-1)
            new_y = th.where(t>args.splitT, generate_y, guide_y)
            s = args.generate_scale if time > args.splitT else args.guide_scale
            selected = log_probs[range(len(logits)), new_y.view(-1)]
            loss = selected.sum() * s
            loss += hidden_loss * args.hidden_scale
            gradient = th.autograd.grad(loss, x_in)[0]
            writer.add_scalar('hidden_loss', hidden_loss.cpu().item(), time)
            writer.add_scalar('selected_loss', selected.sum().cpu().item(), time)
            writer.add_scalar('loss', loss.cpu().item(), time)
        new_mean = (mean.float() + variance * gradient.float())
        writer.add_image('mean', make_grid(new_mean, args.batch_size, normalize=True), time)
        if args.use_pgd and t[0] < 100:
            x = new_mean * 0.5 + 0.5
            advx = pgd(x, guide_y, )
            new_mean = (advx - 0.5) * 2.0
        return new_mean

    def model_fn(x, t, y=None, **kwargs):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("sampling...")
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
        log_probs = classifier(sample, th.zeros_like(generate_y))
        predict = log_probs.argmax(dim=-1)
        # scale from [-1, 1] to [0, 255]
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        # collect sample from each process
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        
        # collect label from each process (it's not neccessary as we are single process)
        gathered_labels = [th.zeros_like(generate_y) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_labels, generate_y)
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        
        # collect prediction from each process 
        gathered_predicts = [th.zeros_like(predict) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_predicts, predict)
        all_predict.extend([predict.cpu().numpy() for predict in gathered_predicts])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    
    arr = np.concatenate(all_images, axis=0)[: args.num_samples] # arr: genrate result
    label_arr = np.concatenate(all_labels, axis=0)[: args.num_samples] # label_arr: generate_y
    predict_arr = np.concatenate(all_predict, axis=0)[: args.num_samples] # predict result
    
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")

        # save picture in "result.jpg"
        picture = arr[:SHOW_PIC_N]
        picture = th.from_numpy(picture)
        picture = picture.permute(0, 3, 1, 2)
        picture = th.cat([pic for pic in picture], 2)
        unloader = transforms.ToPILImage()
        unloader(picture).save(osp.join(logger.get_dir(), "result.jpg"))

        # save argparser json 
        args_path = os.path.join(logger.get_dir(), f"exp.json")
        info_json = json.dumps(vars(args), sort_keys=False, indent=4, separators=(' ', ':'))
        with open(args_path, 'w') as f:
            f.write(info_json)

        # copy code in case some result need to check it's historical implementation.
        shutil.copy(os.path.realpath(__file__), logger.get_dir())

        # print label and save data
        map_i_s = get_idex2name_map()
        for i, (y_, p_) in enumerate(zip(label_arr, predict_arr)):
            g_y = guide_y_np[i%args.batch_size] if args.guide_exp else guide_y_np[i]
            logger.log(f"original_label_{i}:{y_}, guide_y_{i%args.batch_size}:{g_y}, predict{i}:{p_}")
            logger.log(f"{map_i_s[y_]} ; {map_i_s[g_y]}; {map_i_s[p_]}")
        
        if guide_x_np is not None:
            np.savez(out_path, arr, label_arr, guide_y_np, predict_arr, guide_x_np)
        else:
            np.savez(out_path, arr, label_arr, guide_y_np, predict_arr)
    
    dist.barrier()
    logger.log("sampling complete")

if __name__ == "__main__":
    main()
