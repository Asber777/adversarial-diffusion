"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os

import shutil
import json
import numpy as np
import torch as th
import os.path as osp
import datetime
import torch.distributed as dist
import torch.nn.functional as F

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

'''
SAMPLE_FLAGS="--batch_size 5 --num_samples 5 --timestep_respacing 250"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 500 --dropout 0.1 --image_size 64 --learn_sigma True\
--noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_new_attention_order True \
--use_fp16 True --use_scale_shift_norm True"
python guided-diffusion/scripts/adversarial_sample.py $MODEL_FLAGS --classifier_scale 1.0 --classifier_path 64x64_classifier.pt --classifier_depth 4 \
--model_path 64x64_diffusion.pt $SAMPLE_FLAGS  --classifier_scale 1.0 --adv_scale 0.0 --describe ""
'''

def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    dir = osp.join('/root/hhtpro/123/result',
            args.describe,
            datetime.datetime.now().strftime("adv-%Y-%m-%d-%H-%M-%S-%f"),
        )
    logger.configure(dir)

    logger.log("creating model and diffusion...")
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
    # modify conf_fn and model_fn to get adv, conf_fn
    def cond_fn(x, t, y=None, adv_y=None,):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            logits, hidden = classifier(x_in, t, args.get_hidden, args.get_middle)
            
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return -th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    def model_fn(x, t, y=None, adv_y=None,):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("sampling...")
    all_images = []
    all_labels = []
    all_foolys = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        classes = th.randint(
            low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
        )
        fool_classes = th.randint(
            low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
        )
        model_kwargs["y"] = classes
        model_kwargs["adv_y"] = fool_classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_labels, classes)
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        all_foolys.extend(fool_classes.cpu().numpy())
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: args.num_samples]
    fool_label_arr = all_foolys[:args.num_samples]
    
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr, fool_label_arr)
        args_path = os.path.join(logger.get_dir(), f"exp.json")
        info_json = json.dumps(vars(args), sort_keys=False, indent=4, separators=(' ', ':'))
        with open(args_path, 'w') as f:
            f.write(info_json)
        shutil.copy('/root/hhtpro/123/guided-diffusion/scripts/adversarial_sample.py', logger.get_dir())

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=5,
        batch_size=5,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
        adv_scale=0.0,
        get_hidden=True,
        get_middle=True, 
        describe="default_desc"
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
