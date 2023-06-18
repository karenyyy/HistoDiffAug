import argparse, os, sys, glob, datetime, yaml

import cv2
import torch
import time
import numpy as np
from einops import rearrange
from tqdm import trange, tqdm
import torch.nn.functional as F

from omegaconf import OmegaConf
from PIL import Image

from ldm.models.autoencoder import AutoencoderKL
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.diffusionmodules.openaimodel import EncoderUNetModel
from ldm.util import instantiate_from_config

rescale = lambda x: (x + 1.) / 2.


def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def custom_to_np(x):
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample


def logs2pil(logs, keys=["sample"]):
    imgs = dict()
    for k in logs:
        try:
            if len(logs[k].shape) == 4:
                img = custom_to_pil(logs[k][0, ...])
            elif len(logs[k].shape) == 3:
                img = custom_to_pil(logs[k])
            else:
                print(f"Unknown format for key {k}. ")
                img = None
        except:
            img = None
        imgs[k] = img
    return imgs


@torch.no_grad()
def convsample(model, shape, return_intermediates=True,
               verbose=True,
               make_prog_row=False):
    if not make_prog_row:
        return model.p_sample_loop(None, shape,
                                   return_intermediates=return_intermediates, verbose=verbose)
    else:
        return model.progressive_denoising(
            None, shape, verbose=True
        )


@torch.no_grad()
def convsample_ddim(model, steps, shape, cond_fn, model_kwargs, eta=1.0):
    print('model, steps, shape, cond_fn, model_kwargs: ',
          steps, shape, cond_fn, model_kwargs)
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, cond_fn=cond_fn, model_kwargs=model_kwargs, eta=eta, verbose=False, )
    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(model, batch_size, cond_fn=None, model_kwargs=None, vanilla=False, custom_steps=None, eta=1.0, ):
    log = dict()

    shape = [batch_size,
             model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]

    with model.ema_scope("Plotting"):
        t0 = time.time()
        if vanilla:
            sample, progrow = convsample(model, shape,
                                         make_prog_row=True)
        else:
            sample, intermediates = convsample_ddim(model, steps=custom_steps, shape=shape,
                                                    cond_fn=cond_fn,
                                                    model_kwargs=model_kwargs,
                                                    eta=eta)

        t1 = time.time()

    x_sample = model.decode_first_stage(sample)

    # autoencoder = AutoencoderKL(
    #     embed_dim=3,
    #     monitor="val/rec_loss",
    #     # ckpt_path='/data/histo_diffusion_augmentation/autoencoder.ckpt',
    #     ckpt_path='/data/karenyyy/latent-diffusion4/logs/2023-02-25_04-33-20_autoencoder_histo_kl_64x64x3/autoencoder_4_13000.ckpt',
    #     # ckpt_path='/data/karenyyy/latent-diffusion4/logs/2023-02-23T23-36-06_autoencoder_histo_kl_64x64x3/diffusion_8_18000.ckpt',
    #     # ckpt_path='/data/karenyyy/latent-diffusion2/logs/2023-02-06_21-23-40_autoencoder_histo_kl_64x64x3/autoencoder_6_41000.ckpt',
    #     ddconfig={
    #         'double_z': True,
    #         'z_channels': 3,
    #         'resolution': 256,
    #         'in_channels': 3,
    #         'out_ch': 3,
    #         'ch': 128,
    #         'ch_mult': [1, 2, 4],
    #         'num_res_blocks': 2,
    #         'attn_resolutions': [],
    #         'dropout': 0.0
    #     },
    #     lossconfig={
    #         'target': torch.nn.Identity
    #     }
    #
    # )
    # autoencoder.eval()
    # autoencoder = autoencoder.to(sample.device)
    # print('1. / sample.flatten().std(): ', 1. / sample.flatten().std())
    # scale_factor = torch.tensor(0.1574)
    # sample = 1. / scale_factor * sample
    # x_sample = autoencoder.decode(sample)

    log["sample"] = x_sample
    log["time"] = t1 - t0
    log['throughput'] = sample.shape[0] / (t1 - t0)
    print(f'Throughput for this batch: {log["throughput"]}')
    return log


def run(idx, model, logdir, cond_fn=None, model_kwargs=None, batch_size=50, vanilla=False, custom_steps=None, eta=None, n_samples=50000,
        nplog=None):
    if vanilla:
        print(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
    else:
        print(f'Using DDIM sampling with {custom_steps} sampling steps and eta={eta}')

    tstart = time.time()
    n_saved = len(glob.glob(os.path.join(logdir, '*.png'))) - 1
    # path = logdir
    if model.cond_stage_model is None:
        all_images = []

        print(f"Running unconditional sampling for {n_samples} samples")
        for _ in trange(n_samples // batch_size, desc="Sampling Batches (unconditional)"):
            logs = make_convolutional_sample(model, batch_size=batch_size,
                                             cond_fn=cond_fn,
                                             model_kwargs=model_kwargs,
                                             vanilla=vanilla, custom_steps=custom_steps,
                                             eta=eta)
            n_saved = save_logs(logs, logdir, n_saved=n_saved, key="sample")
            all_images.extend([custom_to_np(logs["sample"])])
            if n_saved >= n_samples:
                print(f'Finish after generating {n_saved} samples')
                break
        all_img = np.concatenate(all_images, axis=0)
        all_img = all_img[:n_samples]
        shape_str = "x".join([str(x) for x in all_img.shape])
        labels = ''.join([str(i.item()) for i in model_kwargs['y']])

        nppath = os.path.join('./saved_samples', f"{idx}-samples_{labels}.npz")

        np.savez(nppath, all_img)
        print(f'{nppath} saved!!!')
    else:
        raise NotImplementedError('Currently only sampling for unconditional models supported.')

    print(f"sampling of {n_saved} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")


def save_logs(logs, path, n_saved=0, key="sample", np_path=None):
    for k in logs:
        if k == key:
            batch = logs[key]
            if np_path is None:
                for x in batch:
                    img = custom_to_pil(x)
                    imgpath = os.path.join(path, f"{key}_{n_saved:06}.png")
                    img.save(imgpath)
                    n_saved += 1
            else:
                npbatch = custom_to_np(batch)
                shape_str = "x".join([str(x) for x in npbatch.shape])
                nppath = os.path.join(np_path, f"{n_saved}-{shape_str}-samples.npz")
                np.savez(nppath, npbatch)
                n_saved += npbatch.shape[0]
    return n_saved


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        nargs="?",
        help="number of samples to draw",
        default=50000
    )
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=1.0
    )
    parser.add_argument(
        "-v",
        "--vanilla_sample",
        default=False,
        action='store_true',
        help="vanilla sampling (default option is DDIM sampling)?",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default="none"
    )
    parser.add_argument(
        "-c",
        "--custom_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fastdpm sampling",
        default=50
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        help="the bs",
        default=10
    )
    return parser


def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd, strict=False)
    model.to('cuda:1')
    model.eval()
    return model


def load_model(config, ckpt, gpu, eval_mode):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"])

    return model, global_step


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sys.path.append(os.getcwd())
    command = " ".join(sys.argv)

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    ckpt = None

    if not os.path.exists(opt.resume):
        raise ValueError("Cannot find {}".format(opt.resume))
    if os.path.isfile(opt.resume):
        # paths = opt.resume.split("/")
        try:
            logdir = '/'.join(opt.resume.split('/')[:-1])
            # idx = len(paths)-paths[::-1].index("logs")+1
            print(f'Logdir is {logdir}')
        except ValueError:
            paths = opt.resume.split("/")
            idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
        ckpt = opt.resume
    else:
        assert os.path.isdir(opt.resume), f"{opt.resume} is not a directory"
        logdir = opt.resume.rstrip("/")
        ckpt = os.path.join(logdir, "model.ckpt")

    # base_configs = sorted(glob.glob(os.path.join(logdir, "config.yaml")))
    base_configs = ['configs/latent-diffusion/histo-ldm-kl-8.yaml']
    opt.base = base_configs

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    gpu = True
    eval_mode = True

    if opt.logdir != "none":
        locallog = logdir.split(os.sep)[-1]
        if locallog == "": locallog = logdir.split(os.sep)[-2]
        print(f"Switching logdir from '{logdir}' to '{os.path.join(opt.logdir, locallog)}'")
        logdir = os.path.join(opt.logdir, locallog)

    print(config)

    model, global_step = load_model(config, ckpt, gpu, eval_mode)
    print(f"global step: {global_step}")
    print(75 * "=")
    print("logging to:")
    logdir = os.path.join(logdir, "samples", f"{global_step:08}", now)
    imglogdir = os.path.join(logdir, "img")
    numpylogdir = os.path.join(logdir, "numpy")

    os.makedirs(imglogdir)
    os.makedirs(numpylogdir)
    print(logdir)
    print(75 * "=")

    # write config out
    sampling_file = os.path.join(logdir, "sampling_config.yaml")
    sampling_conf = vars(opt)

    with open(sampling_file, 'w') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)
    print(sampling_conf)

    classifier = EncoderUNetModel(
        image_size=64,
        in_channels=3,
        model_channels=128,
        out_channels=9,
        num_res_blocks=2,
        attention_resolutions=(8, 16),
        channel_mult=(1, 1, 2, 2, 4, 4),
        use_fp16=False,
        num_head_channels=64,
        use_scale_shift_norm=True,
        resblock_updown=True,
        pool='attention',
    )

    pl_sd = torch.load('saved_crc9_5percent_unet_classifier_ckpts/model_010000.pt', map_location="cpu")

    classifier.load_state_dict(pl_sd)
    classifier.eval()


    def cond_fn(x, t, y=None):

        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            classifier.to(x_in.device)

            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return torch.autograd.grad(selected.sum(), x_in)[0]


    for idx in range(100):
        model_kwargs = {}
        lb = 0
        classes = torch.tensor([lb]*10)
        model_kwargs["y"] = classes

        # with torch.no_grad():
        #
        #     samples, x0 = model.progressive_denoising(None, shape=(3, 64, 64), batch_size=10,
        #                                                model_kwargs=model_kwargs,
        #                                                cond_fn=cond_fn)
        #     print('samples, x0: ', samples.size(), [x.size() for x in x0])
        #     x_sample = model.decode_first_stage(samples, force_not_quantize=False)
        #     print('x_sample: ', x_sample.size())
        #
        #     nppath = f"saved_samples/tmp_1_ddpm.npz"
        #     np.savez(nppath, x_sample.data.cpu().numpy())
        #     print(f'{nppath} saved!!!')

        logs = make_convolutional_sample(model, batch_size=opt.batch_size,
                                         cond_fn=cond_fn,
                                         model_kwargs=model_kwargs,
                                         vanilla=opt.vanilla_sample, custom_steps=opt.custom_steps,
                                         eta=opt.eta)

        run(idx=1, model=model, logdir=imglogdir, cond_fn=cond_fn, model_kwargs=model_kwargs, eta=opt.eta,
            vanilla=opt.vanilla_sample, n_samples=opt.n_samples, custom_steps=opt.custom_steps,
            batch_size=opt.batch_size, nplog=numpylogdir)

        npz_path = os.path.join('./saved_samples', f"{idx}-samples_{lb}.npz")
        images = np.load(npz_path, allow_pickle=True)
        images = images["arr_0"]

        # images = rearrange(images, 'b c h w -> b h w c')
        # print(images)
        # norm_images = cv2.normalize(images, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # norm_images = norm_images.astype(np.uint8)
        # print(norm_images)

        # print('num_samples: ', images.shape)
        for img_idx in range(images.shape[0]):
            img = Image.fromarray(np.asarray(images[img_idx, :, :, :]).astype(np.uint8))
            img.save(f'./fake_examples_crc5/fake_{lb}_{idx*images.shape[0]+img_idx}.png')
