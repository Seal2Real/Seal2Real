import os
from pipeline import StableEditPipeline
from datasets import RealDataset, FakeDataset
import argparse
import numpy as np
from torch.utils.data import DataLoader
import torch
from diffusers.schedulers import DDIMScheduler
import torch.nn.functional as F
from torchvision.utils import save_image
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--device',type=str, default='cuda')
    parser.add_argument('--world_size', type=int, default=1)
    args = parser.parse_args()
    return args

def main():
    args = parse()
    if args.world_size == 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '5679'
        os.environ["RANK"] = "0"
        os.environ['WORLD_SIZE'] = '1'
    torch.cuda.set_device(args.local_rank)
    print("my rank",args.local_rank)
    torch.distributed.init_process_group(backend='nccl')

    seed = np.random.randint(1, 10000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device(args.device)

    model_path = "/placeholder/path/to/model"
    generator = torch.Generator(device).manual_seed(2056)

    tmp_path = './placeholder_tmp_dir'
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)
    if not os.path.exists(tmp_path+'/train_vis'):
        os.makedirs(tmp_path+'/train_vis')

    iteration_max = 1000000

    original_prompt = 'an image with a seal'
    real_prompt = 'a real seal'
    fake_prompt = 'a fake seal'

    pipe = StableEditPipeline.from_pretrained(
        pretrained_model_name_or_path=model_path,safety_checker=None,
        scheduler=DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                set_alpha_to_one=False),
    ).to(device)

    optimizer_diff = torch.optim.Adam(
        pipe.unet.parameters(),  
        lr=2e-6,
    )

    print("start loading")
    fake_embedding = torch.load("/placeholder/path/to/fake_embedding.pth", map_location='cpu')
    fake_embedding = fake_embedding.to(device)
    real_embedding = torch.load("/placeholder/path/to/real_embedding.pth", map_location='cpu')
    real_embedding = real_embedding.to(device)

    fake_embedding = torch.nn.Parameter(fake_embedding, requires_grad=True)

    real_embedding = torch.nn.Parameter(real_embedding, requires_grad=True)

    optimizer_prompt = torch.optim.Adam(
        [fake_embedding, real_embedding],
        lr=2e-3,
    )

    real_dataset = RealDataset(dir_train="/placeholder/path/to/real_data")
    real_datasampler = torch.utils.data.distributed.DistributedSampler(real_dataset, shuffle=True)
    fake_dataset = FakeDataset()
    fake_datasampler = torch.utils.data.distributed.DistributedSampler(fake_dataset, shuffle=True)
    batchsize = 3
    real_dataloader = DataLoader(dataset=real_dataset, batch_size=batchsize, sampler=real_datasampler)
    fake_dataloader = DataLoader(dataset=fake_dataset, batch_size=batchsize, sampler=fake_datasampler)

    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.eval()
    pipe.text_encoder.eval()
    pipe.unet.requires_grad_(True)
    pipe.unet.train()
    iteration = 1162
    bs = 3
    while (iteration < iteration_max):

        num = 0
        real_datasampler.set_epoch(iteration)
        fake_datasampler.set_epoch(iteration)
        for x, y in zip(fake_dataloader, real_dataloader):
            num = num + batchsize
            if args.local_rank == 0 and num % len(real_dataset)//(args.world_size*2) == 0 and iteration%5==0:

                if iteration%50==0:
                    save_image(init_real_image,tmp_path + '/train_vis/real-train-{}.jpg'.format(iteration))
                else:
                    save_image(init_real_image,tmp_path + '/train_vis/real-train-new.jpg')

            init_fake_image, init_fake_label = x
            init_real_image = y.to(device)
            init_fake_image, init_fake_label = init_fake_image.to(device), init_fake_label.to(device)

            latents_dtype = real_embedding.dtype

            init_image = init_real_image.to(device=device, dtype=latents_dtype)
            init_latent_image_dist = pipe.vae.encode(init_image).latent_dist
            init_image_latents = init_latent_image_dist.sample(generator=generator)
            init_image_latents = 0.18215 * init_image_latents
            init_image_latents = init_image_latents.repeat((bs,1,1,1))
            noise = torch.randn(init_image_latents.shape).to(init_image_latents.device)
            timesteps = torch.randint(1000, (noise.shape[0],), device=init_image_latents.device)

            noisy_latents = pipe.scheduler.add_noise(init_image_latents, noise, timesteps)

            noise_pred = pipe.unet(noisy_latents, timesteps, real_embedding.repeat((noise.shape[0],1,1))).sample
            loss_real_diff = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()

            if args.local_rank == 0 and num % len(real_dataset)//(args.world_size*2) == 0 and iteration%5==1:
                with torch.no_grad():
                    noise_500 = torch.randn(init_image_latents.shape).to(init_image_latents.device)
                    timesteps_500 = torch.randint(500,501, (noise_500.shape[0],), device=init_image_latents.device)
                    noisy_latents_500 = pipe.scheduler.add_noise(init_image_latents, noise_500, timesteps_500)
                    noise_pred_500 = pipe.unet(noisy_latents_500, timesteps_500, real_embedding.repeat((noise_500.shape[0],1,1))).sample
                    loss_real_diff_500 = F.mse_loss(noise_pred_500, noise_500, reduction="none").mean([1, 2, 3]).mean()
                    print('train diffusion model: num-{} iteration-{} loss 500-{}'.format(num, iteration,loss_real_diff_500))

            loss_real_diff.backward()
            optimizer_prompt.step()
            optimizer_prompt.zero_grad()

            optimizer_diff.step()
            optimizer_diff.zero_grad()

        if args.local_rank == 0 and iteration % 50 == 0:
            torch.save(pipe.unet.state_dict(),'/placeholder/path/to/save/unet/' + f'unet_{iteration}.pth')

            torch.save(real_embedding, '/placeholder/path/to/save/real_embedding.pth')

        if args.local_rank == 0 and iteration % 5 == 0:

            save_real_image = pipe(text_embeddings=real_embedding).images
            if iteration%50==0:
                save_real_image[0].save(tmp_path + '/real_infer-{}.jpg'.format(iteration))
            else:
                save_real_image[0].save(tmp_path + '/real_infer-new.jpg')

        iteration += 1

if __name__ == '__main__':
    main()