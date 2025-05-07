import PIL
from typing import List, Optional, Union
import warnings
import torch 
import inspect

import numpy as np 
import torch.nn.functional as F

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import DDIMScheduler, PNDMScheduler
from diffusers.utils import logging
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from scheduler import LMSDiscreteScheduler
from torch import autocast
from PIL import Image
from torch.utils.data import DataLoader
from datasets import RealDataset, FakeDataset

from utils import init_attention_weights, init_attention_edit, init_attention_func, \
                    use_last_tokens_attention, use_last_tokens_attention_weights, \
                    use_last_self_attention, save_last_tokens_attention, save_last_self_attention

import argparse


from packaging import version
if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }

def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  
    image = image.resize((w, h), resample=PIL_INTERPOLATION["lanczos"])
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--device',type=str, default='cuda')
    args = parser.parse_args()
    return args


class StableEditPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
    
    def train_fake(self,
              prompt: Union[str, List[str]],
              height: Optional[int] = 256,
              width: Optional[int] = 256,
              generator: Optional[torch.Generator] = None,
              embedding_learning_rate: float = 0.001,
              diffusion_model_learning_rate: float = 2e-6,
              text_embedding_optimization_steps: int = 4000,
              model_fine_tuning_optimization_steps: int = 1000,
              **kwargs,
              ):

        args = parse()



        real_dataset = RealDataset(dir_train="/placeholder/path/to/real_data")
        real_datasampler = torch.utils.data.distributed.DistributedSampler(real_dataset, shuffle=True)
        fake_dataset = FakeDataset(dir_train="/placeholder/path/to/fake_data")
        fake_datasampler = torch.utils.data.distributed.DistributedSampler(fake_dataset, shuffle=True)
        batchsize = 1
        real_dataloader = DataLoader(dataset=real_dataset, batch_size=batchsize, sampler=real_datasampler)
        fake_dataloader = DataLoader(dataset=fake_dataset, batch_size=batchsize, sampler=fake_datasampler)
        device = torch.device(args.device)




        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

            
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.eval()
        self.vae.eval()
        self.text_encoder.eval()

        
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = torch.nn.Parameter(
            self.text_encoder(text_input.input_ids.to(self.device))[0], requires_grad=True
        )
        text_embeddings = text_embeddings.detach()
        text_embeddings.requires_grad_()
        text_embeddings_orig = text_embeddings.clone()

        
        optimizer = torch.optim.AdamW(
            [text_embeddings],  
            lr=embedding_learning_rate,
        )



        global_step = 0
        loss_sum = .0


        print("First optimizing the text embedding to better reconstruct the init image")
        for i in range(text_embedding_optimization_steps):
            num1 = 0
            for _, x in enumerate(fake_dataloader):
                num1 = num1 + 1*6
                print('num:{} \n'.format(num1))
                print('embedding, fake \n')
                init_fake_image, init_fake_label = x
                init_image = init_fake_image
                if isinstance(init_image, PIL.Image.Image):
                    init_image = preprocess(init_image)

                latents_dtype = text_embeddings.dtype
                init_image = init_image.to(device=self.device, dtype=latents_dtype)
                init_latent_image_dist = self.vae.encode(init_image).latent_dist
                init_image_latents = init_latent_image_dist.sample(generator=generator)
                init_image_latents = 0.18215 * init_image_latents

                evaluation_time_steps = torch.randint(1000, (1,), device=init_image_latents.device)

                progress_bar = tqdm(range(text_embedding_optimization_steps))
                progress_bar.set_description("Steps")


                
                noise = torch.randn(init_image_latents.shape).to(init_image_latents.device)
                timesteps = torch.randint(1000, (1,), device=init_image_latents.device)

                
                
                noisy_latents = self.scheduler.add_noise(init_image_latents, noise, timesteps)

                
                noise_pred = self.unet(noisy_latents, timesteps, text_embeddings).sample
                loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()

                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                global_step += 1

                loss_sum += loss.detach().item()
                logs = {"loss": loss_sum / global_step}
                progress_bar.set_postfix(**logs)

                if num1 >= 4000:
                    break

                if global_step % 1000 == 0:
                    image = self(
                        text_embeddings=text_embeddings,
                        generator=generator,
                    ).images[0]
                    image.save('./placeholder_tmp_dir/text_inver_' + str(global_step) + '.png')
                    self.text_embedding_evaluate(
                        text_embeddings=text_embeddings,
                        init_image_latents=init_image_latents,
                        evaluation_time_steps=evaluation_time_steps
                    )



        text_embeddings.requires_grad_(False)
        

        self.unet.requires_grad_(True)
        self.unet.train()
        optimizer1 = torch.optim.Adam(
            self.unet.parameters(),  
            lr=diffusion_model_learning_rate,
        )
        loss_sum = .0


        progress_bar = tqdm(range(model_fine_tuning_optimization_steps))
        for idx in range(model_fine_tuning_optimization_steps):
            
            print('unet, fake')
            num1 = 0
            for _, x in enumerate(fake_dataloader):
                init_fake_image, init_fake_label = x
                init_image = init_fake_image
                num1 = num1 + 6

                latents_dtype = text_embeddings.dtype
                init_image = init_image.to(device=self.device, dtype=latents_dtype)
                init_latent_image_dist = self.vae.encode(init_image).latent_dist
                init_image_latents = init_latent_image_dist.sample(generator=generator)
                init_image_latents = 0.18215 * init_image_latents

                noise = torch.randn(init_image_latents.shape).to(init_image_latents.device)
                timesteps = torch.randint(1000, (1,), device=init_image_latents.device)
                evaluation_time_steps = torch.randint(1000, (1,), device=init_image_latents.device)

                
                
                noisy_latents = self.scheduler.add_noise(init_image_latents, noise, timesteps)

                
                noise_pred = self.unet(noisy_latents, timesteps, text_embeddings).sample

                loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
                loss.backward()
                optimizer1.step()
                optimizer1.zero_grad()

                progress_bar.update(1)
                loss_sum += loss.detach().item()
                logs = {"loss": loss_sum / (idx + 1)}
                progress_bar.set_postfix(**logs)

                
                if (idx + 1) % 1000 == 0:
                    image = self(
                        text_embeddings=text_embeddings,
                        generator=generator,
                    ).images[0]
                    image.save('./placeholder_tmp_dir/model_tune_' + str(idx + 1) + '.png')

                    self.text_embedding_evaluate(
                        text_embeddings=text_embeddings,
                        init_image_latents=init_image_latents,
                        evaluation_time_steps=evaluation_time_steps
                    )
                if num1 >= 1000:
                    break

    def train_real(self,
                   prompt: Union[str, List[str]],
                   height: Optional[int] = 256,
                   width: Optional[int] = 256,
                   generator: Optional[torch.Generator] = None,
                   embedding_learning_rate: float = 0.001,
                   diffusion_model_learning_rate: float = 2e-6,
                   text_embedding_optimization_steps: int = 4000,
                   model_fine_tuning_optimization_steps: int = 1000,
                   **kwargs,
                   ):

        args = parse()



        real_dataset = RealDataset(dir_train="/placeholder/path/to/real_data")
        real_datasampler = torch.utils.data.distributed.DistributedSampler(real_dataset, shuffle=True)
        fake_dataset = FakeDataset(dir_train="/placeholder/path/to/fake_data")
        fake_datasampler = torch.utils.data.distributed.DistributedSampler(fake_dataset, shuffle=True)
        batchsize = 1
        real_dataloader = DataLoader(dataset=real_dataset, batch_size=batchsize, sampler=real_datasampler)
        fake_dataloader = DataLoader(dataset=fake_dataset, batch_size=batchsize, sampler=fake_datasampler)
        device = torch.device(args.device)

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

            
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.eval()
        self.vae.eval()
        self.text_encoder.eval()

        
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = torch.nn.Parameter(
            self.text_encoder(text_input.input_ids.to(self.device))[0], requires_grad=True
        )
        text_embeddings = text_embeddings.detach()
        text_embeddings.requires_grad_()
        text_embeddings_orig = text_embeddings.clone()

        
        optimizer = torch.optim.AdamW(
            [text_embeddings],  
            lr=embedding_learning_rate,
        )

        global_step = 0
        loss_sum = .0

        print("First optimizing the text embedding to better reconstruct the init image")
        for i in range(text_embedding_optimization_steps):
            num1 = 0
            for _, y in enumerate(real_dataloader):
                print('embedding, real  \n')
                print('num, {}  \n'.format(num1))
                num1 = num1 + 6
                init_real_image = y
                init_image = init_real_image
                if isinstance(init_image, PIL.Image.Image):
                    init_image = preprocess(init_image)

                latents_dtype = text_embeddings.dtype
                init_image = init_image.to(device=self.device, dtype=latents_dtype)
                init_latent_image_dist = self.vae.encode(init_image).latent_dist
                init_image_latents = init_latent_image_dist.sample(generator=generator)
                init_image_latents = 0.18215 * init_image_latents

                evaluation_time_steps = torch.randint(1000, (1,), device=init_image_latents.device)

                progress_bar = tqdm(range(text_embedding_optimization_steps))
                progress_bar.set_description("Steps")

                
                noise = torch.randn(init_image_latents.shape).to(init_image_latents.device)
                timesteps = torch.randint(1000, (1,), device=init_image_latents.device)

                
                
                noisy_latents = self.scheduler.add_noise(init_image_latents, noise, timesteps)

                
                noise_pred = self.unet(noisy_latents, timesteps, text_embeddings).sample
                loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()

                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                global_step += 1

                loss_sum += loss.detach().item()
                logs = {"loss": loss_sum / global_step}
                progress_bar.set_postfix(**logs)

                if global_step % 1000 == 0:
                    image = self(
                        text_embeddings=text_embeddings,
                        generator=generator,
                    ).images[0]
                    image.save('./placeholder_tmp_dir/text_inver_' + str(global_step) + '.png')
                    self.text_embedding_evaluate(
                        text_embeddings=text_embeddings,
                        init_image_latents=init_image_latents,
                        evaluation_time_steps=evaluation_time_steps
                    )
                if num1 >= 198:
                    break

        text_embeddings.requires_grad_(False)
        

        self.unet.requires_grad_(True)
        self.unet.train()
        optimizer1 = torch.optim.Adam(
            self.unet.parameters(),  
            lr=diffusion_model_learning_rate,
        )
        loss_sum = .0

        progress_bar = tqdm(range(model_fine_tuning_optimization_steps))
        for idx in range(model_fine_tuning_optimization_steps):
            num1 = 0
            
            for _, y in enumerate(real_dataloader):
                print('unet, real')
                num1 = num1 + 6
                init_real_image = y
                init_image = init_real_image

                latents_dtype = text_embeddings.dtype
                init_image = init_image.to(device=self.device, dtype=latents_dtype)
                init_latent_image_dist = self.vae.encode(init_image).latent_dist
                init_image_latents = init_latent_image_dist.sample(generator=generator)
                init_image_latents = 0.18215 * init_image_latents

                noise = torch.randn(init_image_latents.shape).to(init_image_latents.device)
                timesteps = torch.randint(1000, (1,), device=init_image_latents.device)
                evaluation_time_steps = torch.randint(1000, (1,), device=init_image_latents.device)

                
                
                noisy_latents = self.scheduler.add_noise(init_image_latents, noise, timesteps)

                
                noise_pred = self.unet(noisy_latents, timesteps, text_embeddings).sample

                loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
                loss.backward()
                optimizer1.step()
                optimizer1.zero_grad()

                progress_bar.update(1)
                loss_sum += loss.detach().item()
                logs = {"loss": loss_sum / (idx + 1)}
                progress_bar.set_postfix(**logs)

                
                if (idx + 1) % 1000 == 0:
                    image = self(
                        text_embeddings=text_embeddings,
                        generator=generator,
                    ).images[0]
                    image.save('./placeholder_tmp_dir/model_tune_' + str(idx + 1) + '.png')

                    self.text_embedding_evaluate(
                        text_embeddings=text_embeddings,
                        init_image_latents=init_image_latents,
                        evaluation_time_steps=evaluation_time_steps
                    )
                if num1 >= 198:
                    break



    def get_all_fake_embedding(self,
              prompt: Union[str, List[str]],
              height: Optional[int] = 256,
              width: Optional[int] = 256,
              generator: Optional[torch.Generator] = None,
              embedding_learning_rate: float = 0.001,
              diffusion_model_learning_rate: float = 2e-6,
              text_embedding_optimization_steps: int = 4000,
              **kwargs,
              ):

        args = parse()



        real_dataset = RealDataset(dir_train="/placeholder/path/to/real_data")
        real_datasampler = torch.utils.data.distributed.DistributedSampler(real_dataset, shuffle=True)
        fake_dataset = FakeDataset(dir_train="/placeholder/path/to/fake_data")
        fake_datasampler = torch.utils.data.distributed.DistributedSampler(fake_dataset, shuffle=True)
        batchsize = 1
        real_dataloader = DataLoader(dataset=real_dataset, batch_size=batchsize, sampler=real_datasampler)
        fake_dataloader = DataLoader(dataset=fake_dataset, batch_size=batchsize, sampler=fake_datasampler)
        device = torch.device(args.device)




        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

            
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.eval()
        self.vae.eval()
        self.text_encoder.eval()

        
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = torch.nn.Parameter(
            self.text_encoder(text_input.input_ids.to(self.device))[0], requires_grad=True
        )
        text_embeddings = text_embeddings.detach()
        text_embeddings.requires_grad_()
        text_embeddings_orig = text_embeddings.clone()

        
        optimizer = torch.optim.AdamW(
            [text_embeddings],  
            lr=embedding_learning_rate,
        )



        global_step = 0
        loss_sum = .0


        print("First optimizing the text embedding to better reconstruct the init image")
        for i in range(text_embedding_optimization_steps):
            num1 = 0
            for _, x in enumerate(fake_dataloader):
                num1 = num1 + 1*6
                print('num:{} \n'.format(num1))
                print('embedding, fake \n')
                init_fake_image, init_fake_label = x
                init_image = init_fake_image
                if isinstance(init_image, PIL.Image.Image):
                    init_image = preprocess(init_image)

                latents_dtype = text_embeddings.dtype
                init_image = init_image.to(device=self.device, dtype=latents_dtype)
                init_latent_image_dist = self.vae.encode(init_image).latent_dist
                init_image_latents = init_latent_image_dist.sample(generator=generator)
                init_image_latents = 0.18215 * init_image_latents

                evaluation_time_steps = torch.randint(1000, (1,), device=init_image_latents.device)

                progress_bar = tqdm(range(text_embedding_optimization_steps))
                progress_bar.set_description("Steps")


                
                noise = torch.randn(init_image_latents.shape).to(init_image_latents.device)
                timesteps = torch.randint(1000, (1,), device=init_image_latents.device)

                
                
                noisy_latents = self.scheduler.add_noise(init_image_latents, noise, timesteps)

                
                noise_pred = self.unet(noisy_latents, timesteps, text_embeddings).sample
                loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()

                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                global_step += 1

                loss_sum += loss.detach().item()
                logs = {"loss": loss_sum / global_step}
                progress_bar.set_postfix(**logs)

                if num1 >= 5000: 
                    break

                if global_step % 1000 == 0:
                    image = self(
                        text_embeddings=text_embeddings,
                        generator=generator,
                    ).images[0]
                    image.save('./placeholder_tmp_dir/text_inver_' + str(global_step) + '.png')
                    self.text_embedding_evaluate(
                        text_embeddings=text_embeddings,
                        init_image_latents=init_image_latents,
                        evaluation_time_steps=evaluation_time_steps
                    )
        return text_embeddings

    def get_all_real_embedding(self,
                   prompt: Union[str, List[str]],
                   height: Optional[int] = 256,
                   width: Optional[int] = 256,
                   generator: Optional[torch.Generator] = None,
                   embedding_learning_rate: float = 0.001,
                   diffusion_model_learning_rate: float = 2e-6,
                   text_embedding_optimization_steps: int = 4000,
                   **kwargs,
                   ):

        args = parse()



        real_dataset = RealDataset(dir_train="/placeholder/path/to/real_data")
        real_datasampler = torch.utils.data.distributed.DistributedSampler(real_dataset, shuffle=True)
        fake_dataset = FakeDataset(dir_train="/placeholder/path/to/fake_data")
        fake_datasampler = torch.utils.data.distributed.DistributedSampler(fake_dataset, shuffle=True)
        batchsize = 1
        real_dataloader = DataLoader(dataset=real_dataset, batch_size=batchsize, sampler=real_datasampler)
        fake_dataloader = DataLoader(dataset=fake_dataset, batch_size=batchsize, sampler=fake_datasampler)
        device = torch.device(args.device)

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

            
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.eval()
        self.vae.eval()
        self.text_encoder.eval()

        
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = torch.nn.Parameter(
            self.text_encoder(text_input.input_ids.to(self.device))[0], requires_grad=True
        )
        text_embeddings = text_embeddings.detach()
        text_embeddings.requires_grad_()
        text_embeddings_orig = text_embeddings.clone()

        
        optimizer = torch.optim.AdamW(
            [text_embeddings],  
            lr=embedding_learning_rate,
        )

        global_step = 0
        loss_sum = .0

        print("First optimizing the text embedding to better reconstruct the init image")
        for i in range(text_embedding_optimization_steps):
            num1 = 0
            for _, y in enumerate(real_dataloader):
                print('embedding, real  \n')
                print('num, {}  \n'.format(num1))
                num1 = num1 + 6
                init_real_image = y
                init_image = init_real_image
                if isinstance(init_image, PIL.Image.Image):
                    init_image = preprocess(init_image)

                latents_dtype = text_embeddings.dtype
                init_image = init_image.to(device=self.device, dtype=latents_dtype)
                init_latent_image_dist = self.vae.encode(init_image).latent_dist
                init_image_latents = init_latent_image_dist.sample(generator=generator)
                init_image_latents = 0.18215 * init_image_latents

                evaluation_time_steps = torch.randint(1000, (1,), device=init_image_latents.device)

                progress_bar = tqdm(range(text_embedding_optimization_steps))
                progress_bar.set_description("Steps")

                
                noise = torch.randn(init_image_latents.shape).to(init_image_latents.device)
                timesteps = torch.randint(1000, (1,), device=init_image_latents.device)

                
                
                noisy_latents = self.scheduler.add_noise(init_image_latents, noise, timesteps)

                
                noise_pred = self.unet(noisy_latents, timesteps, text_embeddings).sample
                loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()

                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                global_step += 1

                loss_sum += loss.detach().item()
                logs = {"loss": loss_sum / global_step}
                progress_bar.set_postfix(**logs)

                if global_step % 1000 == 0:
                    image = self(
                        text_embeddings=text_embeddings,
                        generator=generator,
                    ).images[0]
                    image.save('./placeholder_tmp_dir/text_inver_' + str(global_step) + '.png')
                    self.text_embedding_evaluate(
                        text_embeddings=text_embeddings,
                        init_image_latents=init_image_latents,
                        evaluation_time_steps=evaluation_time_steps
                    )
                if num1 >= 198: 
                    break
        return text_embeddings


    def train_text_embedding(
        self,
        prompt: Union[str, List[str]],
        init_image: Union[torch.FloatTensor, PIL.Image.Image],
        height: Optional[int] = 256,
        width: Optional[int] = 256,
        generator: Optional[torch.Generator] = None,
        embedding_learning_rate: float = 0.001,
        text_embedding_optimization_steps: int = 4000,
        **kwargs,
    ): 
        if "torch_device" in kwargs:
            device = kwargs.pop("torch_device")
            warnings.warn(
                "`torch_device` is deprecated as an input argument to `__call__` and will be removed in v0.3.0."
                " Consider using `pipe.to(torch_device)` instead."
            )

            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.to(device)

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.") 

        
        
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.eval()
        self.vae.eval()
        self.text_encoder.eval() 

        
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = torch.nn.Parameter(
            self.text_encoder(text_input.input_ids.to(self.device))[0], requires_grad=True
        )
        text_embeddings = text_embeddings.detach()
        text_embeddings.requires_grad_()
        text_embeddings_orig = text_embeddings.clone() 

        
        optimizer = torch.optim.AdamW(
            [text_embeddings],  
            lr=embedding_learning_rate,
        )

        if isinstance(init_image, PIL.Image.Image):
            init_image = preprocess(init_image) 

        latents_dtype = text_embeddings.dtype
        init_image = init_image.to(device=self.device, dtype=latents_dtype)
        init_latent_image_dist = self.vae.encode(init_image).latent_dist
        init_image_latents = init_latent_image_dist.sample(generator=generator)
        init_image_latents = 0.18215 * init_image_latents

        progress_bar = tqdm(range(text_embedding_optimization_steps))
        progress_bar.set_description("Steps")

        global_step = 0
        loss_sum = .0
        evaluation_time_steps = torch.randint(1000, (1,), device=init_image_latents.device) 

        print("First optimizing the text embedding to better reconstruct the init image") 
        for _ in range(text_embedding_optimization_steps):
            
            
            noise = torch.randn(init_image_latents.shape).to(init_image_latents.device)
            timesteps = torch.randint(1000, (1,), device=init_image_latents.device)

            
            
            noisy_latents = self.scheduler.add_noise(init_image_latents, noise, timesteps)
            
            
            noise_pred = self.unet(noisy_latents, timesteps, text_embeddings).sample
            loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1 

            loss_sum += loss.detach().item()
            logs = {"loss": loss_sum / global_step}
            progress_bar.set_postfix(**logs) 

            if global_step % 500 == 0: 
                image = self(
                    text_embeddings=text_embeddings,
                    generator=generator,
                ).images[0] 
                image.save('./placeholder_tmp_dir/text_inver_' + str(global_step) + '.png')
                self.text_embedding_evaluate(
                    text_embeddings=text_embeddings,
                    init_image_latents=init_image_latents,
                    evaluation_time_steps=evaluation_time_steps 
                )

        text_embeddings.requires_grad_(False)
        return (text_embeddings, text_embeddings_orig, init_image_latents)


    def train_model(
        self,
        text_embeddings,
        init_image_latents, 
        generator: Optional[torch.Generator] = None,
        diffusion_model_learning_rate: float = 2e-6,
        model_fine_tuning_optimization_steps: int = 1000,
    ):
        self.unet.requires_grad_(True)
        self.unet.train()
        optimizer = torch.optim.Adam(
            self.unet.parameters(),  
            lr=diffusion_model_learning_rate,
        )

        loss_sum = .0 
        evaluation_time_steps = torch.randint(1000, (1,), device=init_image_latents.device) 

        progress_bar = tqdm(range(model_fine_tuning_optimization_steps))

        
        

        for idx in range(model_fine_tuning_optimization_steps): 
            
            noise = torch.randn(init_image_latents.shape).to(init_image_latents.device)
            timesteps = torch.randint(1000, (1,), device=init_image_latents.device)

            
            
            noisy_latents = self.scheduler.add_noise(init_image_latents, noise, timesteps)

            
            noise_pred = self.unet(noisy_latents, timesteps, text_embeddings).sample

            loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
            loss.backward()

            optimizer.step() 
            optimizer.zero_grad() 

            progress_bar.update(1)
            loss_sum += loss.detach().item()
            logs = {"loss": loss_sum / (idx + 1)}
            progress_bar.set_postfix(**logs)

            
            
            if (idx + 1) % 500 == 0:
                image = self(
                    text_embeddings=text_embeddings,
                    generator=generator,
                ).images[0] 
                image.save('./placeholder_tmp_dir/model_tune_' + str(idx+1) + '.png')
                
                self.text_embedding_evaluate(
                    text_embeddings=text_embeddings,
                    init_image_latents=init_image_latents,
                    evaluation_time_steps=evaluation_time_steps 
                )

    


    
    def text_embedding_evaluate(
        self,
        text_embeddings, 
        init_image_latents,
        evaluation_time_steps,
        evaluation_steps = 500,
    ):
        loss_sum = .0 
        with torch.no_grad(): 
            progress_bar = tqdm(range(evaluation_steps))
            for idx in range(evaluation_steps): 
                
                noise = torch.randn(init_image_latents.shape).to(init_image_latents.device) 

                
                
                noisy_latents = self.scheduler.add_noise(init_image_latents, noise, evaluation_time_steps) 

                
                noise_pred = self.unet(noisy_latents, evaluation_time_steps, text_embeddings).sample

                loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean() 

                progress_bar.update(1)
                loss_sum += loss.detach().item()
                logs = {"loss": loss_sum / (idx + 1)}
                progress_bar.set_postfix(**logs) 
        
        print('evaluate text embedding loss: ', loss_sum / evaluation_steps)



    @torch.no_grad()
    def __call__(
        self,
        text_embeddings,
        height: Optional[int] = 256,
        width: Optional[int] = 256,
        num_inference_steps: Optional[int] = 50,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        **kwargs,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
        
        do_classifier_free_guidance = guidance_scale > 1.0
        
        if do_classifier_free_guidance:
            uncond_tokens = [""]
            max_length = self.tokenizer.model_max_length
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

            
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.view(1, seq_len, -1)

            
            
            
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        

        
        
        
        latents_shape = (1, self.unet.in_channels, height // 8, width // 8)
        latents_dtype = text_embeddings.dtype
        if self.device.type == "mps":
            
            latents = torch.randn(latents_shape, generator=generator, device="cpu", dtype=latents_dtype).to(
                self.device
            )
        else:
            latents = torch.randn(latents_shape, generator=generator, device=self.device, dtype=latents_dtype)

        
        self.scheduler.set_timesteps(num_inference_steps)

        
        
        timesteps_tensor = self.scheduler.timesteps.to(self.device)

        
        latents = latents * self.scheduler.init_noise_sigma

        
        
        
        
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for i, t in enumerate(self.progress_bar(timesteps_tensor)):
            
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)

        
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(
                self.device
            )
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(text_embeddings.dtype)
            )
        else:
            has_nsfw_concept = None

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


    @torch.no_grad() 
    def weighted_generate(
        self,
        prompt_emb, 
        prompt_edit_emb,
        prompt_edit_ids,
        prompt_ids=None,  
        prompt_edit_token_weights=[], 
        prompt_edit_tokens_start=0.0, 
        prompt_edit_tokens_end=1.0, 
        prompt_edit_spatial_start=0.0, 
        prompt_edit_spatial_end=1.0, 
        guidance_scale=7.5, 
        steps=50, 
        generator=None, 
        width=256,
        height=256,
        init_image=None, 
    ):
        
        
        
        scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
        scheduler.set_timesteps(steps) 

        init_latent = torch.zeros((1, self.unet.in_channels, height // 8, width // 8), device=self.device) 
        t_start = 0

        
        noise = torch.randn(init_latent.shape, generator=generator, device=self.device)


        if init_image is not None: 
            noise = init_image

        init_latents = noise
        
        latent = scheduler.add_noise(init_latent, noise, torch.tensor([scheduler.timesteps[t_start]], device=self.device)).to(self.device)

        
        with autocast('cuda'):
            tokens_unconditional = self.tokenizer("", padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
            embedding_unconditional = self.text_encoder(tokens_unconditional.input_ids.to(self.device)).last_hidden_state

            
            
            embedding_conditional = prompt_emb 

            
            if prompt_edit_emb is not None:
                tokens_conditional_edit = self.tokenizer(prompt_edit_ids, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
                embedding_conditional_edit = prompt_edit_emb 
                
            
            
                
            timesteps = scheduler.timesteps[t_start:]
            
            for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
                t_index = t_start + i

                
                latent_model_input = latent
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                
                noise_pred_uncond = self.unet(latent_model_input, t, encoder_hidden_states=embedding_unconditional).sample
                
                
                if prompt_edit_emb is not None:
                    save_last_tokens_attention(self.unet)
                    save_last_self_attention(self.unet)
                else:
                    
                    use_last_tokens_attention_weights(self.unet)
                    
                
                noise_pred_cond = self.unet(latent_model_input, t, encoder_hidden_states=embedding_conditional).sample

                
                if prompt_edit_emb is not None:
                    t_scale = t / scheduler.num_train_timesteps
                    if t_scale >= prompt_edit_tokens_start and t_scale <= prompt_edit_tokens_end:
                        use_last_tokens_attention(self.unet)
                    if t_scale >= prompt_edit_spatial_start and t_scale <= prompt_edit_spatial_end:
                        use_last_self_attention(self.unet)
                        
                    
                    use_last_tokens_attention_weights(self.unet)

                    
                    noise_pred_cond = self.unet(latent_model_input, t, encoder_hidden_states=embedding_conditional_edit).sample

                
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                latent = scheduler.step(noise_pred, t_index, latent).prev_sample

            
            latent =  1 / 0.18215 * latent
            image = self.vae.decode(latent.to(self.vae.dtype)).sample


        image = (image / 2 + 0.5).clamp(0, 1)

        
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = self.numpy_to_pil(image)

        return image


    @torch.no_grad() 
    def image_inversion(
            self,
            init_image, 
            prompt='', 
            prompt_emb=None, 
            guidance_scale=3.0, 
            steps=50, 
            generator=None,
            refine_iterations=3, 
            refine_strength=0.9, 
            refine_skip=0.7
        ):
            train_steps = 1000
            step_ratio = train_steps // steps
            timesteps = torch.from_numpy(np.linspace(0, train_steps - 1, steps + 1, dtype=float)).int().to(self.device)
            
            betas = torch.linspace(0.00085**0.5, 0.012**0.5, train_steps, dtype=torch.float32) ** 2
            alphas = torch.cumprod(1 - betas, dim=0)
            
            init_step = 0 
            with autocast('cuda'):
                tokens_unconditional = self.tokenizer("", padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
                embedding_unconditional = self.text_encoder(tokens_unconditional.input_ids.to(self.device)).last_hidden_state

                if prompt_emb is not None: 
                    embedding_conditional = prompt_emb
                else:
                    tokens_conditional = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
                    embedding_conditional = self.text_encoder(tokens_conditional.input_ids.to(self.device)).last_hidden_state
        
                latent = init_image 

                for i in tqdm(range(steps), total=steps):
                    t_index = i + init_step
            
                    t = timesteps[t_index]
                    t1 = timesteps[t_index + 1]
                    
                    tless = t - (t1 - t) * 0.25
                    
                    ap = alphas[t] ** 0.5
                    bp = (1 - alphas[t]) ** 0.5
                    ap1 = alphas[t1] ** 0.5
                    bp1 = (1 - alphas[t1]) ** 0.5
                    
                    latent_model_input = latent
                    
                    noise_pred_uncond = self.unet(latent_model_input, t, encoder_hidden_states=embedding_unconditional).sample
                    
                    
                    noise_pred_cond = self.unet(latent_model_input, t, encoder_hidden_states=embedding_conditional).sample
                    
                    
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    
                    
                    px0 = (latent_model_input - bp * noise_pred) / ap
                    latent = ap1 * px0 + bp1 * noise_pred
                    
                    
                    latent_refine = latent
                    latent_orig = latent_model_input
                    min_error = 1e10
                    lr = refine_strength

                    
                    if i > (steps * refine_skip):
                        for k in range(refine_iterations):
                            
                            
                            noise_pred_uncond = self.unet(latent_refine, tless, encoder_hidden_states=embedding_unconditional).sample
                            noise_pred_cond = self.unet(latent_refine, t, encoder_hidden_states=embedding_conditional).sample
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                            
                            
                            px0 = (latent_refine - bp1 * noise_pred) / ap1
                            latent_refine_orig = ap * px0 + bp * noise_pred
                            
                            
                            error = float((latent_orig - latent_refine_orig).abs_().sum())
                            if error < min_error:
                                latent = latent_refine
                                min_error = error

                            
                            
                            
                            if min_error < 5:
                                break
                            
                            
                            if (min_error - error) < 1:
                                lr *= 0.9

                            
                            latent_refine = latent_refine + (latent_model_input - latent_refine_orig) * lr 

            return latent 


    @torch.no_grad() 
    def prompt_generate(
        self,
        prompt_ids,
        prompt_edit_ids=None, 
        prompt_edit_token_weights=[], 
        prompt_edit_tokens_start=0.0, 
        prompt_edit_tokens_end=1.0, 
        prompt_edit_spatial_start=0.0, 
        prompt_edit_spatial_end=1.0, 
        guidance_scale=7.5, 
        steps=50, 
        generator=None, 
        width=256,
        height=256,
        init_image=None, 
    ):

        
        scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
        scheduler.set_timesteps(steps) 

        init_latent = torch.zeros((1, self.unet.in_channels, height // 8, width // 8), device=self.device) 
        t_start = 0

        
        noise = torch.randn(init_latent.shape, generator=generator, device=self.device)


        if init_image is not None: 
            noise = init_image

        init_latents = noise
        
        latent = scheduler.add_noise(init_latent, noise, torch.tensor([scheduler.timesteps[t_start]], device=self.device)).to(self.device)

        
        with autocast('cuda'):
            tokens_unconditional = self.tokenizer("", padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
            embedding_unconditional = self.text_encoder(tokens_unconditional.input_ids.to(self.device)).last_hidden_state

            tokens_conditional = self.tokenizer(prompt_ids, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
            embedding_conditional = self.text_encoder(tokens_conditional.input_ids.to(self.device)).last_hidden_state

            
            if prompt_edit_ids is not None:
                tokens_conditional_edit = self.tokenizer(prompt_edit_ids, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
                embedding_conditional_edit = self.text_encoder(tokens_conditional_edit.input_ids.to(self.device)).last_hidden_state
                
                
                init_attention_edit(tokens_conditional, tokens_conditional_edit, self.tokenizer, self.unet, self.device)
                
            init_attention_func(unet=self.unet)
            init_attention_weights(prompt_edit_token_weights, self.tokenizer, self.unet, self.device)
                
            timesteps = scheduler.timesteps[t_start:]
            
            for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
                t_index = t_start + i

                
                latent_model_input = latent
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                
                noise_pred_uncond = self.unet(latent_model_input, t, encoder_hidden_states=embedding_unconditional).sample
                
                
                if prompt_edit_ids is not None:
                    save_last_tokens_attention(self.unet)
                    save_last_self_attention(self.unet)
                else:
                    
                    use_last_tokens_attention_weights(self.unet)
                    
                
                noise_pred_cond = self.unet(latent_model_input, t, encoder_hidden_states=embedding_conditional).sample

                
                if prompt_edit_ids is not None:
                    t_scale = t / scheduler.num_train_timesteps
                    if t_scale >= prompt_edit_tokens_start and t_scale <= prompt_edit_tokens_end:
                        use_last_tokens_attention(self.unet)
                    if t_scale >= prompt_edit_spatial_start and t_scale <= prompt_edit_spatial_end:
                        use_last_self_attention(self.unet)
                        
                    
                    use_last_tokens_attention_weights(self.unet)

                    
                    noise_pred_cond = self.unet(latent_model_input, t, encoder_hidden_states=embedding_conditional_edit).sample

                
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                latent = scheduler.step(noise_pred, t_index, latent).prev_sample

            
            latent =  1 / 0.18215 * latent
            image = self.vae.decode(latent.to(self.vae.dtype)).sample


        image = (image / 2 + 0.5).clamp(0, 1)

        
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = self.numpy_to_pil(image)

        return image

