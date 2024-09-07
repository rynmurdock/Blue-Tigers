





# TODO may want to balance input negs & poss because of no clamping

# TODO add a prompt bar?




import pandas as pd
import gradio as gr
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import LinearSVC, LinearSVR
from sklearn import preprocessing
from matplotlib import pyplot as plt

from tqdm import tqdm
from PIL import Image
import random
import time

import kornia
import torch
import torchvision

from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from patch_sdxl import SDEmb

device = 'cuda'
dtype = torch.float16

base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_8step_unet.safetensors" # Use the correct ckpt for your step setting!

# Load model.
unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(device, dtype)
unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))
pipe = SDEmb.from_pretrained(base, unet=unet, torch_dtype=dtype, variant="fp16").to(device)

# Ensure sampler uses "trailing" timesteps.
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")


pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
output_hidden_state = False
pipe.set_ip_adapter_scale(1)


#import open_clip
#model, _, preprocess = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k')
#model = torch.compile(model)


# transform = kornia.augmentation.RandomResizedCrop(size=(224, 224), scale=(.3, .5))
nom = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
def patch_encode_image(image):
    image = torch.tensor(torchvision.transforms.functional.pil_to_tensor(image).to(torch.float16)).to('cuda')[None, ...]
    image = torch.nn.functional.interpolate(image, (224, 224)) / 255
    patches = nom(image)
    output, _ = pipe.encode_image(
                patches, 'cuda', 1, output_hidden_state
            )
    return output




prompt_list = [p for p in list(set(
                pd.read_csv('/home/ryn_mote/Misc/twitter_prompts.csv').iloc[:, 1].tolist())) if type(p) == str]
random.shuffle(prompt_list)



NOT_calibrate_prompts = [
    "4k photo",
    'surrealist art photography',
    'a psychedelic, fractal view',
    'a beautiful collage',
    'abstract art',
    'an eldritch image',
    #'a sketch',
    ]

calibrate_prompts = [
    "4k photo",
    'surrealist art',
    'a psychedelic, fractal view',
    'a beautiful collage',
    'an intricate portrait',
    'an impressionist painting',
    'abstract art',
    'an eldritch image',
    'a sketch',
    'a city full of darkness and graffiti',
    'a black & white photo',
    'a brilliant, timeless tarot card of the world',
    '''eternity: a timeless, vivid painted portrait by ryan murdock''',
    '''a simple, timeless, & dark charcoal on canvas: death itself by ryan murdock''',
    '''a painted image with gorgeous red gradients: Persephone by ryan murdock''',
    '''a simple, timeless, & dark photo with gorgeous gradients: last night of my life by ryan murdock''',
    '''the sunflower -- a dark, simple painted still life by ryan murdock''',
    '''silence in the macrocosm -- a dark, intricate painting by ryan murdock''',
    '''beauty here -- a photograph by ryan murdock''',
    '''a timeless, haunting portrait: the necrotic jester''',
    '''a simple, timeless, & dark art piece with gorgeous gradients: serenity''',
    '''an elegant image of nature with gorgeous swirling gradients''',
    '''simple, timeless digital art with gorgeous purple spirals''',
    '''timeless digital art with gorgeous gradients: eternal slumber''',
    '''a simple, timeless image with gorgeous gradients''',
    '''a simple, timeless painted image of nature with beautiful gradients''',
    'a timeless, dark digital art piece with gorgeous gradients: the hanged man',
    '',
]

global_idx = 0
embs = []
ys = []

start_time = time.time()

def next_image():
    with torch.no_grad():
        if len(calibrate_prompts) > 0:
            prompt = calibrate_prompts.pop(0)
            print(f'######### Calibrating with sample: {prompt} #########')

            image = pipe(
                prompt=prompt,
                ip_adapter_emb=torch.zeros(1, 1280, device='cuda', dtype=torch.float16),
                height=1024,
                width=1024,
                num_inference_steps=8,
                guidance_scale=0,
            ).images


            ####### optional step; we could take the prior output instead
            with torch.cuda.amp.autocast():
                pooled_embeds = patch_encode_image(
                image[0]
                )
            #######

            embs.append(pooled_embeds)
            return image[0]
        else:
            print('######### Roaming #########')

            # sample only as many negatives as there are positives
            indices = range(len(ys))                
            pos_indices = [i for i in indices if ys[i] > .5]
            neg_indices = [i for i in indices if ys[i] <= .5]
            
            mini = min(len(pos_indices), len(neg_indices))
            
            if mini < 1:
                feature_embs = torch.stack([torch.randn(1280), torch.randn(1280)])
                ys_t = [0, 1]
                print('Not enough ratings.')
            else:
                # indices = random.sample(pos_indices, mini) + random.sample(neg_indices, mini)
                ys_t = [ys[i] for i in indices]
                feature_embs = torch.stack([embs[e].detach().cpu() for e in indices]).squeeze()

                # # balance pos/negatives?
                # for e in indices:
                #     nw = (len(indices) / len(neg_indices))
                #     w = (len(indices) / len(pos_indices))
                #     feature_embs[e] = feature_embs[e] * w if ys_t[e] > .5 else feature_embs[e] * nw
                
                # if len(pos_indices) > 8:
                #    to_drop = pos_indices.pop(0)
                #    ys.pop(to_drop)
                #    embs.pop(to_drop)
                #    print('\n\n\ndropping\n\n\n')
                # elif len(neg_indices) > 8:
                #    to_drop = neg_indices.pop(0)
                #    ys.pop(to_drop)
                #    embs.pop(to_drop)
                #    print('\n\n\ndropping\n\n\n')
                
                
                # scaler = preprocessing.StandardScaler().fit(feature_embs)
                # feature_embs = scaler.transform(feature_embs)
                # ys_t = ys
                
                print(np.array(feature_embs).shape, np.array(ys_t).shape)
            
            # sol = LogisticRegression().fit(np.array(feature_embs), np.array(torch.tensor(ys_t).unsqueeze(1).float() * 2 - 1)).coef_
            # sol = torch.linalg.lstsq(torch.tensor(ys_t).unsqueeze(1).float()*2-1, torch.tensor(feature_embs).float(),).solution
            # neg_sol = torch.linalg.lstsq((torch.tensor(ys_t).unsqueeze(1).float() - 1) * -1, torch.tensor(feature_embs).float()).solution
            # sol = torch.tensor(sol, dtype=dtype).to(device)


            pos_sol = torch.stack([feature_embs[i] for i in range(len(ys_t)) if ys_t[i] > .5]).mean(0, keepdim=True).to(device, dtype)
            neg_sol = torch.stack([feature_embs[i] for i in range(len(ys_t)) if ys_t[i] < .5]).mean(0, keepdim=True).to(device, dtype)
            
            # could j have a base vector of a black image
            latest_pos = (random.sample([feature_embs[i] for i in range(len(ys_t)) if ys_t[i] > .5], 1)[0]).to(device, dtype)

            dif = pos_sol - neg_sol
            dif = ((dif / dif.std()) * latest_pos.std())

            sol = latest_pos + dif

            if global_idx % 2 == 0:
                w = 32
                prompt = random.choice(prompt_list)
                pipe.set_ip_adapter_scale(.7)
            else:
                w = 32
                prompt = 'an image'
                pipe.set_ip_adapter_scale(1.)
            
            image_emb = (w * sol)
            

            image = pipe(
                prompt=prompt,
                ip_adapter_emb=image_emb,
                height=1024,
                width=1024,
                num_inference_steps=8,
                guidance_scale=1,
            ).images

            ####### optional step; we could take the prior output instead
            with torch.cuda.amp.autocast():
                pooled_embeds = patch_encode_image(
                image[0]
                )
            #######
            print(pooled_embeds.max(), image_emb.max())
            embs.append(pooled_embeds)
            
            print('\n\n**********',prompt,'\n**********')

            torch.save(sol, f'./{start_time}.pt')
            return image[0]
            






def start(_):
    return [
            gr.Button(value='Like', interactive=True), 
            gr.Button(value='Neither', interactive=True), 
            gr.Button(value='Dislike', interactive=True),
            gr.Button(value='Start', interactive=False),
            next_image()
            ]


def choose(choice):
    global global_idx
    global_idx += 1
    if choice == 'Like':
        choice = 1
    elif choice == 'Neither':
        _ = embs.pop(-1)
        return next_image()
    else:
        choice = 0
    ys.append(choice)
    return next_image()

css = "div#output-image {height: 512px !important; width: 512px !important; margin:auto;}"
with gr.Blocks(css=css) as demo:
    with gr.Row():
        html = gr.HTML('''<div style='text-align:center; font-size:32'>You will callibrate for several prompts and then roam.</ div>''')
    with gr.Row(elem_id='output-image'):
        img = gr.Image(interactive=False, elem_id='output-image',)
    with gr.Row(equal_height=True):
        b3 = gr.Button(value='Dislike', interactive=False,)
        b2 = gr.Button(value='Neither', interactive=False,)
        b1 = gr.Button(value='Like', interactive=False,)
        b1.click(
        choose, 
        [b1],
        [img]
        )
        b2.click(
        choose, 
        [b2],
        [img]
        )
        b3.click(
        choose, 
        [b3],
        [img]
        )
    with gr.Row():
        b4 = gr.Button(value='Start')
        b4.click(start,
                 [b4],
                 [b1, b2, b3, b4, img,])

demo.launch(server_name="0.0.0.0")  # Share your demo with just 1 extra parameter 🚀

