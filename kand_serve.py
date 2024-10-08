





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

from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline, EulerAncestralDiscreteScheduler
import torch

pipe_prior = KandinskyV22PriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-prior",
                                                       torch_dtype=torch.float16)
pipe_prior.to("cuda")


pipe = KandinskyV22Pipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder",
                                            torch_dtype=torch.float16)
pipe.to("cuda")
# pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
#pipe.unet = torch.compile(pipe.unet)


import open_clip
model, _, preprocess = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k')
#model = torch.compile(model)






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
            print('######### Calibrating with sample prompts #########')
            prompt = calibrate_prompts.pop(0)
            image_emb, negative_image_emb = pipe_prior(prompt).to_tuple()

            image = pipe(
                image_embeds=image_emb,
                negative_image_embeds=negative_image_emb,
                height=512,
                width=512,
                num_inference_steps=50,
            ).images


            ####### optional step; we could take the prior output instead
            tensor_image = preprocess(image[0]).unsqueeze(0)
            with torch.cuda.amp.autocast():
                image_features = model.encode_image(tensor_image)
            image_emb = image_features
            #######

            embs.append(image_emb)
            return image[0]
        else:
            print('######### Roaming #########')

            # sample only as many negatives as there are positives
            
            indices = range(len(ys))                
            pos_indices = [i for i in indices if ys[i] > .5]
            neg_indices = [i for i in indices if ys[i] <= .5]
            
            if False:#len(pos_indices) > 15:
                to_drop = pos_indices.pop(0)
                ys.pop(to_drop)
                embs.pop(to_drop)
                print('\n\n\ndropping\n\n\n')
            elif False:#len(neg_indices) > 15:
                to_drop = neg_indices.pop(0)
                ys.pop(to_drop)
                embs.pop(to_drop)
                print('\n\n\ndropping\n\n\n')
            
            

            feature_embs = torch.stack([e[0].detach().cpu() for e in embs])
            scaler = preprocessing.StandardScaler().fit(feature_embs)
            feature_embs = scaler.transform(feature_embs)
            print(np.array(feature_embs).shape, np.array(ys).shape)
            #if feature_embs.norm() != 0:
            #    feature_embs = feature_embs / feature_embs.norm()
            

            sol = LogisticRegression().fit(np.array(feature_embs), np.array(torch.tensor(ys).unsqueeze(1).float())).coef_
            #sol = torch.linalg.lstsq(torch.tensor(ys).unsqueeze(1).float() / 2 + .5, torch.tensor(feature_embs).float()).solution
            sol = torch.tensor(sol, dtype=torch.double).to('cuda')

            if global_idx % 2 == 0:
                w = 50
                rng_prompt = random.choice(prompt_list)
                image_emb, negative_image_emb = pipe_prior(rng_prompt).to_tuple()
                print('\n\n**********',rng_prompt,'\n**********')
            else:
                w = 70
                image_emb, negative_image_emb = pipe_prior('an image').to_tuple()
                image_emb = negative_image_emb
            print(sol.abs().max(), image_emb.abs().max())
            
            image_emb = image_emb + w * sol
            

            image = pipe(
                image_embeds=image_emb,
                negative_image_embeds=negative_image_emb,
                height=512,
                width=512,
                num_inference_steps=50,
            ).images

            ####### optional step; we could take the prior output instead
            tensor_image = preprocess(image[0]).unsqueeze(0)
            with torch.cuda.amp.autocast():
                image_features = model.encode_image(tensor_image)
            image_emb = image_features
            #######

            print(image_emb.shape)
            embs.append(image_emb)

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

