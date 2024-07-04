

# TODO unify/merge origin and this
# TODO save & restart from (if it exists) dataframe parquet
import torch

# lol
DEVICE = 'cuda'
STEPS = 6
output_hidden_state = False
device = "cuda"
dtype = torch.bfloat16

# gemma
EMB_LEN = 3072



import spaces

import matplotlib.pyplot as plt
import matplotlib
import logging


import os
import imageio
import gradio as gr
import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
import sched
import threading

import random
import time
from PIL import Image
from safety_checker_improved import maybe_nsfw


torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

prevs_df = pd.DataFrame(columns=['paths', 'embeddings', 'ips', 'user:rating', 'latest_user_to_rate', 'from_user_id', 'text', 'gemb'])

import spaces
start_time = time.time()

####################### Setup Model
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler, LCMScheduler, AutoencoderTiny, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from PIL import Image
from transformers import CLIPVisionModelWithProjection
import uuid
import av

def write_video(file_name, images, fps=16):
    container = av.open(file_name, mode="w")

    stream = container.add_stream("h264", rate=fps)
    # stream.options = {'preset': 'faster'}
    stream.thread_count = 1
    stream.width = 512
    stream.height = 512
    stream.pix_fmt = "yuv420p"

    for img in images:
        img = np.array(img)
        img = np.round(img).astype(np.uint8)
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    # Flush stream
    for packet in stream.encode():
        container.mux(packet)
    # Close the file
    container.close()

def imio_write_video(file_name, images, fps=15):
    writer = imageio.get_writer(file_name, fps=fps)

    for im in images:
        writer.append_data(np.array(im))
    writer.close()


image_encoder = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder="sdxl_models/image_encoder", torch_dtype=dtype, 
device_map='cuda')
#vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=dtype)

# vae = ConsistencyDecoderVAE.from_pretrained("openai/consistency-decoder", torch_dtype=dtype)
# vae = compile_unet(vae, config=config)

#finetune_path = '''/home/ryn_mote/Misc/finetune-sd1.5/dreambooth-model best'''''
#unet = UNet2DConditionModel.from_pretrained(finetune_path+'/unet/').to(dtype)
#text_encoder = CLIPTextModel.from_pretrained(finetune_path+'/text_encoder/').to(dtype)

#rynmurdock/Sea_Claws
unet = UNet2DConditionModel.from_pretrained('rynmurdock/Sea_Claws', subfolder='unet',).to(dtype).to('cpu')
text_encoder = CLIPTextModel.from_pretrained('rynmurdock/Sea_Claws', subfolder='text_encoder', 
device_map='cpu').to(dtype)

adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM")
pipe = AnimateDiffPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", motion_adapter=adapter, image_encoder=image_encoder, 
                                            torch_dtype=dtype,     
                                            unet=unet, text_encoder=text_encoder,)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")
pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora",)
pipe.set_adapters(["lcm-lora"], [.9])
pipe.fuse_lora()
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()


#pipe = AnimateDiffPipeline.from_pretrained('emilianJR/epiCRealism', torch_dtype=dtype, image_encoder=image_encoder)
#pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")
#repo = "ByteDance/AnimateDiff-Lightning"
#ckpt = f"animatediff_lightning_4step_diffusers.safetensors"


pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15_vit-G.bin", map_location='cpu')
# This IP adapter improves outputs substantially.
target_blocks = {"up": {'block_1': 1.}, "mid": 1.}
pipe.set_ip_adapter_scale(target_blocks)



pipe.unet.fuse_qkv_projections()
#pipe.enable_free_init(method="gaussian", use_fast_sampling=True)

pipe.to(device=DEVICE)
#pipe.unet = torch.compile(pipe.unet)
#pipe.vae = torch.compile(pipe.vae)


from types import MethodType

from transformers import AutoTokenizer

import gemma_portion

tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it",)
gem_model = gemma_portion.GemmaForCausalLM.from_pretrained("google/gemma-7b-it", 
                                                                device_map="auto", 
                                                                torch_dtype=torch.bfloat16, load_in_4bit=True)
gem_model._sample = MethodType(gemma_portion._sample, gem_model)
gem_model.generate = MethodType(gemma_portion.generate, gem_model)

# SEE GEMMA_PORTION FOR PLUCK_LAYER

@spaces.GPU()
def generate_gemm(prompt='describe a compelling scene:', in_embs=torch.zeros(1, 1, EMB_LEN),):
  prompt = tokenizer(prompt, return_tensors="pt").to("cuda").input_ids
  in_embs = in_embs / in_embs.abs().max() * 2
  text, in_embs = gem_model.generate(prompt, probe_direction=in_embs.squeeze()[None, None, :].to(device='cuda', dtype=dtype), do_sample=True, top_p=.7, max_new_tokens=20)
  text = tokenizer.decode(text[0], skip_special_tokens=True)
  print('\n\n\n', text, '\n\n\n')
  return text, torch.cat(in_embs[1:], 1).mean(1).to('cpu').to(torch.float32)

@spaces.GPU()
def cal_generate(prompt, in_embs=torch.zeros(1, 1, EMB_LEN),):
  prompt = tokenizer(prompt, return_tensors="pt").to("cuda").input_ids
  text, in_embs = gem_model.generate(prompt, probe_direction=in_embs.squeeze()[None, None, :].to(device='cuda', dtype=dtype), max_new_tokens=2)
  text = tokenizer.decode(text[0][len(prompt):], skip_special_tokens=True)
  return text, torch.cat(in_embs[1:], 1).mean(1).to('cpu').to(torch.float32)


@spaces.GPU()
def generate_gpu(in_im_embs, prompt='the scene'):
    with torch.no_grad():
        print(prompt)
        in_im_embs = in_im_embs.to('cuda').unsqueeze(0).unsqueeze(0)
        in_im_embs = in_im_embs / in_im_embs.abs().max() * 3
        output = pipe(prompt=prompt, guidance_scale=1, added_cond_kwargs={}, ip_adapter_image_embeds=[in_im_embs], num_inference_steps=STEPS)
        im_emb, _ = pipe.encode_image(
                    output.frames[0][len(output.frames[0])//2], 'cuda', 1, output_hidden_state
                )
        im_emb = im_emb.detach().to('cpu').to(torch.float32)
    return output, im_emb


def generate(in_im_embs, prompt='the scene'):
    output, im_emb = generate_gpu(in_im_embs, prompt)
    nsfw = maybe_nsfw(output.frames[0][len(output.frames[0])//2])
    
    name = str(uuid.uuid4()).replace("-", "")
    path = f"/tmp/{name}.mp4"
    
    if nsfw:
        gr.Warning("NSFW content detected.")
        # TODO could return an automatic dislike of auto dislike on the backend for neither as well; just would need refactoring.
        return None, im_emb
    
    
    output.frames[0] = output.frames[0] + list(reversed(output.frames[0]))

    write_video(path, output.frames[0])
    return path, im_emb


#######################






@spaces.GPU()
def solver(embs, ys):
    print('ys:', ys,'EMBS:', embs.shape, embs)
    ys = torch.tensor(ys).to('cpu', dtype=torch.float32).squeeze().unsqueeze(1)
    
#     if ys.abs().max() != 0:
#         ys = ys / ys.abs().max()
    ys = (ys - .5) * 2

    #if embs.norm() != 0:
    #    embs = embs / embs.norm()
    

    sol = torch.linalg.lstsq(ys, embs).solution
    return sol.to('cpu', dtype=torch.float32)




def get_user_emb(embs, ys):
    # handle case where every instance of calibration videos is 'Neither' or 'Like' or 'Dislike'

    indices = random.sample(list(range(len(embs))), len(embs)//2) # sample half
    # sample only as many negatives as there are positives
    pos_indices = [i for i in indices if ys[i] == 1]
    neg_indices = [i for i in indices if ys[i] == 0]

    if len(neg_indices) <= 1 or len(pos_indices) <= 1:
        embs = [i for i in embs]
        aways = [torch.rand_like(embs[0]) - .5 for i in range(4)]
        embs += aways
        awal = [0 for i in range(2)] + [1 for i in range(2)]
        ys += awal

        indices = random.sample(list(range(len(embs))), len(embs)//2) # sample half
        # sample only as many negatives as there are positives
        pos_indices = [i for i in indices if ys[i] == 1]
        neg_indices = [i for i in indices if ys[i] == 0]


    lower = min(len(pos_indices), len(neg_indices))
    neg_indices = random.sample(neg_indices, lower)
    pos_indices = random.sample(pos_indices, lower)
    indices = pos_indices + neg_indices
    
    
    # we may have just encountered a rare multi-threading diffusers issue (https://github.com/huggingface/diffusers/issues/5749);
    # this ends up adding a rating but losing an embedding, it seems.
    # let's take off a rating if so to continue without indexing errors.
    if len(ys) > len(embs):
        print('ys are longer than embs; popping latest rating')
        ys.pop(-1)
    
    feature_embs = torch.stack([embs[i].squeeze().to('cpu') for i in indices]).to('cpu')
    #scaler = preprocessing.StandardScaler().fit(feature_embs)
    #feature_embs = scaler.transform(feature_embs)
    chosen_y = np.array([ys[i] for i in indices])
    
    #if torch.all(feature_embs.norm(-1) != torch.zeros_like(feature_embs.norm(-1))):
    #    feature_embs = feature_embs / feature_embs.norm(-1, keepdim=True) * np.sqrt(feature_embs.shape[-1])
    
    #lin_class = Ridge(fit_intercept=False).fit(feature_embs, chosen_y)
    #lin_class = SVC(max_iter=20, kernel='linear', C=.1, class_weight='balanced').fit(feature_embs, chosen_y)
    #coef_ = torch.tensor(lin_class.coef_, dtype=torch.float32).detach().to('cpu')
    coef_ = solver(feature_embs, chosen_y)

    w = 1# if len(embs) % 2 == 0 else 0
    im_emb = w * coef_.to(dtype=dtype)
    return im_emb


def pluck_img(user_id, user_emb):
    not_rated_rows = prevs_df[[i[1]['user:rating'].get(user_id, 'gone') == 'gone' for i in prevs_df.iterrows()]]
    while len(not_rated_rows) == 0:
        not_rated_rows = prevs_df[[i[1]['user:rating'].get(user_id, 'gone') == 'gone' for i in prevs_df.iterrows()]]
        time.sleep(.1)
    # TODO optimize this lol
    best_sim = -100000
    for i in not_rated_rows.iterrows():
        # TODO sloppy .to but it is 3am.
        sim = torch.cosine_similarity(i[1]['embeddings'].detach().to('cpu'), user_emb.detach().to('cpu'))
        if sim > best_sim:
            best_sim = sim
            best_row = i[1]
    img = best_row['paths']
    text = best_row.get('text', '')
    return img, text


def background_next_image():
        global prevs_df
        # only let it get N (maybe 3) ahead of the user
        #not_rated_rows = prevs_df[[i[1]['user:rating'] == {' ': ' '} for i in prevs_df.iterrows()]]
        rated_rows = prevs_df[[i[1]['user:rating'] != {' ': ' '} for i in prevs_df.iterrows()]]
        if len(rated_rows) < 4:
            time.sleep(.1)
        #    not_rated_rows = prevs_df[[i[1]['user:rating'] == {' ': ' '} for i in prevs_df.iterrows()]]
            return

        user_id_list = set(rated_rows['latest_user_to_rate'].to_list())
        for uid in user_id_list:
            rated_rows = prevs_df[[i[1]['user:rating'].get(uid, None) is not None for i in prevs_df.iterrows()]]
            not_rated_rows = prevs_df[[i[1]['user:rating'].get(uid, None) is None for i in prevs_df.iterrows()]]
            
            # we need to intersect not_rated_rows from this user's embed > 7. Just add a new column on which user_id spawned the 
            #   media. 
            
            unrated_from_user = not_rated_rows[[i[1]['from_user_id'] == uid for i in not_rated_rows.iterrows()]]
            rated_from_user = rated_rows[[i[1]['from_user_id'] == uid for i in rated_rows.iterrows()]]

            # we pop previous ratings if there are > n
            if len(rated_from_user) >= 15:
                oldest = rated_from_user.iloc[0]['paths']
                prevs_df = prevs_df[prevs_df['paths'] != oldest]
            # we don't compute more after n are in the queue for them
            if len(unrated_from_user) >= 10:
                continue
            
            if len(rated_rows) < 5:
                continue
            
            embs, ys, gembs = pluck_embs_ys(uid)
            
            user_emb = get_user_emb(embs, [y[1] for y in ys])
            
            if len(gembs) > 4:
                new_gem = get_user_emb(gembs, [y[0] for y in ys])
                text, _ = generate_gemm(in_embs=new_gem)
                _, gembs = cal_generate(prompt=text)
            else:
                text, gembs = generate_gemm(in_embs=torch.zeros(1, EMB_LEN))
            img, embs = generate(user_emb, text)
            
            if img:
                tmp_df = pd.DataFrame(columns=['paths', 'embeddings', 'ips', 'user:rating', 'latest_user_to_rate', 'text', 'gemb'])
                tmp_df['paths'] = [img]
                tmp_df['embeddings'] = [embs]
                tmp_df['user:rating'] = [{' ': ' '}]
                tmp_df['from_user_id'] = [uid]
                tmp_df['text'] = [text]
                tmp_df['gemb'] = [gembs]
                prevs_df = pd.concat((prevs_df, tmp_df))
                # we can free up storage by deleting the image
                if len(prevs_df) > 500:
                    oldest_path = prevs_df.iloc[6]['paths']
                    if os.path.isfile(oldest_path):
                        os.remove(oldest_path)
                    else:
                        # If it fails, inform the user.
                        print("Error: %s file not found" % oldest_path)
                    # only keep 50 images & embeddings & ips, then remove oldest besides calibrating
                    prevs_df = pd.concat((prevs_df.iloc[:6], prevs_df.iloc[7:]))
    

def pluck_embs_ys(user_id):
    rated_rows = prevs_df[[i[1]['user:rating'].get(user_id, None) != None for i in prevs_df.iterrows()]]
    #not_rated_rows = prevs_df[[i[1]['user:rating'].get(user_id, None) == None for i in prevs_df.iterrows()]]
    #while len(not_rated_rows) == 0:
    #    not_rated_rows = prevs_df[[i[1]['user:rating'].get(user_id, None) == None for i in prevs_df.iterrows()]]
    #    rated_rows = prevs_df[[i[1]['user:rating'].get(user_id, None) != None for i in prevs_df.iterrows()]]
    #    time.sleep(.01)
    #    print('current user has 0 not_rated_rows')
    
    embs = rated_rows['embeddings'].to_list()
    ys = [i[user_id] for i in rated_rows['user:rating'].to_list()]
    gembs = rated_rows['gemb'].to_list()
    return embs, ys, gembs

def next_image(calibrate_prompts, user_id):
    with torch.no_grad():
        if len(calibrate_prompts) > 0:
            cal_video = calibrate_prompts.pop(0)
            image = prevs_df[prevs_df['paths'] == cal_video]['paths'].to_list()[0]
            text = prevs_df[prevs_df['paths'] == cal_video]['text'].to_list()[0]
            return image, calibrate_prompts, text
        else:
            embs, ys, gembs = pluck_embs_ys(user_id)
            ys_here = [y[1] for y in ys]
            user_emb = get_user_emb(embs, ys_here)
            image, text = pluck_img(user_id, user_emb)
            return image, calibrate_prompts, text









def start(_, calibrate_prompts, user_id, request: gr.Request):
    user_id = int(str(time.time())[-7:].replace('.', ''))
    image, calibrate_prompts, text = next_image(calibrate_prompts, user_id)
    return [
            gr.Button(value='👍', interactive=True), 
            gr.Button(value='Neither (Space)', interactive=True, visible=False), 
            gr.Button(value='👎', interactive=True),
            gr.Button(value='Start', interactive=False),
            gr.Button(value='👍 Content', interactive=True, visible=False),
            gr.Button(value='👍 Style', interactive=True, visible=False),
            image,
            calibrate_prompts,
            user_id,
            text
            ]


def choose(img, choice, calibrate_prompts, user_id, request: gr.Request):
    global prevs_df
    
    
    if choice == '👍':
        choice = [1, 1]
    elif choice == 'Neither (Space)':
        img, calibrate_prompts, text = next_image(calibrate_prompts, user_id)
        return img, calibrate_prompts, text
    elif choice == '👎':
        choice = [0, 0]
    elif choice == '👍 Style':
        choice = [0, 1]
    elif choice == '👍 Content':
        choice = [1, 0]
    else:
        assert False, f'choice is {choice}'
    
    # if we detected NSFW, leave that area of latent space regardless of how they rated chosen.
    # TODO skip allowing rating & just continue
    if img == None:
        print('NSFW -- choice is disliked')
        choice = [0, 0]
    
    row_mask = [p.split('/')[-1] in img for p in prevs_df['paths'].to_list()]
    # if it's still in the dataframe, add the choice
    if len(prevs_df.loc[row_mask, 'user:rating']) > 0:
        prevs_df.loc[row_mask, 'user:rating'][0][user_id] = choice
        prevs_df.loc[row_mask, 'latest_user_to_rate'] = [user_id]
    img, calibrate_prompts, text = next_image(calibrate_prompts, user_id)
    return img, calibrate_prompts, text

css = '''.gradio-container{max-width: 700px !important}
#description{text-align: center}
#description h1, #description h3{display: block}
#description p{margin-top: 0}
.fade-in-out {animation: fadeInOut 3s forwards}
@keyframes fadeInOut {
    0% {
      background: var(--bg-color);
    }
    100% {
      background: var(--button-secondary-background-fill);
    }
}
'''
js_head = '''
<script>
document.addEventListener('keydown', function(event) {
    if (event.key === 'a' || event.key === 'A') {
        // Trigger click on 'dislike' if 'A' is pressed
        document.getElementById('dislike').click();
    } else if (event.key === ' ' || event.keyCode === 32) {
        // Trigger click on 'neither' if Spacebar is pressed
        document.getElementById('neither').click();
    } else if (event.key === 'l' || event.key === 'L') {
        // Trigger click on 'like' if 'L' is pressed
        document.getElementById('like').click();
    }
});
function fadeInOut(button, color) {
  button.style.setProperty('--bg-color', color);
  button.classList.remove('fade-in-out');
  void button.offsetWidth; // This line forces a repaint by accessing a DOM property
  
  button.classList.add('fade-in-out');
  button.addEventListener('animationend', () => {
    button.classList.remove('fade-in-out'); // Reset the animation state
  }, {once: true});
}
document.body.addEventListener('click', function(event) {
    const target = event.target;
    if (target.id === 'dislike') {
      fadeInOut(target, '#ff1717');
    } else if (target.id === 'like') {
      fadeInOut(target, '#006500');
    } else if (target.id === 'neither') {
      fadeInOut(target, '#cccccc');
    }
});

</script>
'''

with gr.Blocks(css=css, head=js_head) as demo:
    gr.Markdown('''# Blue Tigers
### Generative Recommenders for Exporation of Video

Explore the latent space without text prompts based on your preferences. Learn more on [the write-up](https://rynmurdock.github.io/posts/2024/3/generative_recomenders/).
    ''', elem_id="description")
    user_id = gr.State()
    # calibration videos -- this is a misnomer now :D
    calibrate_prompts = gr.State([
    './first.mp4',
    './second.mp4',
    './third.mp4',
    './fourth.mp4',
    './fifth.mp4',
    './sixth.mp4',
    ])
    def l():
        return None

    with gr.Row(elem_id='output-image'):
        img = gr.Video(
        label='Lightning',
        autoplay=True,
        interactive=False,
        height=512,
        width=512,
        #include_audio=False,
        elem_id="video_output"
       )
        img.play(l, js='''document.querySelector('[data-testid="Lightning-player"]').loop = true''')
    
    
    
    with gr.Row(equal_height=True):
        b3 = gr.Button(value='👎', interactive=False, elem_id="dislike")

        b2 = gr.Button(value='Neither (Space)', interactive=False, elem_id="neither", visible=False)

        b1 = gr.Button(value='👍', interactive=False, elem_id="like")
    with gr.Row(equal_height=True):
        b6 = gr.Button(value='👍 Style', interactive=False, elem_id="dislike like", visible=False)
        
        b5 = gr.Button(value='👍 Content', interactive=False, elem_id="like dislike", visible=False)
    with gr.Row():
        text = gr.Textbox(interactive=False, visible=True)
        
        
        b1.click(
        choose, 
        [img, b1, calibrate_prompts, user_id],
        [img, calibrate_prompts, text],
        )
        b2.click(
        choose, 
        [img, b2, calibrate_prompts, user_id],
        [img, calibrate_prompts, text],
        )
        b3.click(
        choose, 
        [img, b3, calibrate_prompts, user_id],
        [img, calibrate_prompts, text],
        )
        b5.click(
        choose, 
        [img, b5, calibrate_prompts, user_id],
        [img, calibrate_prompts, text],
        )
        b6.click(
        choose, 
        [img, b6, calibrate_prompts, user_id],
        [img, calibrate_prompts, text],
        )
    with gr.Row():
        b4 = gr.Button(value='Start')
        b4.click(start,
                 [b4, calibrate_prompts, user_id],
                 [b1, b2, b3, b4, b5, b6, img, calibrate_prompts, user_id, text]
                 )
    with gr.Row():
        html = gr.HTML('''<div style='text-align:center; font-size:20px'>You will calibrate for several videos and then roam. </ div><br><br><br>
<div style='text-align:center; font-size:14px'>Note that while the AnimateLCM model with NSFW filtering is unlikely to produce NSFW images, this may still occur, and users should avoid NSFW content when rating.
</ div>
<br><br>
<div style='text-align:center; font-size:14px'>Thanks to @multimodalart for their contributions to the demo, esp. the interface and @maxbittker for feedback.
</ div>''')

# TODO quiet logging

scheduler = BackgroundScheduler()
scheduler.add_job(func=background_next_image, trigger="interval", seconds=.2)
scheduler.start()

#thread = threading.Thread(target=background_next_image,)
#thread.start()

# TODO shouldn't call this before gradio launch, yeah?
@spaces.GPU()
def encode_space(x):
    im_emb, _ = pipe.encode_image(
                image, DEVICE, 1, output_hidden_state
            )
    return im_emb.detach().to('cpu').to(torch.float32)

# prep our calibration videos
for im, txt in [ # TODO more movement
    ('./first.mp4', 'describe the scene: a painted still life in red & bright colors'),
    ('./second.mp4', 'describe the scene: surrealist landscape with solid red circle'),
    ('./third.mp4', 'describe the scene: geometric abstract art'),
    ('./fourth.mp4', 'describe the scene: a painted landscape on fire'),
    ('./fifth.mp4', 'describe the scene: a bizarre plant growing from a glass vase'),
    ('./sixth.mp4', 'describe the scene: dark fractals'),
    ('./seventh.mp4', 'describe the scene: a dark object'),
    ('./eigth.mp4', 'describe the scene: an image: an object; small tree made of green & black wood'),
    ('./ninth.mp4', 'describe the scene: an image of gorgeous flowing art; a sunflower'),
    ('./tenth.mp4', 'describe the scene: a fluffy creature turns towards you'),
    ]:
    tmp_df = pd.DataFrame(columns=['paths', 'embeddings', 'ips', 'user:rating', 'text', 'gemb'])
    tmp_df['paths'] = [im]
    image = list(imageio.imiter(im))
    image = image[len(image)//2]
    im_emb = encode_space(image)
    
    _, gemb = cal_generate(prompt=txt)

    tmp_df['embeddings'] = [im_emb.detach().to('cpu')]
    tmp_df['user:rating'] = [{' ': ' '}]
    tmp_df['gemb'] = [gemb.detach().to('cpu')]
    tmp_df['text'] = [txt]
    prevs_df = pd.concat((prevs_df, tmp_df))


demo.launch(share=True, server_port=8443)


