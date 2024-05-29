








# TODO unify/merge origin and this
# TODO save & restart from (if it exists) dataframe parquet
import torch

torch.set_grad_enabled(False)



from collections import OrderedDict



import matplotlib.pyplot as plt
import matplotlib
import logging

from sklearn.linear_model import Ridge

import os
import imageio
import gradio as gr
import numpy as np
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn import preprocessing
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
import sched
import threading

import random
import time
from PIL import Image


torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


import spaces
start_time = time.time()


output_hidden_state = False
device = "cuda"
dtype = torch.bfloat16
DEVICE = 'cuda'




####################### Setup Model
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler, LCMScheduler, AutoencoderTiny, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from PIL import Image
from transformers import CLIPVisionModelWithProjection
import uuid
import av

def write_video(file_name, images, fps=17):
    print('Saving')
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
    print('Saved')

def imio_write_video(file_name, images, fps=15):
    writer = imageio.get_writer(file_name, fps=fps)

    for im in images:
        writer.append_data(np.array(im))
    writer.close()




unet = UNet2DConditionModel.from_pretrained('rynmurdock/Sea_Claws', subfolder='unet',).to(dtype).to('cpu')
text_encoder = CLIPTextModel.from_pretrained('rynmurdock/Sea_Claws', subfolder='text_encoder', 
device_map='cpu').to(dtype)


adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM")
pipe = AnimateDiffPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", motion_adapter=adapter, torch_dtype=dtype, unet=unet, text_encoder=text_encoder)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")
pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora",)
pipe.set_adapters(["lcm-lora"], [.8])
pipe.fuse_lora()
pipe.unet.fuse_qkv_projections()
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()

#image_encoder = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder="sdxl_models/image_encoder", torch_dtype=dtype, 
#      device_map='cpu')

#pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", image_encoder=image_encoder, weight_name="ip-adapter_sd15_vit-G.bin", map_location='cpu')
#pipe.set_ip_adapter_scale(1.)

pipe.to(DEVICE)












def remove_all_hooks(model: torch.nn.Module) -> None:
    for name, child in model._modules.items():
        if child is not None:
            if hasattr(child, "_forward_hooks"):
                child._forward_hooks: Dict[int, Callable] = OrderedDict()
            elif hasattr(child, "_forward_pre_hooks"):
                child._forward_pre_hooks: Dict[int, Callable] = OrderedDict()
            elif hasattr(child, "_backward_hooks"):
                child._backward_hooks: Dict[int, Callable] = OrderedDict()
            remove_all_hooks(child)

remove_all_hooks(pipe.text_encoder)
remove_all_hooks(pipe.unet)

emb_len = 1280

STEPS = 8


from torchvision.transforms import PILToTensor
from PIL import Image


# TODO have an actual patch instead of forward hooks lol
prevs_df = pd.DataFrame(columns=['paths', 'embeddings', 'ips', 'user:rating', 'latest_user_to_rate', 'from_user_id'])
counter = 0
activation = []
our_guy = 0 * torch.randn(1, emb_len)
w = 0

def sd_get_in_activation():
    def hook(model, input, output):
        global counter
        global activation
        global our_guy
        
        #if counter == 'ayo':
        #    activation = output[1].mean(0)[None].to('cpu').to(torch.float32)
        #    return
        
        counter += 1
        if counter < 2:
            return output
        else:
            print('w', w)
            if w != 0:
                output[1, :] = output[1, :] + our_guy.to('cuda') * w
        activation = output[1].mean(0)[None].to('cpu').to(torch.float32)

        return output
    return hook

pipe.unet.mid_block.attentions[0].transformer_blocks[0].attn2.register_forward_hook(sd_get_in_activation())

pipe.to(device=DEVICE)


#ip_em = torch.load('/home/ryn_mote/Misc/linear_probe/1708230329.2274947.pt').view(1, 1, 1, emb_len).to(dtype=dtype, device=DEVICE)

# should already be...
#@torch.no_grad()
#def get_img_attentions(img):
#    img = PILToTensor()(img.resize((512, 512))) / 255 * 2 - 1 # TODO verify -1, 1 for VAE
#    img = img[None]
#    img = torch.cat([pipe.vae.config.scaling_factor * pipe.vae.encode(
#        i.unsqueeze(0).to(device='cuda', dtype=dtype)).latent_dist.sample().unsqueeze(2).repeat(1, 1, 16, 1, 1) for i in img])

#    t = torch.tensor(300).long()
#    noisedim = pipe.scheduler.add_noise(img, torch.randn_like(img), t)
#    input_ids = pipe.tokenizer('the scene', return_tensors='pt').to('cuda')
#    tembs = pipe.text_encoder(**input_ids)[0]
#    o = pipe.unet(noisedim, t, encoder_hidden_states=tembs, added_cond_kwargs={'image_embeds': ip_em})

#get_img_attentions(Image.open('/home/ryn_mote/Pictures/Pills-that-make-you-stare-at-meme-3.jpg'))


@spaces.GPU()
def generate_gpu(in_im_embs, prompt='the scene'):
    global activation
    global our_guy
    global counter

    counter = 0
    our_guy = in_im_embs
    
    output = pipe(prompt=prompt, guidance_scale=1., num_inference_steps=STEPS,)[0][0]
    
    # we try to get the first frame's activations
    #counter = 'ayo'
    #get_img_attentions(output[0])
    print('image is made')
    return output


def generate(in_im_embs, prompt='', set_path=None):
    output = generate_gpu(in_im_embs, prompt)
    # TODO put back
    nsfw = False#maybe_nsfw(output.frames[0][len(output.frames[0])//2])
    if set_path is None:
        name = str(uuid.uuid4()).replace("-", "")
        path = f"/tmp/{name}.mp4"
    else:
        path = set_path
    
    if nsfw:
        gr.Warning("NSFW content detected.")
        # TODO could return an automatic dislike of auto dislike on the backend for neither as well; just would need refactoring.
        return None
    
    
    output = output + list(reversed(output))

    write_video(path, output)
    return path


#######################

def get_user_emb(embs, ys):


    print('Gathering coefficients')
    # handle case where every instance of calibration videos is 'Neither' or 'Like' or 'Dislike'
    if len(list(set(ys))) <= 1:
        embs.append(.01*torch.randn(1, emb_len))
        embs.append(.01*torch.randn(1, emb_len))
        ys.append(0)
        ys.append(1)
        print('Fixing only one feedback class available.\n')
    
    indices = list(range(len(ys)))
    pos_indices = [i for i in indices if ys[i] == 1]
    neg_indices = [i for i in indices if ys[i] == 0]
    print(len(neg_indices), len(pos_indices))
    
    
    # we may have just encountered a rare multi-threading diffusers issue (https://github.com/huggingface/diffusers/issues/5749);
    # this ends up adding a rating but losing an embedding, it seems.
    # let's take off a rating if so to continue without indexing errors.
    if len(ys) > len(embs):
        print('ys are longer than embs; popping latest rating')
        ys.pop(-1)
    
    feature_embs = torch.cat([embs[i].to('cpu') for i in indices])
    feature_embs = feature_embs / feature_embs.norm(-1, keepdim=True)
    #scaler = preprocessing.StandardScaler().fit(feature_embs)
    #feature_embs = scaler.transform(feature_embs)
#     chosen_y = np.array([ys[i] for i in indices])
    
    lin_class = SVC(max_iter=20, kernel='linear', C=.1, class_weight='balanced').fit(feature_embs, ys)
    direction = torch.tensor(lin_class.coef_, dtype=torch.float32).detach().to('cpu')[0]
    
    embs = torch.cat(embs)
    # direction scaling from https://github.com/saprmarks/geometry-of-truth/blob/main/probes.py
    true_acts, false_acts = embs[torch.tensor(ys)==1], embs[torch.tensor(ys)==0]
    true_mean, false_mean = true_acts[:4].mean(0), false_acts[:4].mean(0)
    direction = direction / (direction.norm())
    direction.to('cpu')
    diff = (true_mean - false_mean) @ direction
    direction = diff * direction
    
    coef_ = direction.unsqueeze(0)
    im_emb = coef_.to(dtype=dtype)
    
    print('Gathered')
    return im_emb


def pluck_img(user_id, user_emb):
    print(user_id, 'user_id')
    not_rated_rows = prevs_df[[i[1]['user:rating'].get(user_id, 'gone') == 'gone' for i in prevs_df.iterrows()]]
    while len(not_rated_rows) == 0:
        not_rated_rows = prevs_df[[i[1]['user:rating'].get(user_id, 'gone') == 'gone' for i in prevs_df.iterrows()]]
        time.sleep(.1)
    # TODO optimize this lol
    best_sim = -10000000
    for i in not_rated_rows.iterrows():
        # TODO sloppy .to but it is 3am.
        sim = torch.cosine_similarity(i[1]['embeddings'].detach().to('cpu'), user_emb.detach().to('cpu'), -1)
        if sim > best_sim:
            best_sim = sim
            best_row = i[1]
    img = best_row['paths']
    return img


def background_next_image():
        global our_guy
        global prevs_df
        # only let it get N (maybe 3) ahead of the user
        #not_rated_rows = prevs_df[[i[1]['user:rating'] == {' ': ' '} for i in prevs_df.iterrows()]]
        rated_rows = prevs_df[[i[1]['user:rating'] != {' ': ' '} for i in prevs_df.iterrows()]]
        while len(rated_rows) < 4:
        #    not_rated_rows = prevs_df[[i[1]['user:rating'] == {' ': ' '} for i in prevs_df.iterrows()]]
            rated_rows = prevs_df[[i[1]['user:rating'] != {' ': ' '} for i in prevs_df.iterrows()]]
            time.sleep(.01)
            # TODO sleep less
#             print('all users have 4 or less rows rated')

        user_id_list = set(rated_rows['latest_user_to_rate'].to_list())
        for uid in user_id_list:
            rated_rows = prevs_df[[i[1]['user:rating'].get(uid, None) is not None for i in prevs_df.iterrows()]]
            not_rated_rows = prevs_df[[i[1]['user:rating'].get(uid, None) is None for i in prevs_df.iterrows()]]
            
            # we need to intersect not_rated_rows from this user's embed > 7. Just add a new column on which user_id spawned the 
            #   media. 
            
            unrated_from_user = not_rated_rows[[i[1]['from_user_id'] == uid for i in not_rated_rows.iterrows()]]
            rated_from_user = rated_rows[[i[1]['from_user_id'] == uid for i in rated_rows.iterrows()]]

            # we pop previous ratings if there are > 10
            if len(rated_from_user) >= 10:
                oldest = rated_from_user.iloc[0]['paths']
                prevs_df = prevs_df[prevs_df['paths'] != oldest]
            # we don't compute more after 10 are in the queue for them
            if len(unrated_from_user) >= 10:
                continue
            
            if len(rated_rows) < 6:
                print(f'latest user {uid} has < 6 rows') # or > 7 unrated rows')
                continue
            
            print(uid)
            embs, ys = pluck_embs_ys(uid)
            
            user_emb = get_user_emb(embs, ys)
            our_guy = user_emb
            img = generate(user_emb, prompt='a scene')
            embs = activation

            if img:
                tmp_df = pd.DataFrame(columns=['paths', 'embeddings', 'ips', 'user:rating', 'latest_user_to_rate'])
                tmp_df['paths'] = [img]
                tmp_df['embeddings'] = (embs,)
                tmp_df['user:rating'] = [{' ': ' '}]
                tmp_df['from_user_id'] = [uid]
                prevs_df = pd.concat((prevs_df, tmp_df))


                # we can free up storage by deleting the image
                if len(prevs_df) > 30:
                    cands = prevs_df.iloc[6:]
                    cands['sum_bad_ratings'] = [sum([int(t==0) for t in i.values()]) for i in cands['user:rating']]
                    worst_row = cands.loc[cands['sum_bad_ratings']==cands['sum_bad_ratings'].max()].iloc[0]
                    worst_path = worst_row['paths']
                    print('Removing worst row:', worst_row, 'from prevs_df of len', len(prevs_df))
                    if os.path.isfile(worst_path):
                        os.remove(worst_path)
                    else:
                        # If it fails, inform the user.
                        print("Error: %s file not found" % worst_path)

                    # only keep x images & embeddings & ips, then remove the most often disliked besides calibrating
                    prevs_df = prevs_df[prevs_df['paths'] != worst_path]
                    print('prevs_df is now length:', len(prevs_df))
    


def pluck_embs_ys(user_id):
    rated_rows = prevs_df[[i[1]['user:rating'].get(user_id, None) != None for i in prevs_df.iterrows()]]
    embs = rated_rows['embeddings'].to_list()
    ys = [i[user_id] for i in rated_rows['user:rating'].to_list()]
    return embs, ys

def next_image(calibrate_prompts, user_id):
    print(prevs_df)
    
    with torch.no_grad():
        if len(calibrate_prompts) > 0:
            print('######### Calibrating with sample media #########')
            cal_video = calibrate_prompts.pop(0)
            image = prevs_df[prevs_df['paths'] == cal_video]['paths'].to_list()[0]
            
            return image, calibrate_prompts
        else:
            print('######### Roaming #########')
            embs, ys = pluck_embs_ys(user_id)
            user_emb = get_user_emb(embs, ys)
            image = pluck_img(user_id, user_emb)
            return image, calibrate_prompts









def start(_, calibrate_prompts, user_id, request: gr.Request):
    user_id = int(str(time.time())[-7:].replace('.', ''))
    image, calibrate_prompts = next_image(calibrate_prompts, user_id)
    return [
            gr.Button(value='Like (L)', interactive=True), 
            gr.Button(value='Neither (Space)', interactive=True, visible=False), 
            gr.Button(value='Dislike (A)', interactive=True),
            gr.Button(value='Start', interactive=False),
            image,
            calibrate_prompts,
            user_id
            ]


def choose(img, choice, calibrate_prompts, user_id, request: gr.Request):
    global prevs_df
    
    
    if choice == 'Like (L)':
        choice = 1
    elif choice == 'Neither (Space)':
        img, calibrate_prompts = next_image(calibrate_prompts, user_id)
        return img, calibrate_prompts
    else:
        choice = 0
    
    # if we detected NSFW, leave that area of latent space regardless of how they rated chosen.
    # TODO skip allowing rating & just continue
    if img == None:
        print('NSFW -- choice is disliked')
        choice = 0
    
    row_mask = [p.split('/')[-1].replace('.mp4', '') in img for p in prevs_df['paths'].to_list()]
    # if it's still in the dataframe, add the choice
    if len(prevs_df.loc[row_mask, 'user:rating']) > 0:
        prevs_df.loc[row_mask, 'user:rating'][0][user_id] = choice
        prevs_df.loc[row_mask, 'latest_user_to_rate'] = user_id
    img, calibrate_prompts = next_image(calibrate_prompts, user_id)
    return img, calibrate_prompts

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
    './a painting of the land.mp4',
    './gorgeous weeping tree.mp4',
    './the ocean.mp4',
    './the skyline with neon.mp4',
    './a jellyfish.mp4',
    './still life in blue.mp4',
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
        b3 = gr.Button(value='Dislike (A)', interactive=False, elem_id="dislike")
        b2 = gr.Button(value='Neither (Space)', interactive=False, elem_id="neither", visible=False)
        b1 = gr.Button(value='Like (L)', interactive=False, elem_id="like")
        b1.click(
        choose, 
        [img, b1, calibrate_prompts, user_id],
        [img, calibrate_prompts],
        )
        b2.click(
        choose, 
        [img, b2, calibrate_prompts, user_id],
        [img, calibrate_prompts],
        )
        b3.click(
        choose, 
        [img, b3, calibrate_prompts, user_id],
        [img, calibrate_prompts],
        )
    with gr.Row():
        b4 = gr.Button(value='Start')
        b4.click(start,
                 [b4, calibrate_prompts, user_id],
                 [b1, b2, b3, b4, img, calibrate_prompts, user_id]
                 )
    with gr.Row():
        html = gr.HTML('''<div style='text-align:center; font-size:20px'>You will calibrate for several videos and then roam. </ div><br><br><br>
<div style='text-align:center; font-size:14px'>Note that while the AnimateLCM model with NSFW filtering is unlikely to produce NSFW images, this may still occur, and users should avoid NSFW content when rating.
</ div>
<br><br>
<div style='text-align:center; font-size:14px'>Thanks to @multimodalart for their contributions to the demo, esp. the interface and @maxbittker for feedback.
</ div>''')

# TODO quiet logging
log = logging.getLogger('log_here')
log.setLevel(logging.ERROR)

scheduler = BackgroundScheduler()
scheduler.add_job(func=background_next_image, trigger="interval", seconds=1)
scheduler.start()

#thread = threading.Thread(target=background_next_image,)
#thread.start()

# prep our calibration prompts
for im in [
    './a painting of the land.mp4',
    './gorgeous weeping tree.mp4',
    './the ocean.mp4',
    './the skyline with neon.mp4',
    './a jellyfish.mp4',
    './still life in blue.mp4',
    ]:
    #counter = 'ayo'
    pth = generate(in_im_embs=0*torch.randn(1, emb_len), prompt=im.replace('.mp4', ''), set_path=im)
    tmp_df = pd.DataFrame(columns=['paths', 'embeddings', 'ips', 'user:rating'])
    tmp_df['paths'] = (im,)
    image = list(imageio.imiter(pth))
    image = image[len(image)//2]
    tmp_df['embeddings'] = (activation,)
    tmp_df['user:rating'] = [{' ': ' '}]
    prevs_df = pd.concat((prevs_df, tmp_df))
    print(prevs_df)

w = 100
demo.launch(share=True, debug=True)


