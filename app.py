

# TODO unify/merge origin and this
# TODO save & restart from (if it exists) dataframe parquet

import torch

# lol
DEVICE = 'cuda'
STEPS = 6
output_hidden_state = False
device = "cuda"
dtype = torch.bfloat16
N_IMG_EMBS = 3

import logging
import os
import imageio
import gradio as gr
import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler

import random
import time
from PIL import Image



torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

prevs_df = pd.DataFrame(columns=['paths', 'embeddings', 'ips', 'user:rating', 'latest_user_to_rate', 'from_user_id', 'text', 'audio'])

import spaces
start_time = time.time()

####################### Setup Models
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler, LCMScheduler, AutoencoderTiny, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from PIL import Image
from transformers import CLIPVisionModelWithProjection
import uuid
import av
import torchvision

def write_video(file_name, images, fps=17):
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





# VILA
#####################################################################################################

from transformers import pipeline

from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IMAGE_TOKEN, IMAGE_PLACEHOLDER,
                             IMAGE_TOKEN_INDEX)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (KeywordsStoppingCriteria, get_model_name_from_path,
                            process_images, tokenizer_image_token)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init



vilap = 'Efficient-Large-Model/VILA1.5-3b'
model_name = get_model_name_from_path(vilap)
tokenizer, vila, image_processor, context_len = load_pretrained_model(vilap, model_name, None, torch_dtype=torch.bfloat16, 
                                                                        device=0,
                                                                        use_safetensors=True, load_8bit=True)
#vila = torch.compile(vila)
############################################################################################################
@spaces.GPU()
def eval_model(images, qs=f"<image> is bad. <image> and <image> are good. Give a one-word description of a different good image.", model_name='vicuna_v1'):
    global vila

    images = [torchvision.transforms.ToPILImage(mode='RGB')(i) for i in images]
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if vila.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if DEFAULT_IMAGE_TOKEN not in qs:
            print("no <image> tag found in input. Automatically append one at the beginning of text.")
            # do not repeatively append the prompt.
            if vila.config.mm_use_im_start_end:
                qs = (image_token_se + "\n") * len(images) + qs
            else:
                qs = (DEFAULT_IMAGE_TOKEN + "\n") * len(images) + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"


    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
        
    images_tensor = process_images(images, image_processor, vila.config).to(torch.float32)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.cuda.amp.autocast():
        with torch.inference_mode():
            output_ids = vila.generate(
                input_ids.to(torch.float32),
                images=[
                    images_tensor.to(torch.float32),
                ],
                do_sample=True,
                temperature=.8,
                top_p=.97,
                max_new_tokens=128,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    print(outputs)
    return outputs

############################################################################################################
image_encoder = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder="sdxl_models/image_encoder", torch_dtype=dtype, 
device_map='cuda')
#vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=dtype)

# vae = ConsistencyDecoderVAE.from_pretrained("openai/consistency-decoder", torch_dtype=dtype)
# vae = compile_unet(vae, config=config)

#finetune_path = '''/home/ryn_mote/Misc/finetune-sd1.5/dreambooth-model best'''''
#unet = UNet2DConditionModel.from_pretrained(finetune_path+'/unet/').to(dtype)
#text_encoder = CLIPTextModel.from_pretrained(finetune_path+'/text_encoder/').to(dtype)


unet = UNet2DConditionModel.from_pretrained('rynmurdock/Sea_Claws', subfolder='unet',).to(dtype).to('cpu')
text_encoder = CLIPTextModel.from_pretrained('rynmurdock/Sea_Claws', subfolder='text_encoder', 
device_map='cpu').to(dtype)

adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM")
pipe = AnimateDiffPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", motion_adapter=adapter, image_encoder=image_encoder, torch_dtype=dtype,     
                                            unet=unet, text_encoder=text_encoder)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")
pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora",)
pipe.set_adapters(["lcm-lora"], [.95])
pipe.fuse_lora()
pipe.enable_vae_slicing()

#pipe = AnimateDiffPipeline.from_pretrained('emilianJR/epiCRealism', torch_dtype=dtype, image_encoder=image_encoder)
#pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")
#repo = "ByteDance/AnimateDiff-Lightning"
#ckpt = f"animatediff_lightning_4step_diffusers.safetensors"


pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15_vit-G.bin", map_location='cpu')
# This IP adapter improves outputs substantially.
pipe.set_ip_adapter_scale(.8) # .6
pipe.unet.fuse_qkv_projections()
#pipe.enable_free_init(method="gaussian", use_fast_sampling=True)



pipe.to(device=DEVICE)

#pipe.unet = torch.compile(pipe.unet)
#pipe.vae = torch.compile(pipe.vae)


##########################################################################################################################
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

os.environ['HF_TOKEN'] = "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"

# Download model
audio_model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0", )
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]

audio_model = audio_model.to(device)

def get_audio(text):

    # Set up text and timing conditioning
    conditioning = [{
        "prompt": text,
        "seconds_start": 0, 
        "seconds_total": 4
    }]

    # Generate stereo audio
    output = generate_diffusion_cond(
        audio_model,
        steps=40,
        cfg_scale=7,
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=0.3,
        sigma_max=500,
        sampler_type="dpmpp-3m-sde",
        device=device
    )

    # Rearrange audio batch to a single sequence
    output = rearrange(output, "b d n -> d (b n)")

    # Peak normalize, clip, convert to int16, and save to file
    output = output[:, :4*sample_rate]
    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    name = str(uuid.uuid4()).replace("-", "")
    path = f"/tmp/{name}.mp4"
    
    torchaudio.save(path, output, sample_rate)
    return path






##########################################################################################################################

from safety_checker_improved import maybe_nsfw

@spaces.GPU()
def generate_gpu(in_im_embs, prompt='the scene'):
    with torch.no_grad():
        in_im_embs = in_im_embs.to('cuda').unsqueeze(0).unsqueeze(0)

        output = pipe(prompt=prompt, guidance_scale=1, added_cond_kwargs={}, ip_adapter_image_embeds=[in_im_embs], num_inference_steps=STEPS,)

        im_emb, _ = pipe.encode_image(
                    output.frames[0][len(output.frames[0])//2], 'cuda', 1, output_hidden_state
                )
        audio = get_audio(prompt)
    return output, im_emb, audio


def generate(in_im_embs, prompt='the scene'):
    output, im_emb, audio = generate_gpu(in_im_embs, prompt)
    nsfw = maybe_nsfw(output.frames[0][len(output.frames[0])//2])
    name = str(uuid.uuid4()).replace("-", "")
    path = f"/tmp/{name}.mp4"
    
    if nsfw:
        gr.Warning("NSFW content detected.")
        # TODO could return an automatic dislike of auto dislike on the backend for neither as well; just would need refactoring.
        return None, im_emb, audio
    
    
    output.frames[0] = output.frames[0] + list(reversed(output.frames[0]))
    write_video(path, output.frames[0])
    
    return path, im_emb, audio


#######################

def get_user_emb(embs, ys):
    # handle case where every instance of calibration videos is 'Neither' or 'Like' or 'Dislike'
    
    if len(list(ys)) <= 10:
        aways = [torch.zeros_like(embs[0]) for i in range(10)]
        embs += aways
        awal = [0 for i in range(5)] + [1 for i in range(5)]
        ys += awal
    
    indices = list(range(len(embs)))
    # sample only as many negatives as there are positives
    pos_indices = [i for i in indices if ys[i] == 1]
    neg_indices = [i for i in indices if ys[i] == 0]
    #lower = min(len(pos_indices), len(neg_indices))
    #neg_indices = random.sample(neg_indices, lower)
    #pos_indices = random.sample(pos_indices, lower)
    
    
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
    
    if feature_embs.norm() != 0:
        feature_embs = feature_embs / feature_embs.norm()
    
    #lin_class = Ridge(fit_intercept=False).fit(feature_embs, chosen_y)
    #class_weight='balanced'
    lin_class = SVC(max_iter=500, kernel='linear', C=.1, ).fit(feature_embs.squeeze(), chosen_y)
    coef_ = torch.tensor(lin_class.coef_, dtype=torch.float32).detach().to('cpu')
    coef_ = coef_ / coef_.abs().max()

    w = 1# if len(embs) % 2 == 0 else 0
    im_emb = w * coef_.to(dtype=dtype)
    return im_emb


def pluck_img(user_id, user_emb):
    not_rated_rows = prevs_df[[i[1]['user:rating'].get(user_id, 'gone') == 'gone' for i in prevs_df.iterrows()]]
    while len(not_rated_rows) == 0:
        not_rated_rows = prevs_df[[i[1]['user:rating'].get(user_id, 'gone') == 'gone' for i in prevs_df.iterrows()]]
        time.sleep(.001)
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
    audio = best_row.get('audio')
    if not isinstance(audio, str) or audio == 'nan':
        audio = None
    return img, text, audio


def background_next_image():
        global prevs_df
        # only let it get N (maybe 3) ahead of the user
        #not_rated_rows = prevs_df[[i[1]['user:rating'] == {' ': ' '} for i in prevs_df.iterrows()]]
        rated_rows = prevs_df[[i[1]['user:rating'] != {' ': ' '} for i in prevs_df.iterrows()]]
        while len(rated_rows) < 5:
        #    not_rated_rows = prevs_df[[i[1]['user:rating'] == {' ': ' '} for i in prevs_df.iterrows()]]
            rated_rows = prevs_df[[i[1]['user:rating'] != {' ': ' '} for i in prevs_df.iterrows()]]
            time.sleep(.01)

        user_id_list = set(rated_rows['latest_user_to_rate'].to_list())
        for uid in user_id_list:
            rated_rows = prevs_df[[i[1]['user:rating'].get(uid, None) is not None for i in prevs_df.iterrows()]]
            not_rated_rows = prevs_df[[i[1]['user:rating'].get(uid, None) is None for i in prevs_df.iterrows()]]
            
            # we need to intersect not_rated_rows from this user's embed > 7. Just add a new column on which user_id spawned the 
            #   media. 
            
            unrated_from_user = not_rated_rows[[i[1]['from_user_id'] == uid for i in not_rated_rows.iterrows()]]
            rated_from_user = rated_rows[[i[1]['from_user_id'] == uid for i in rated_rows.iterrows()]]

            # we pop previous ratings if there are > n
            if len(rated_from_user) >= 25:
                oldest = rated_from_user.iloc[0]['paths']
                prevs_df = prevs_df[prevs_df['paths'] != oldest]
            # we don't compute more after n are in the queue for them
            if len(unrated_from_user) >= 20:
                continue
            
            embs, ys = pluck_embs_ys(uid)
            user_emb = get_user_emb(embs, ys) * 3
            
            
            pos_mask = [i[uid] == 1 for i in rated_rows['user:rating'].to_list()]
            neg_mask = [i[uid] == 0 for i in rated_rows['user:rating'].to_list()]
            paths_pos_from_user = rated_rows[pos_mask]['paths'].to_list()
            paths_neg_from_user= rated_rows[neg_mask]['paths'].to_list()
            # TODO keep middle frame in row
            
            images = [paths_neg_from_user[random.randint(0, len(paths_neg_from_user)-1)]]
            for _ in range(N_IMG_EMBS):
                images += [paths_pos_from_user[random.randint(0, len(paths_pos_from_user)-1)]]
            ims = []
            for im in images:
                image = list(imageio.imiter(im))
                image = image[len(image)//2]
                ims.append(image)
            text = eval_model(ims)
            img, embs, audio = generate(user_emb, text)
            
            if img:
                tmp_df = pd.DataFrame(columns=['paths', 'embeddings', 'ips', 'user:rating', 'latest_user_to_rate', 'text', 'audio'])
                tmp_df['paths'] = [img]
                tmp_df['embeddings'] = [embs.detach().to(device='cpu', dtype=torch.float32)]
                tmp_df['user:rating'] = [{' ': ' '}]
                tmp_df['from_user_id'] = [uid]
                tmp_df['text'] = [text]
                tmp_df['audio'] = [audio]
                
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
    return embs, ys

def next_image(calibrate_prompts, user_id):
    with torch.no_grad():
        if len(calibrate_prompts) > 0:
            cal_video = calibrate_prompts.pop(0)
            image = prevs_df[prevs_df['paths'] == cal_video]['paths'].to_list()[0]
            return image, calibrate_prompts, '', None
        else:
            embs, ys = pluck_embs_ys(user_id)
            user_emb = get_user_emb(embs, ys) * 3
            image, text, audio = pluck_img(user_id, user_emb)
            return image, calibrate_prompts, text, audio



def start(_, calibrate_prompts, user_id, request: gr.Request):
    user_id = int(str(time.time())[-7:].replace('.', ''))
    image, calibrate_prompts, text, audio = next_image(calibrate_prompts, user_id)
    return [
            gr.Button(value='Like (L)', interactive=True), 
            gr.Button(value='Neither (Space)', interactive=True, visible=False), 
            gr.Button(value='Dislike (A)', interactive=True),
            gr.Button(value='Start', interactive=False),
            image,
            calibrate_prompts,
            user_id,
            None
            ]


def choose(img, choice, calibrate_prompts, user_id, request: gr.Request):
    global prevs_df
    
    
    if choice == 'Like (L)':
        choice = 1
    elif choice == 'Neither (Space)':
        img, calibrate_prompts, text, audio = next_image(calibrate_prompts, user_id)
        return img, calibrate_prompts, text
    else:
        choice = 0
    
    # if we detected NSFW, leave that area of latent space regardless of how they rated chosen.
    # TODO skip allowing rating & just continue
    if img == None:
        print('NSFW -- choice is disliked')
        choice = 0
    
    row_mask = [p.split('/')[-1] in img for p in prevs_df['paths'].to_list()]
    # if it's still in the dataframe, add the choice
    if len(prevs_df.loc[row_mask, 'user:rating']) > 0:
        prevs_df.loc[row_mask, 'user:rating'][0][user_id] = choice
        prevs_df.loc[row_mask, 'latest_user_to_rate'] = [user_id]
    img, calibrate_prompts, text, audio = next_image(calibrate_prompts, user_id)
    return img, calibrate_prompts, text, audio

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
      fadeInOut(target, '#0099ff');
    } else if (target.id === 'like') {
      fadeInOut(target, '#0099ff');
    } else if (target.id === 'neither') {
      fadeInOut(target, '#cccccc');
    }
});

</script>
'''

with gr.Blocks(css=css, head=js_head, theme=gr.themes.Soft()) as demo:
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
    def l(audio):
        print(audio)
        return audio

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
    with gr.Row():
        audio = gr.Audio(interactive=False, visible=False, label='Audio', autoplay=True)
        audio.play(l, js='''document.querySelector("#waveform > div").shadowRoot.querySelector("audio").loop = true''',)
        text = gr.Textbox(interactive=False, visible=False, label='Text')
    with gr.Row(equal_height=True):
        b3 = gr.Button(value='Dislike (A)', interactive=False, elem_id="dislike")
        b2 = gr.Button(value='Neither (Space)', interactive=False, elem_id="neither", visible=False)
        b1 = gr.Button(value='Like (L)', interactive=False, elem_id="like")
        b1.click(
        choose, 
        [img, b1, calibrate_prompts, user_id],
        [img, calibrate_prompts, text, audio],
        )
        b2.click(
        choose, 
        [img, b2, calibrate_prompts, user_id],
        [img, calibrate_prompts, text, audio],
        )
        b3.click(
        choose, 
        [img, b3, calibrate_prompts, user_id],
        [img, calibrate_prompts, text, audio],
        )
    with gr.Row():
        b4 = gr.Button(value='Start')
        b4.click(start,
                 [b4, calibrate_prompts, user_id],
                 [b1, b2, b3, b4, img, calibrate_prompts, user_id, audio]
                 )
    with gr.Row():
        html = gr.HTML('''<div style='text-align:center; font-size:20px'>You will calibrate for several videos and then roam. </ div><br><br><br>
<div style='text-align:center; font-size:14px'>Note that while the AnimateLCM model with NSFW filtering & the Villa model are unlikely to produce NSFW images, this may still occur, and users should avoid NSFW content when rating.
</ div>
<br><br>
<div style='text-align:center; font-size:14px'>Thanks to @multimodalart for their contributions to the demo, esp. the interface and @maxbittker for feedback.
</ div>''')

# TODO quiet logging
log = logging.getLogger('log_here')
log.setLevel(logging.ERROR)

scheduler = BackgroundScheduler()
scheduler.add_job(func=background_next_image, trigger="interval", seconds=.5)
scheduler.start()


# prep our calibration videos
for im in [
    './first.mp4',
    './second.mp4',
    './third.mp4',
    './fourth.mp4',
    './fifth.mp4',
    './sixth.mp4',
    './seventh.mp4',
    './eigth.mp4',
    './ninth.mp4',
    './tenth.mp4',
    ]:
    tmp_df = pd.DataFrame(columns=['paths', 'embeddings', 'ips', 'user:rating', 'text',])
    tmp_df['paths'] = [im]
    image = list(imageio.imiter(im))
    image = image[len(image)//2]
    tmp_df['embeddings'] = [torch.load(im.replace('mp4', 'im_.pt'))]
    tmp_df['user:rating'] = [{' ': ' '}]
    prevs_df = pd.concat((prevs_df, tmp_df))

if __name__ == "__main__":
    demo.launch(share=True, server_port=8443)


