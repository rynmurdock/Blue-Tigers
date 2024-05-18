
import torch

# lol
DEVICE = 'cuda'
STEPS = 6
output_hidden_state = False
device = "cuda"
dtype = torch.float16

import matplotlib.pyplot as plt
import matplotlib

from sklearn.linear_model import Ridge
from sfast.compilers.diffusion_pipeline_compiler import (compile, compile_unet,
                                                         CompilationConfig)
config = CompilationConfig.Default()

try:
    import triton
    config.enable_triton = True
except ImportError:
    print('Triton not installed, skip')
config.enable_cuda_graph = True
config.enable_jit = True
config.enable_jit_freeze = True
config.enable_cnn_optimization = True
config.preserve_parameters = False
config.prefer_lowp_gemm = True

import imageio
import gradio as gr
import numpy as np
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn import preprocessing
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler

import random
import time
from PIL import Image
from safety_checker_improved import maybe_nsfw


torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

prevs_df = pd.DataFrame(columns=['paths', 'embeddings', 'ips', 'user:rating', 'latest_user_to_rate'])

import spaces
prompt_list = [p for p in list(set(
                pd.read_csv('./twitter_prompts.csv').iloc[:, 1].tolist())) if type(p) == str]

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

def write_video(file_name, images, fps=17):
    print('Saving')
    container = av.open(file_name, mode="w")

    stream = container.add_stream("h264", rate=fps)
    # stream.options = {'preset': 'faster'}
    stream.thread_count = 0
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

bases = {
    #"basem": "emilianJR/epiCRealism"
    #SG161222/Realistic_Vision_V6.0_B1_noVAE
    #runwayml/stable-diffusion-v1-5
    #frankjoshua/realisticVisionV51_v51VAE
    #Lykon/dreamshaper-7
}

image_encoder = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder="sdxl_models/image_encoder", torch_dtype=dtype).to(DEVICE)
#vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=dtype)

# vae = ConsistencyDecoderVAE.from_pretrained("openai/consistency-decoder", torch_dtype=dtype)
# vae = compile_unet(vae, config=config)

#finetune_path = '''/home/ryn_mote/Misc/finetune-sd1.5/dreambooth-model best'''''
#unet = UNet2DConditionModel.from_pretrained(finetune_path+'/unet/').to(dtype)
#text_encoder = CLIPTextModel.from_pretrained(finetune_path+'/text_encoder/').to(dtype)


unet = UNet2DConditionModel.from_pretrained('rynmurdock/Sea_Claws', subfolder='unet').to(dtype)
text_encoder = CLIPTextModel.from_pretrained('rynmurdock/Sea_Claws', subfolder='text_encoder').to(dtype)

adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM")
pipe = AnimateDiffPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", motion_adapter=adapter, image_encoder=image_encoder, torch_dtype=dtype, unet=unet, text_encoder=text_encoder)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")
pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora",)
pipe.set_adapters(["lcm-lora"], [.9])
pipe.fuse_lora()

#pipe = AnimateDiffPipeline.from_pretrained('emilianJR/epiCRealism', torch_dtype=dtype, image_encoder=image_encoder)
#pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")
#repo = "ByteDance/AnimateDiff-Lightning"
#ckpt = f"animatediff_lightning_4step_diffusers.safetensors"


pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15_vit-G.bin", map_location='cpu')
# This IP adapter improves outputs substantially.
pipe.set_ip_adapter_scale(.8)
pipe.unet.fuse_qkv_projections()
#pipe.enable_free_init(method="gaussian", use_fast_sampling=True)

#pipe = compile(pipe, config=config)
pipe.to(device=DEVICE)
#pipe.unet = torch.compile(pipe.unet)
#pipe.vae = torch.compile(pipe.vae)


im_embs = torch.zeros(1, 1, 1, 1280, device=DEVICE, dtype=dtype)
output = pipe(prompt='a person', guidance_scale=0, added_cond_kwargs={}, ip_adapter_image_embeds=[im_embs], num_inference_steps=STEPS)
leave_im_emb, _ = pipe.encode_image(
                output.frames[0][len(output.frames[0])//2], DEVICE, 1, output_hidden_state
)
assert len(output.frames[0]) == 16
leave_im_emb.to('cpu')


@spaces.GPU()
def generate(in_im_embs):

    in_im_embs = in_im_embs.to('cuda').unsqueeze(0).unsqueeze(0)
    #im_embs = torch.cat((torch.zeros(1, 1280, device=DEVICE, dtype=dtype), in_im_embs), 0)

    output = pipe(prompt='', guidance_scale=0, added_cond_kwargs={}, ip_adapter_image_embeds=[in_im_embs], num_inference_steps=STEPS)

    im_emb, _ = pipe.encode_image(
                output.frames[0][len(output.frames[0])//2], DEVICE, 1, output_hidden_state
            )
    im_emb = im_emb.detach().to('cpu')

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

# TODO add to state instead of shared across all
glob_idx = 0

# TODO
# We can keep a df of media paths, embeddings, and user ratings. 
#   We can drop by lowest user ratings to keep enough RAM available when we get too many rows.
#   We can continuously update by who is most recently active in the background & server as we go, plucking using "has been seen" and similarity
#   to user embeds

def get_user_emb(embs, ys):
    # handle case where every instance of calibration prompts is 'Neither' or 'Like' or 'Dislike'
    if len(list(set(ys))) <= 1:
        embs.append(.01*torch.randn(1280))
        embs.append(.01*torch.randn(1280))
        ys.append(0)
        ys.append(1)
    
    indices = list(range(len(embs)))
    # sample only as many negatives as there are positives
    pos_indices = [i for i in indices if ys[i] == 1]
    neg_indices = [i for i in indices if ys[i] == 0]
    #lower = min(len(pos_indices), len(neg_indices))
    #neg_indices = random.sample(neg_indices, lower)
    #pos_indices = random.sample(pos_indices, lower)
    print(len(neg_indices), len(pos_indices))
    
    
    # we may have just encountered a rare multi-threading diffusers issue (https://github.com/huggingface/diffusers/issues/5749);
    # this ends up adding a rating but losing an embedding, it seems.
    # let's take off a rating if so to continue without indexing errors.
    if len(ys) > len(embs):
        print('ys are longer than embs; popping latest rating')
        ys.pop(-1)
    
    feature_embs = np.array(torch.stack([embs[i].to('cpu') for i in indices] + [leave_im_emb[0].to('cpu')]).to('cpu'))
    #scaler = preprocessing.StandardScaler().fit(feature_embs)
    #feature_embs = scaler.transform(feature_embs)
    chosen_y = np.array([ys[i] for i in indices] + [0])
    
    print('Gathering coefficients')
    #lin_class = Ridge(fit_intercept=False).fit(feature_embs, chosen_y)
    lin_class = SVC(max_iter=50000, kernel='linear', C=.1, class_weight='balanced').fit(feature_embs, chosen_y)
    coef_ = torch.tensor(lin_class.coef_, dtype=torch.double)
    coef_ = coef_ / coef_.abs().max() * 3
    print(coef_.shape, 'COEF')

    print(coef_)
    print('Gathered')

    w = 1# if len(embs) % 2 == 0 else 0
    im_emb = w * coef_.to(dtype=dtype)
    return im_emb


def pluck_img(user_id, user_emb):
    not_rated_rows = prevs_df[[i['user:rating'].get(user_id, None) is None for i in prevs_df]]
    rated_rows = prevs_df[[i['user:rating'].get(user_id, None) is not None for i in prevs_df]]
    while len(not_rated_rows) == 0:
        not_rated_rows = [i for i in prevs_df if i['user:rating'].get(user_id, None) == None]
        time.sleep(.01)
    # TODO optimize this lol
    best_sim = -1
    for i in not_rated_rows:
        sim = torch.cosine_similarity(i['embeddings'], user_emb)
        if sim > best_sim:
            best_sim = sim
            best_row = i
    img = best_row['paths']
    embs = rated_rows['embeddings'].to_list()
    ys = [i[user_id] for i in rated_rows['user:rating'].to_list()]
    
    return embs, ys, img


def background_next_image():
    global prevs_df
    latest_user_id = prevs_df.iloc[-1]['latest_user_to_rate']
    print([i[1]['user:rating'] for i in prevs_df.iterrows()])
    rated_rows = prevs_df[[i[1]['user:rating'] != None and i[1]['user:rating'].get(latest_user_id, None) != None for i in prevs_df.iterrows()]]
    
    ys = [i[user_id] for i in rated_rows['user:rating'].to_list()]
    embs = [i[user_id] for i in rated_rows['embeddings'].to_list()]
    
    user_emb = get_user_emb(embs, ys)
    img, embs = generate(user_emb)
    tmp_df = pd.DataFrame(columns=['paths', 'embeddings', 'ips', 'user:rating'])
    tmp_df['paths'] = [img]
    tmp_df['embeddings'] = [embs]
    tmp_df['user:rating'] = [{' ': ' '}]
    tmp_df['latest_user_to_rate'] = user_id
    prevs_df = pd.concat((prevs_df, tmp_df))
    # we can free up storage by deleting the image
    if len(prevs_df) > 50:
        oldest_path = prevs_df.iloc[0]['paths']
        if os.path.isfile(oldest_path):
            os.remove(oldest_path)
        else:
            # If it fails, inform the user.
            print("Error: %s file not found" % oldest_path)
        # only keep 50 images & embeddings & ips, then remove oldest
        prevs_df = prevs_df.iloc[1:]
    print(prevs_df)
    

def pluck_embs_ys(user_id):
    rated_rows = prevs_df[[i[1]['user:rating'].get(user_id, None) != None for i in prevs_df.iterrows()]]
    not_rated_rows = prevs_df[[i['user:rating'].get(user_id, None) is None for i in prevs_df]]
    while len(not_rated_rows) == 0:
        not_rated_rows = [i for i in prevs_df if i['user:rating'].get(user_id, None) == None]
        time.sleep(.01)
    
    embs = rated_rows['embeddings'].to_list()
    ys = [i[user_id] for i in rated_rows['user:rating'].to_list()]
    
    return embs, ys

def next_image(calibrate_prompts, user_id):
    global glob_idx
    glob_idx = glob_idx + 1
    user_id_t = user_id[0]
    
    with torch.no_grad():
        if len(calibrate_prompts) > 0:
            print('######### Calibrating with sample prompts #########')
            cal_video = calibrate_prompts.pop(0)
            image = prevs_df[prevs_df['paths'] == cal_video]['paths'].to_list()
            return image, calibrate_prompts
        else:
            print('######### Roaming #########')
            
            embs, ys = pluck_embs_ys(user_id_t)
            image = pluck_img(user_id_t, im_emb)
            return image, calibrate_prompts









def start(_, calibrate_prompts, user_id, request: gr.Request):
    user_id_t = user_id[0]
    image, calibrate_prompts = next_image(calibrate_prompts, user_id_t)
    print(image, calibrate_prompts)
    return [
            gr.Button(value='Like (L)', interactive=True), 
            gr.Button(value='Neither (Space)', interactive=True), 
            gr.Button(value='Dislike (A)', interactive=True),
            gr.Button(value='Start', interactive=False),
            image[0],
            calibrate_prompts
            ]


def choose(img, choice, user_id, calibrate_prompts, request: gr.Request):
    global prevs_df
    
    
    if choice == 'Like (L)':
        choice = 1
    elif choice == 'Neither (Space)':
        img, calibrate_prompts = next_image(user_id, calibrate_prompts)
        return img, calibrate_prompts
    else:
        choice = 0
    
    # if we detected NSFW, leave that area of latent space regardless of how they rated chosen.
    # TODO skip allowing rating & just continue
    if img == None:
        print('NSFW -- choice is disliked')
        choice = 0
    
    prevs_df['user:rating']['user_id'] = choice
    
    img, calibrate_prompts = next_image(user_id, calibrate_prompts)
    
    
    return img[0], calibrate_prompts

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
    user_id = gr.State([torch.randint(2**6, (1,))])
    calibrate_prompts = gr.State([
    './first.mp4',
    './second.mp4',
    './third.mp4',
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
        include_audio=False,
        elem_id="video_output"
       )
        img.play(l, js='''document.querySelector('[data-testid="Lightning-player"]').loop = true''')
    with gr.Row(equal_height=True):
        b3 = gr.Button(value='Dislike (A)', interactive=False, elem_id="dislike")
        b2 = gr.Button(value='Neither (Space)', interactive=False, elem_id="neither")
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
                 [b1, b2, b3, b4, img, calibrate_prompts]
                 )
    with gr.Row():
        html = gr.HTML('''<div style='text-align:center; font-size:20px'>You will calibrate for several videos and then roam. </ div><br><br><br>
<div style='text-align:center; font-size:14px'>Note that while the AnimateLCM model with NSFW filtering is unlikely to produce NSFW images, this may still occur, and users should avoid NSFW content when rating.
</ div>
<br><br>
<div style='text-align:center; font-size:14px'>Thanks to @multimodalart for their contributions to the demo, esp. the interface and @maxbittker for feedback.
</ div>''')

scheduler = BackgroundScheduler()
scheduler.add_job(func=background_next_image, trigger="interval", seconds=10)
scheduler.start()

# prep our calibration prompts
for im in [
    './first.mp4',
    './second.mp4',
    './third.mp4',
    ]:
    tmp_df = pd.DataFrame(columns=['paths', 'embeddings', 'ips', 'user:rating'])
    tmp_df['paths'] = [im]
    image = list(imageio.imiter(im))[0]
    im_emb, _ = pipe.encode_image(
                image, DEVICE, 1, output_hidden_state
            )

    tmp_df['embeddings'] = [im_emb.detach().to('cpu')]
    tmp_df['user:rating'] = [{' ': ' '}]
    prevs_df = pd.concat((prevs_df, tmp_df))


demo.launch(share=True)
