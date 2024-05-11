
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

import random
import time
from PIL import Image
from safety_checker_improved import maybe_nsfw


torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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

#finetune_path = '''/home/ryn_mote/Misc/finetune-sd1.5/yes dreambooth-model/'''''
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
pipe.set_ip_adapter_scale(.6)
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
def generate(prompt, in_im_embs=None, base='basem'):

    if in_im_embs == None:
        #in_im_embs = torch.randn(1, 1, 1, 1280, device=DEVICE, dtype=dtype)
        in_im_embs= torch.load('1708230329.2274947.pt').to('cuda').unsqueeze(0).unsqueeze(0).to(dtype)
        in_im_embs = in_im_embs / in_im_embs.norm()
    else:
        in_im_embs = in_im_embs.to('cuda').unsqueeze(0).unsqueeze(0)
        #im_embs = torch.cat((torch.zeros(1, 1280, device=DEVICE, dtype=dtype), in_im_embs), 0)

    output = pipe(prompt=prompt, guidance_scale=0, added_cond_kwargs={}, ip_adapter_image_embeds=[in_im_embs], num_inference_steps=STEPS)

    im_emb, _ = pipe.encode_image(
                output.frames[0][len(output.frames[0])//2], DEVICE, 1, output_hidden_state
            )

    nsfw = maybe_nsfw(output.frames[0][len(output.frames[0])//2])
    
    name = str(uuid.uuid4()).replace("-", "")
    path = f"/tmp/{name}.mp4"
    
    if nsfw:
        gr.Warning("NSFW content detected.")
        # TODO could return an automatic dislike of auto dislike on the backend for neither as well; just would need refactoring.
        return None, im_emb
    
    plt.close('all')
    plt.hist(np.array(im_emb.to('cpu')).flatten(), bins=5)
    plt.savefig('real_im_emb_plot.jpg')
    
    output.frames[0] = output.frames[0] + list(reversed(output.frames[0]))

    write_video(path, output.frames[0])
    return path, im_emb.to('cpu')


#######################

# TODO add to state instead of shared across all
glob_idx = 0

def next_image(embs, ys, calibrate_prompts):
    global glob_idx
    glob_idx = glob_idx + 1
        
    with torch.no_grad():
        if len(calibrate_prompts) > 0:
            print('######### Calibrating with sample prompts #########')
            prompt = calibrate_prompts.pop(0)
            print(prompt)
            image, img_embs = generate(prompt)
            embs += img_embs
            print(len(embs))
            return image, embs, ys, calibrate_prompts
        else:
            print('######### Roaming #########')
            
            # sample a .8 of rated embeddings for some stochasticity, or at least two embeddings.
            # could take a sample < len(embs)
            #n_to_choose = max(int((len(embs))), 2)
            #indices = random.sample(range(len(embs)), n_to_choose)
            
            # handle case where every instance of calibration prompts is 'Neither' or 'Like' or 'Dislike'
            if len(list(set(ys))) <= 1:
                embs.append(.01*torch.randn(1280))
                embs.append(.01*torch.randn(1280))
                ys.append(-1)
                ys.append(1)
            #if len(list(ys)) < 10:
            #    embs += [.01*torch.randn(1280)] * 3
            #    ys += [-1] * 3
            
            indices = range(len(embs))            
            # sample only as many negatives as there are positives
            pos_indices = [i for i in indices if ys[i] == 1]
            neg_indices = [i for i in indices if ys[i] == -1]
            #lower = min(len(pos_indices), len(neg_indices))
            #neg_indices = random.sample(neg_indices, lower)
            #pos_indices = random.sample(pos_indices, lower)
            #indices = neg_indices + pos_indices
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
            chosen_y = np.array([ys[i] for i in indices] + [-1])
            
            print('Gathering coefficients')
            #lin_class = Ridge(fit_intercept=False).fit(feature_embs, chosen_y)
            lin_class = SVC(max_iter=50000, kernel='linear', C=.1, class_weight='balanced').fit(feature_embs, chosen_y)
            coef_ = torch.tensor(lin_class.coef_, dtype=torch.double).unsqueeze(0)
            coef_ = coef_ / coef_.abs().max() * 3
            print(coef_.shape, 'COEF')

            plt.close('all')
            plt.hist(np.array(coef_).flatten(), bins=5)
            plt.savefig('plot.jpg')
            print(coef_)
            print('Gathered')

            rng_prompt = random.choice(prompt_list)
            w = 1# if len(embs) % 2 == 0 else 0
            im_emb = w * coef_.to(dtype=dtype)

            prompt= '' if glob_idx % 3 != 0 else rng_prompt
            print(prompt)
            image, im_emb = generate(prompt, im_emb)
            embs += im_emb
            
            
            return image, embs, ys, calibrate_prompts









def start(_, embs, ys, calibrate_prompts):
    image, embs, ys, calibrate_prompts = next_image(embs, ys, calibrate_prompts)
    return [
            gr.Button(value='Like (L)', interactive=True), 
            gr.Button(value='Neither (Space)', interactive=True), 
            gr.Button(value='Dislike (A)', interactive=True),
            gr.Button(value='Start', interactive=False),
            image,
            embs,
            ys,
            calibrate_prompts
            ]


def choose(img, choice, embs, ys, calibrate_prompts):
    if choice == 'Like (L)':
        choice = 1
    elif choice == 'Neither (Space)':
        embs = embs[:-1]
        img, embs, ys, calibrate_prompts = next_image(embs, ys, calibrate_prompts)
        return img, embs, ys, calibrate_prompts
    else:
        choice = -1

    # if we detected NSFW, leave that area of latent space regardless of how they rated chosen.
    # TODO skip allowing rating
    if img == None:
        print('NSFW -- choice is disliked')
        choice = -1
    
    ys += [choice]*1
    img, embs, ys, calibrate_prompts = next_image(embs, ys, calibrate_prompts)
    return img, embs, ys, calibrate_prompts

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
    embs = gr.State([])
    ys = gr.State([])
    calibrate_prompts = gr.State([
    'the moon is melting into my glass of tea',
    'The city is made of wires, lightning, and science fiction.',
    'a sea slug -- pair of claws scuttling -- jelly fish floats',
    'an adorable creature. It may be a goblin or a pig or a slug.',
    'an animation about a gorgeous nebula',
    'a sketch of an impressive mountain by da vinci',
    'a watercolor painting: the octopus writhes',
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
        [img, b1, embs, ys, calibrate_prompts],
        [img, embs, ys, calibrate_prompts]
        )
        b2.click(
        choose, 
        [img, b2, embs, ys, calibrate_prompts],
        [img, embs, ys, calibrate_prompts]
        )
        b3.click(
        choose, 
        [img, b3, embs, ys, calibrate_prompts],
        [img, embs, ys, calibrate_prompts]
        )
    with gr.Row():
        b4 = gr.Button(value='Start')
        b4.click(start,
                 [b4, embs, ys, calibrate_prompts],
                 [b1, b2, b3, b4, img, embs, ys, calibrate_prompts])
    with gr.Row():
        html = gr.HTML('''<div style='text-align:center; font-size:20px'>You will calibrate for several prompts and then roam. </ div><br><br><br>
<div style='text-align:center; font-size:14px'>Note that while the AnimateLCM model with NSFW filtering is unlikely to produce NSFW images, this may still occur, and users should avoid NSFW content when rating.
</ div>
<br><br>
<div style='text-align:center; font-size:14px'>Thanks to @multimodalart for their contributions to the demo, esp. the interface and @maxbittker for feedback.
</ div>''')

demo.launch(share=True)
