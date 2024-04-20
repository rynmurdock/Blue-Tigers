
# weights from https://github.com/LAION-AI/safety-pipeline/tree/main
from PIL import Image
import tensorflow_hub as hub
import tensorflow
import numpy as np
#import h5py
import keras
import sys
sys.path.append('/home/ryn_mote/Misc/generative_recommender/gradio_video/automl/efficientnetv2/')
import tensorflow as tf

# edited hparams to num_classes=5
#model = effnetv2_model.EffNetV2Model('efficientnetv2-b2')
model = tf.keras.models.load_model('nsfweffnetv2-b02-3epochs.h5',custom_objects={"KerasLayer":hub.KerasLayer})
# The image classifier had been trained on 682550 images from the 5 classes "Drawing" (39026), "Hentai" (28134), "Neutral" (369507), "Porn" (207969) & "Sexy" (37914).
# ... we created a manually inspected test set that consists of 4900 samples, that contains images & their captions.

# Run prediction
def maybe_nsfw(pil_image):
    # Run prediction
    imm = tensorflow.image.resize(np.array(pil_image)[:, :, :3], (260, 260))
    imm = (imm / 255) * 2 - 1
    pred = model(tensorflow.expand_dims(imm, 0)).numpy()
    print(tensorflow.math.softmax(pred[0]).numpy())
    if all([i < .25 for i in tensorflow.math.softmax(pred[0]).numpy()[[1, 3]]]):
        return False
    return True

# pre-initializing prediction
maybe_nsfw(Image. new("RGB", (260, 260), 255))
model.load_weights('nsfweffnetv2-b02-3epochs.h5', by_name=True, )










