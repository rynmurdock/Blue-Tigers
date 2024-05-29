
# TODO required tensorflow==2.14 for me
# weights from https://github.com/LAION-AI/safety-pipeline/tree/main
from PIL import Image
import tensorflow_hub as hub
import tensorflow
import numpy as np
import sys
sys.path.append('/home/ryn_mote/Misc/generative_recommender/gradio_video/automl/efficientnetv2/')
import tensorflow as tf
from tensorflow.keras import mixed_precision

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(
        physical_devices[0], True
    )

model = tf.keras.models.load_model('nsfweffnetv2-b02-3epochs.h5',custom_objects={"KerasLayer":hub.KerasLayer})
# "The image classifier had been trained on 682550 images from the 5 classes "Drawing" (39026), "Hentai" (28134), "Neutral" (369507), "Porn" (207969) & "Sexy" (37914).
# ... we created a manually inspected test set that consists of 4900 samples, that contains images & their captions."

# Run prediction
def maybe_nsfw(pil_image):
    # Run prediction
    imm = tensorflow.image.resize(np.array(pil_image)[:, :, :3], (260, 260))
    imm = (imm / 255)
    pred = model(tensorflow.expand_dims(imm, 0)).numpy()
    probs = tensorflow.math.softmax(pred[0]).numpy()
    print(probs)
    if all([i < .3 for i in probs[[1, 3, 4]]]):
        return False
    return True

# pre-initializing prediction
maybe_nsfw(Image. new("RGB", (260, 260), 255))
model.load_weights('nsfweffnetv2-b02-3epochs.h5', by_name=True, )









