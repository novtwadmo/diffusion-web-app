import keras_cv
import tensorflow as tf
from tensorflow import keras
import os

# 設定資料集路徑
DATA_DIR = 'wikiart'

# 加載並預處理資料集
def preprocess_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [512, 512])
    img = img / 255.0
    return img

def load_dataset(data_dir, batch_size=32):
    image_files = tf.data.Dataset.list_files(os.path.join(data_dir, '*/*.jpg'))
    dataset = image_files.map(preprocess_image)
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

# 加載資料集
train_ds = load_dataset(DATA_DIR)

# 設定 Keras_CV Diffusion Model
diffusion_model = keras_cv.models.StableDiffusion(
    img_size=(512, 512),
    text_encoder=keras_cv.models.stable_diffusion.TextEncoder(),
    image_encoder=keras_cv.models.stable_diffusion.ImageEncoder(),
    diffusion_model=keras_cv.models.stable_diffusion.DiffusionModel(),
    noise_scheduler=keras_cv.models.stable_diffusion.NoiseScheduler()
)

# 訓練模型
diffusion_model.fit(train_ds, epochs=10)
diffusion_model.save('saved_model/diffusion_model')