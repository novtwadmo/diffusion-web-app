from tensorflow import keras
import keras_cv
import matplotlib.pyplot as plt

# 從保存的路徑加載 diffusion model
diffusion_model = keras.models.load_model('saved_model/diffusion_model')

def generate_image(prompt, model):
    generated_img = model.text_to_image(prompt)  # 假設模型有 text_to_image 方法
    plt.imshow(generated_img[0])
    plt.axis('off')
    plt.show()

# 測試生成圖像
generate_image("A beautiful painting of a landscape", diffusion_model)