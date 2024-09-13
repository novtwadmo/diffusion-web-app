from flask import Flask, render_template, request, redirect, url_for
import keras_cv
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# 初始化 Stable Diffusion 模型
model = keras_cv.models.StableDiffusion(img_width=512, img_height=512, jit_compile=False)

def plot_images(images):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return image_base64

@app.route('/', methods=['GET', 'POST'])
def index():
    generated_image = None
    if request.method == 'POST':
        prompt = request.form['prompt']
        # 使用模型生成圖像
        images = model.text_to_image(prompt, batch_size=1)
        generated_image = plot_images(images)

    return render_template('index.html', image=generated_image)

if __name__ == '__main__':
    app.run(debug=True)
