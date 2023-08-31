import os
import time
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 파일 경로 설정
SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
IMAGE_PATH = "./test/0a2ad9a3-fb69-4656-b494-880e57e7d961.jpg"  # 원하는 이미지 파일 경로

# ESRGAN 모델 불러오기
model = hub.load(SAVED_MODEL_PATH)

# 이미지 전처리
def preprocess_image(image_path):
    hr_image = tf.image.decode_image(tf.io.read_file(image_path))
    
    # 컬러 이미지를 흑백으로 변환
    hr_image = tf.image.rgb_to_grayscale(hr_image)
    
    # 흑백 이미지를 컬러로 변환
    hr_image = tf.image.grayscale_to_rgb(hr_image)
    
    hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
    hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
    hr_image = tf.cast(hr_image, tf.float32)
    return tf.expand_dims(hr_image, 0)

# 이미지 저장
def save_image(image, filename):
    if not isinstance(image, Image.Image):
        image = tf.clip_by_value(image, 0, 255)
        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    image.save(filename)

# 원하는 이미지에 대한 ESRGAN 적용
hr_image = preprocess_image(IMAGE_PATH)
start = time.time()
fake_image = model(hr_image)
fake_image = tf.squeeze(fake_image)
print("Time Taken: %f" % (time.time() - start))

# 개선된 이미지로 변경 (덮어쓰기)
save_image(tf.squeeze(fake_image), filename=IMAGE_PATH)
print("Image processed and saved.")