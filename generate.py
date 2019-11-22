# coding=gbk
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import ssim
import copy
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model = keras.models.load_model("model.h5")

def generate(images, shape):
    generate_images = []
    # ��һ��
    count = 0
    ssim_sum = 0.0
    start = time.time()
    i = 0
    #����
    for img in images: 
        print("����Ŀ�꣺", i)
        i = i + 1
        now_start = time.time()
        attack_image = random_attack(img)
        if not get_label(attack_image) == get_label(img):
            count += 1
            ssim_img = ssim.calculate(attack_image, img)
            ssim_sum += ssim_img
            now_end = time.time()
            print("�������: ", ssim_img, "��ʱ: ", now_end - now_start, "ƽ�����: ", ssim_sum/count)
            generate_images.append(attack_image)
        else: #����ʧ��
            generate_images.append(security_attack(img))
    end = time.time()
    print("�ܺ�ʱ: ", end - start)
    print("�ɹ�������", count, ", ������", ssim_sum/count)
    
    #����Ϊ.npy
    generate_images = np.array(generate_images)
    np.save("./attack_data/attack_data.npy", generate_images)
    print("�ɹ����� attack_data.npy")
    return generate_images

# ���ر�ǩ
def get_label(image):
    prediction = model.predict(np.expand_dims(image, 0))[0]
    lable = np.argmax(prediction)
    return lable

# �������ص�
def change_pixel(image, num=1):
    new_image = copy.deepcopy(image)
    (row, col) = new_image.shape
    # ����޸����ص�  
    for i in range(num):
        x = random.randint(0, row-1)
        y = random.randint(0, col-1)
        new_image[x][y] = random.randint(0, 255)
    return new_image

def random_attack(image, times=500):
    # ѭ������
    label = get_label(image)

    final_image = change_pixel(image)
    max_ssim = ssim.calculate(final_image, image)
    for i in range(5): 
        new_image = change_pixel(image, i + 1)
        index = 0
        while(index < times and (ssim.calculate(new_image, image)<0.9 or label == get_label(new_image))):
            new_image = change_pixel(image, i + 1)
            index = index +  1
            #���������Ƶ�ͼƬ
            if (not label == get_label(new_image)) and max_ssim < ssim.calculate(new_image, image):
                final_image = copy.deepcopy(new_image)
    return final_image

#���׹���
def security_attack(image):
    new_image = copy.deepcopy(image)
    (row, col) = new_image.shape
    # ����������
    for i in range(5):
        x=np.random.randint(0,row-1)
        y=np.random.randint(0,col-1)
        new_image[x][y]=255
    return new_image

# ��������, ����Ϊ.npy
def load_data():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images,
                                   test_labels) = fashion_mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    np.save("./test_data/test_data.npy", train_images[:1000])
    print("�ɹ����� test_data.npy")
    return train_images, train_labels, test_images, test_labels

if __name__ == '__main__':

    # ��������
    train_images, train_labels, test_images, test_labels = load_data()

     # ��ʼ����
    print("----������ʼ----")
    images = generate(test_images[0:1000], (1000, 28, 28, 1))
    print("----��������----")
     

