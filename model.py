'''
训练模型, 参考CSDN博客,
《TensorFlow之tf.keras的基础分类》
https://blog.csdn.net/qq_20989105/article/details/82760815
'''

import tensorflow as tf
from tensorflow import keras
# 忽略tensorflow 版本 报错
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def create_model():
    # 设置图层
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    # 编译模型
    model.compile(optimizer=tf.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # 训练模型
    # (epochs, accuracy) = (5, 0.8734) 
    model.fit(train_images, train_labels, epochs=100)
    # 评估准确性
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)
    model.save('model.h5', include_optimizer=True)
    return model

def load_data():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images,
                                   test_labels) = fashion_mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    return train_images, train_labels, test_images, test_labels

if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = load_data()
    model = create_model()
    print("successfully")

    # 结构相似性
    #ssim = tf.image.ssim(im1, im2, max_val=255)
