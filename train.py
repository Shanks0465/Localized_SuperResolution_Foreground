import tensorflow as tf
from skimage.transform import resize, rescale
import cv2
from matplotlib import image
from skimage import img_as_ubyte
import numpy as np
import random
import time
import glob
from model import build_unet
import matplotlib.pyplot as plt

# Model Configurations
input_shape = (256, 256, 3)
scale = 50
epochs = 40
steps = 20
optimizer = "adam"
loss = "mse"

# List Folder Images


def folder_images(path):
    image_list = glob.glob(path)
    return image_list

# Batch Generator for Dataset Batch Resolution Pairs


def BatchDataGenerator(rgb_list, input_shape, scale=50, batch_size=5):
    while True:
        idx = random.sample(range(0, len(rgb_list)), batch_size)
        X_train = []
        y_train = []
        for i in idx:
            rgb = image.imread(rgb_list[i])
            rgb = resize(rgb, (input_shape[0], input_shape[1]))
            width = int(rgb.shape[1] * scale / 100)
            height = int(rgb.shape[0] * scale / 100)
            dsize = (width, height)
            output = cv2.resize(rgb, dsize)
            output = cv2.resize(output, (input_shape[0], input_shape[1]))
            rgb = np.asarray(img_as_ubyte(rgb))
            output = np.asarray(img_as_ubyte(output))
            X_train.append(output)
            y_train.append(rgb)
        yield np.asarray(X_train), np.asarray(y_train)

# Timer for Training


def timer(start, end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
# Function for Plotting Loss Curve


def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')


if __name__ == "__main__":
    # Initialize Batch Data Generator
    train_hr = folder_images("../../data/DIV2K_train_HR/DIV2K_train_HR/*.png")
    train_gen = BatchDataGenerator(
        train_hr, input_shape=input_shape, scale=scale)

    # Initialize U-Net Model
    model = build_unet(input_shape)
    model.summary()

    # Compile Model and Set Callbacks
    optimizer = "adam"
    loss = "mse"
    model.compile(optimizer=optimizer, loss=loss)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss', factor=0.2, patience=5, min_lr=0.001)
    save_point = tf.keras.callbacks.ModelCheckpoint(
        "models/model_at_ep_{epoch:02d}_{loss:02f}.h5", monitor='loss', verbose=1, mode='min', period=5)
    callbacks = [reduce_lr, save_point]

    start = time.time()
    history = model.fit(train_gen, epochs=epochs,
                        steps_per_epoch=steps, verbose=1, callbacks=callbacks)
    end = time.time()
    print("Training Time: ")
    timer(start, end)
    plot_loss(history)