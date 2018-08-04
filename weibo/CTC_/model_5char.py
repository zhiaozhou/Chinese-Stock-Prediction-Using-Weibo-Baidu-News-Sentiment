# 
# Reference: https://github.com/ypwhs/captcha_break

# 
import keras
import numpy as np
import pandas as pd
np.random.seed(0)
from keras.models import Model,load_model
from keras.layers import Dense, Input, Dropout, LSTM, Activation, AveragePooling2D, Conv2D, MaxPooling2D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from unicodedata import normalize
from keras import regularizers
from keras.applications import inception_resnet_v2
from keras.layers import Flatten
from keras.utils.np_utils import to_categorical
from PIL import Image as Image__
import glob
from sklearn.model_selection import train_test_split
from captcha.image import ImageCaptcha
import random
import string
import cv2
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint

# 
characters = string.digits + string.ascii_uppercase
print(characters)

width, height, n_len, n_class = 170, 80, 5, len(characters)

generator = ImageCaptcha(width=width, height=height)
random_str = ''.join([random.choice(characters) for j in range(5)])
img = generator.generate_image(random_str)

# 
def gen_code(batch_size=32):
    X = np.zeros((batch_size, height*4, width*4, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
    generator = ImageCaptcha(width=width, height=height)
    while True:
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(n_len)])
            img = generator.generate_image(random_str)
            img = Image__.fromarray(cv2.resize(np.asarray(img), (width*4, height*4)))
            X[i] = img
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        yield np.array(X), y

# model
input_ = Input((height*4, width*4, 3))
X = input_
for i in range(5):
    X = Conv2D(32*2**i, (3, 3), activation='relu')(X)
    X = Conv2D(32*2**i, (3, 3), activation='relu')(X)
    X = MaxPooling2D((2, 2))(X)
X = Flatten()(X)
X = Dropout(0.25)(X)
outputs = [Dense(n_class, activation='softmax')(X) for i in range(5)]

model = Model(inputs=[input_], outputs=outputs)
#model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
print('model generated')
# 
parallel_model = multi_gpu_model(model, gpus=8)

parallel_model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
print('multi-gpu model generated')

# define the checkpoint
class MyCbk(keras.callbacks.Callback):
    def __init__(self, model):
        self.model_to_save = model
    def on_epoch_end(self, epoch, logs=None):
        self.model_to_save.save('model_at_epoch_%d.h5' % epoch)
        
cbk = MyCbk(parallel_model)

#filepath = "model.h5"
#checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
#callbacks_list = [checkpoint]

print('start to train')
# 
parallel_model.fit_generator(gen_code(batch_size=32),validation_data=gen_code(batch_size=32), 
                             steps_per_epoch=51200,epochs=5 , validation_steps=1280,callbacks=[cbk])

parallel_model.save('final_model.h5')