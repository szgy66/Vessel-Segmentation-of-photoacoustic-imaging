# Before you run this code, please change the image path, in our experiment, we have three
# file named 'Aug-imgae', 'Aug-label', 'test' which all in 'content3' file
import numpy as np
import os
import matplotlib.pyplot as plt
from model import *
from keras.layers import Input, add, Multiply, Activation
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from datetime import datetime

time1= datetime.now()
file_path = "./content3/"
xpath = os.path.join(file_path, "Aug-image")
ypath = os.path.join(file_path, "Aug-label")

data = os.listdir(xpath)
label = os.listdir(ypath)

x_train = np.zeros((len(data), 256, 256))
y_train = np.zeros((len(data), 256, 256))

for idx, imname in enumerate(data):
    x_train[idx, :, :] = plt.imread(os.path.join(xpath, imname))

for idx, imname in enumerate(label):
    y_train[idx, :, :] = plt.imread(os.path.join(ypath, imname))
x_train = np.expand_dims(x_train, axis=3)
y_train = np.expand_dims(y_train, axis=3)

# val = np.array([0.5])
# var = K.variable(value=val)
inpt = Input(shape=(256, 256, 1))
output1 = FCN_8S(1, inpt)
# output2 = unet(inpt)
# output = add([output1, output2])
output = Activation('sigmoid')(output1)
model = Model(inpt, output)
adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
result = model.fit(x_train, y_train, batch_size=4, epochs=50, validation_split=0.2)
plt.plot(np.arange(len(result.history['acc'])), result.history['acc'], label='training')
plt.plot(np.arange(len(result.history['val_acc'])), result.history['val_acc'], label='validation')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
model.save('hybrid-content3.h5')
time2 = datetime.now()
print('Total time is %ds'%((time2-time1).seconds))

for i in range(4):
    output = model.predict(np.expand_dims(x_train[3+i, :, :, :], axis=0))
    plt.figure()
    plt.subplot(131)
    plt.imshow(np.squeeze(output), cmap="gray")
    plt.subplot(132)
    plt.imshow(y_train[3+i, :, :, 0], cmap="gray")
    plt.subplot(133)
    plt.imshow(x_train[3+i, :, :, 0], cmap="gray")
    plt.show()
