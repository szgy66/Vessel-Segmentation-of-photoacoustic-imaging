import numpy as np
import os
import matplotlib.pyplot as plt
from keras.models import load_model
from datetime import datetime

a = datetime.now()
file_path = "D:/Desktop/result"
xpath = os.path.join(file_path, "image")
data = os.listdir(xpath)

x_train = np.zeros((len(data), 256, 256))

for idx, imname in enumerate(data):
    x_train[idx, :, :] = plt.imread(os.path.join(xpath, imname))

x_train = np.expand_dims(x_train, axis=3)


model = load_model('hybrid-content3')
#
predict = model.predict(x_train, verbose=1)
print(predict.shape)
for i in range(len(predict)):
    # cv2.imwrite('result/%d.png'%i, predict[i, :, :, :])
    plt.imsave('D:/Desktop/result/unet/%d.png'%i, predict[i, :, :, 0])
b = datetime.now()
print('共花费%ds'%(b-a).seconds)
