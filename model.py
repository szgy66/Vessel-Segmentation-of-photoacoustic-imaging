from keras.layers import Input, Conv2D, MaxPooling2D, add, Dropout, Conv2DTranspose, \
    UpSampling2D, concatenate, Activation, Multiply, Lambda
import numpy as np
import tensorflow as tf
from keras import backend as K
inpt = Input(shape=(256, 256, 1))

# val1 = np.zeros((256, 256))
# val2 = np.zeros((256, 256))
# for i in range(256):
#     for j in range(256):
#         val1[i][j] = 0.5
#         val2[i][j] = 0.5
val1 = np.array([0.5])
val2 = np.array([0.5])
v1 = K.variable(value=val1)
v2 = K.variable(value=val2)

# def mul(x):
#     tf.keras.layers.multiply(x, np.array([1], dtype='float32'))

def FCN_8S(nClasses, inputs=inpt):
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block1_conv1')(inputs)
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu',  name='block1_conv2')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='block1_pool')(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block2_conv1')(pool1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', name='block2_conv2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='block2_pool')(conv2)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block3_conv1')(pool2)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block3_conv2')(conv3)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block3_conv3')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), name='block3_pool')(conv3)
    score_pool3 = Conv2D(filters=3, kernel_size=(3, 3), padding='same',
                         activation='relu', name='score_pool3')(pool3)
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block4_conv1')(pool3)
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block4_conv2')(conv4)
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block4_conv3')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), name='block4_pool')(conv4)
    score_pool4 = Conv2D(filters=3, kernel_size=(3, 3), padding='same',
                         activation='relu', name='score_pool4')(pool4)
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block5_conv1')(pool4)
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block5_conv2')(conv5)
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block5_conv3')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2), name='block5_pool')(conv5)
    fc6 = Conv2D(filters=1024, kernel_size=(1, 1), padding='same', activation='relu', name='fc6')(pool5)
    fc6 = Dropout(0.3, name='dropout_3')(fc6)
    fc7 = Conv2D(filters=1024, kernel_size=(1, 1), padding='same', activation='relu', name='fc7')(fc6)
    fc7 = Dropout(0.3, name='dropour_2')(fc7)
    score_fr = Conv2D(filters=nClasses, kernel_size=(1, 1), padding='same',
                      activation='relu', name='score_fr')(fc7)
    score2 = Conv2DTranspose(filters=nClasses, kernel_size=(2, 2), strides=(2, 2),
                             padding="valid", activation=None, name="score2")(score_fr)
    add1 = add(inputs=[score2, score_pool4], name="add_3")
    score4 = Conv2DTranspose(filters=nClasses, kernel_size=(2, 2), strides=(2, 2),
                             padding="valid", activation=None, name="score4")(add1)
    add2 = add(inputs=[score4, score_pool3], name="add_2")
    result_FCN= Conv2DTranspose(filters=nClasses, kernel_size=(8, 8), strides=(8, 8),
                                padding="valid", name="UpSample")(add2)
    # result_FCN = Activation('sigmoid')(result_FCN)
    # result_FCN = Lambda(mul)(result_FCN)
    # print(result_FCN)
    return result_FCN

def unet(inputs=inpt):
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    drop5 = Dropout(0.5)(conv5)

    ## Now the decoder starts

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv10 = Conv2D(1, 3, padding='same')(conv9)
    # conv10 = Activation('sigmoid')(conv10)
    # conv10 = conv10*v2
    # print(conv10)
    return conv10