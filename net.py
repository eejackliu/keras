import tensorflow as tf

from tensorflow import keras
from data_keras import label_acc_score,keras_data
import glob
import math
import numpy as np
import tensorflow.keras.layers as layer

# import Input, merge, Conv2D, ZeroPadding2D, UpSampling2D, Dense, concatenate, Conv2DTranspose
# def Nest_Net(img_rows, img_cols, color_type=1, num_class=1, deep_supervision=False):
mobile=keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet', include_top=True)
vgg=keras.applications.vgg16.VGG16(weights='imagenet', include_top=True)

vgg_part = keras.models.Model(inputs=vgg.input, outputs=vgg.get_layer('block1_conv2').output)
mobile_part=keras.models.Model(inputs=mobile.input, outputs=mobile.get_layer('block_14_add').output)
act='relu'

def standard_unit(input_tensor, stage, nb_filter, kernel_size=3):

    x = keras.layers.Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv'+stage+'_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=keras.regularizers.l2(1e-4))(input_tensor)
    # x = Dropout(dropout_rate, name='dp'+stage+'_1')(x)
    x=keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv'+stage+'_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    # x = Dropout(dropout_rate, name='dp'+stage+'_2')(x)
    x=keras.layers.BatchNormalization()(x)
    return x


def nest():
    img_input=keras.layers.Input(shape=(320,240,3),name='main_input')
    l0_0=keras.models.Model(inputs=img_input, outputs=vgg.get_layer('block1_conv2').output)
    x=keras.layers.Conv2D(3,(1,1),activation='relu',name='changesize',padding='same')(l0_0)
    l1_0=keras.models.Model(inputs=x, outputs=mobile.get_layer('Conv1_relu').output)  #32
    l2_0=keras.models.Model(inputs=img_input, outputs=mobile.get_layer('block_2_add').output) #24
    l3_0=keras.models.Model(inputs=img_input, outputs=mobile.get_layer('block_5_add').output) #32
    l4_0=keras.models.Model(inputs=img_input, outputs=mobile.get_layer('block_12_add').output)#96
    midd=keras.models.Model(inputs=img_input, outputs=mobile.get_layer('block_14_add').output)#160
    up4_1=keras.layers.Conv2DTranspose(48,(2,2),(2,2),name='l4_1',padding='same')(midd)
    up4_1=keras.layers.Cropping2D([0,15])(up4_1)
    up4_1=keras.layers.concatenate([up4_1,l4_0],axis=3)
    l4_1=standard_unit(up4_1,'l4',48)
    up3_1=keras.layers.Conv2DTranspose(16,(2,2),(2,2),name='l3_1',padding='same')(l4_0)
    up3_1=keras.layers.concatenate([up3_1,l3_0],axis=3)
    l3_1=standard_unit(up3_1,'l3_1',16)
    up3_2=keras.layers.Conv2DTranspose(16,(2,2),(2,2),name='l3_2',padding='same')(l4_1)
    up3_2=keras.layers.concatenate([l3_0,l3_1,up3_2],axis=3)
    l3_2=standard_unit(up3_2,'l3_2',16)
    up2_1=keras.layers.Conv2DTranspose(16,(2,2),(2,2),name='l2_1',padding='same')(l3_0)
    up2_1=keras.layers.concatenate([up2_1,l2_0],axis=3)
    l2_1=standard_unit(up2_1,'l2_1',16)
    up2_2=keras.layers.Conv2DTranspose(16,(2,2),(2,2),name='l2_2',padding='same')(l3_1)
    up2_2=keras.layers.concatenate([up2_2,l2_0,l2_1],axis=3)
    l2_2=standard_unit(up2_2,'l2_2',16)
    up2_3=keras.layers.Conv2DTranspose(16,(2,2),(2,2),name='l2_3',padding='same')(l3_2)
    up2_3=keras.layers.concatenate([up2_3,l2_0,l2_1,l2_2],axis=3)
    l2_3=standard_unit(up2_3,'l2_3',16)
    up1_1=keras.layers.Conv2DTranspose(16,(2,2),(2,2),name='l1_1',padding='same')(l2_0)
    up1_1=keras.layers.concatenate([up1_1,l1_0],axis=3)
    l1_1=standard_unit(up1_1,'l1_1',16)
    up1_2=keras.layers.Conv2DTranspose(16,(2,2),(2,2),name='l1_2',padding='same')(l2_1)
    up1_2=keras.layers.concatenate([up1_2,l1_0,l1_1],axis=3)
    l1_2=standard_unit(up1_2,'l1_2',16)
    up1_3=keras.layers.Conv2DTranspose(16,(2,2),(2,2),name='l1_3',padding='same')(l2_2)
    up1_3=keras.layers.concatenate([up1_3,l1_0,l1_1,l1_2],axis=3)
    l1_3=standard_unit(up1_3,'l1_3',16)
    up1_4=keras.layers.Conv2DTranspose(16,(2,2),(2,2),name='l1_4',padding='same')(l2_3)
    up1_4=keras.layers.concatenate([up1_4,l1_0,l1_1,l1_2,l1_3],axis=3)
    l1_4=standard_unit(up1_4,'l1_4',16)
    up0_1=keras.layers.Conv2DTranspose(32,(2,2),(2,2),name='l0_1',padding='same')(l1_0)
    up0_1=keras.layers.concatenate([up0_1,l0_0],axis=3)
    l0_1=standard_unit(up0_1,'l0_1',1)
    up0_2=keras.layers.Conv2DTranspose(16,(2,2),(2,2),name='l0_2',padding='same')(l1_1)
    up0_2=keras.layers.concatenate([up0_2,l0_0,l0_1],axis=3)
    l0_2=standard_unit(up0_2,'l0_2',1)
    up0_3=keras.layers.Conv2DTranspose(16,(2,2),(2,2),name='l0_3',padding='same')(l1_2)
    up0_3=keras.layers.concatenate([up0_3,l0_0,l0_1,l0_2],axis=3)
    l0_3=standard_unit(up0_3,'l0_3',1)
    up0_4=keras.layers.Conv2DTranspose(16,(2,2),(2,2),name='l0_4',padding='same')(l1_3)
    up0_4=keras.layers.concatenate([up0_4,l0_0,l0_1,l0_2,l0_3],axis=3)
    l0_4=standard_unit(up0_4,'l0_4',1)
    up0_5=keras.layers.Conv2DTranspose(16,(2,2),(2,2),name='l0_5',padding='same')(l1_4)
    up0_5=keras.layers.concatenate([up0_5,l0_0,l0_1,l0_2,l0_3,l0_4],axis=3)
    l0_5=standard_unit(up0_5,'l0_5',1)

    model=keras.Model(input=img_input,output=[l0_1,l0_2,l0_3,l0_4,l0_5])

    return model
def diceloss(y_true,y_pred):
    numerator=2*keras.backend.sum(y_true*y_pred)+0.0001
    denominator=keras.backend.sum(y_true**2)+keras.backend.sum(y_pred**2)+0.0001
    return numerator/denominator/float(len(y_pred))

model=nest()
batch_size=2
train_data=keras_data(batch_size=2)
optim=keras.optimizers.Adam()
steps = math.ceil(len(glob.glob('data/train/'+ '*.png')) / batch_size)
model.compile(optimizer=optim,loss=diceloss,metrics=['accuracy',label_acc_score])
model.fit_generator(train_data,steps_per_epoch=steps,epochs=20,verbose=1,
                    use_multiprocessing=True, workers=2)
#https://github.com/qubvel/segmentation_models