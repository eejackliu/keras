import tensorflow as tf

from tensorflow import keras
# import keras
from data_keras import label_acc_score,keras_data
import glob
import math
import numpy as np
import matplotlib.pyplot as plt
# from shufflenet import ShuffleNetV2
import tensorflow.keras.layers as layer
# img_input=keras.layers.Input(shape=(None,None,3),name='main_input')
# # import Input, merge, Conv2D, ZeroPadding2D, UpSampling2D, Dense, concatenate, Conv2DTranspose
# # def Nest_Net(img_rows, img_cols, color_type=1, num_class=1, deep_supervision=False):
# mobile=keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False,input_shape=(None, None, 3),input_tensor=img_input)
# vgg=keras.applications.vgg16.VGG16(weights='imagenet', include_top=False,input_shape=(None, None, 3),input_tensor=img_input)

# vgg_part = keras.models.Model(inputs=vgg.input, outputs=vgg.get_layer('block1_conv2').output)
# mobile_part=keras.models.Model(inputs=mobile.input, outputs=mobile.get_layer('block_14_add').output)
act='relu'

def standard_unit(input_tensor, stage, nb_filter, kernel_size=3):

    x = keras.layers.Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv'+stage+'_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=keras.regularizers.l2(1e-4))(input_tensor)
    # x = Dropout(dropout_rate, name='dp'+stage+'_1')(x)
    x=keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv'+stage+'_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    # x = Dropout(dropout_rate, name='dp'+stage+'_2')(x)
    x=keras.layers.BatchNormalization()(x)
    return x
def final_unit(input_tensor, stage, nb_filter, kernel_size=3):
    x = keras.layers.Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, kernel_initializer = 'he_normal', padding='same', kernel_regularizer=keras.regularizers.l2(1e-4))(input_tensor)
    x=keras.layers.BatchNormalization()(x)
    x=keras.layers.Conv2D(1,(1,1),activation=act,kernel_initializer = 'he_normal', name='conv'+stage+'')(x)
    return x
def duplicate_last_col(tensor):
    return tf.concat((tensor, tf.expand_dims(tensor[:, :, -1, ...], 2)), axis=2)
def nest():
    # l0_0=keras.models.Model(inputs=img_input, outputs=vgg.get_layer('block1_conv2').output)
    # x=keras.layers.Conv2D(3,(1,1),activation='relu',name='changesize',padding='same')(l0_0)
    # l1_0=keras.models.Model(inputs=x, outputs=mobile.get_layer('Conv1_relu').output)  #32
    # l2_0=keras.models.Model(inputs=img_input, outputs=mobile.get_layer('block_2_add').output) #24
    # l3_0=keras.models.Model(inputs=img_input, outputs=mobile.get_layer('block_5_add').output) #32
    # l4_0=keras.models.Model(inputs=img_input, outputs=mobile.get_layer('block_12_add').output)#96
    # midd=keras.models.Model(inputs=img_input, outputs=mobile.get_layer('block_14_add').output)#160
    img_input=keras.layers.Input(shape=(None,None,3),name='main_input')
    mobile=keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet',alpha=0.5, include_top=False,input_shape=(None, None, 3))

    # vgg=keras.applications.vgg16.VGG16(weights='imagenet', include_top=False,input_shape=(None, None, 3),input_tensor=img_input)



    vgg=keras.applications.vgg16.VGG16(weights='imagenet',include_top=False,input_shape=(None, None, 3))
    v=keras.models.Model(inputs=vgg.input,outputs=vgg.get_layer('block1_conv2').output)
    l0_0=v(img_input)
    x=keras.layers.Conv2D(3,(1,1),activation='relu',name='changesize',padding='same')(l0_0)



    x=keras.layers.BatchNormalization()(x)
    # ml1_0=keras.models.Model(inputs=mobile.input, outputs=mobile.get_layer('Conv1_relu').output)  #32
    # ml2_0=keras.models.Model(inputs=mobile.input, outputs=mobile.get_layer('block_2_add').output) #24
    # ml3_0=keras.models.Model(inputs=mobile.input, outputs=mobile.get_layer('block_5_add').output) #32
    # ml4_0=keras.models.Model(inputs=mobile.input, outputs=mobile.get_layer('block_12_add').output)#96
    # mmidd=keras.models.Model(inputs=mobile.input, outputs=mobile.get_layer('block_14_add').output)#160
    m=keras.models.Model(inputs=mobile.input,outputs=[mobile.get_layer('Conv1_relu').output,
                                                      mobile.get_layer('block_2_add').output,
                                                      mobile.get_layer('block_5_add').output,
                                                      mobile.get_layer('block_12_add').output,
                                                      mobile.get_layer('block_14_add').output])

    l1_0,l2_0,l3_0,l4_0,midd=m(x) # mobilenet in torch midd will get 10,8 otherwise mobile in keras will get 10,7

    up4_1=keras.layers.Conv2DTranspose(48,(2,2),strides=(2,2),name='l4_1',padding='same')(midd)
    # up4_1=keras.layers.Cropping2D(([0,0],[0,1]))(up4_1)
    up4_1=keras.layers.Lambda(lambda t:duplicate_last_col(t))(up4_1)
    # up4_1=keras.layers.Lambda(duplicate_last_col)(up4_1)
    # up4_1 = keras.layers.ZeroPadding2D(([0, 0], [0, 1]))(up4_1)
    up4_1=keras.layers.concatenate([up4_1,l4_0],axis=3)

    l4_1=standard_unit(up4_1,'l4',48)
    up3_1=keras.layers.Conv2DTranspose(16,(2,2),strides=(2,2),name='l3_1',padding='same')(l4_0)
    up3_1=keras.layers.concatenate([up3_1,l3_0],axis=3)
    l3_1=standard_unit(up3_1,'l3_1',16)
    up3_2=keras.layers.Conv2DTranspose(16,(2,2),strides=(2,2),name='l3_2',padding='same')(l4_1)
    up3_2=keras.layers.concatenate([l3_0,l3_1,up3_2],axis=3)
    l3_2=standard_unit(up3_2,'l3_2',16)
    up2_1=keras.layers.Conv2DTranspose(16,(2,2),strides=(2,2),name='l2_1',padding='same')(l3_0)
    up2_1=keras.layers.concatenate([up2_1,l2_0],axis=3)
    l2_1=standard_unit(up2_1,'l2_1',16)
    up2_2=keras.layers.Conv2DTranspose(16,(2,2),strides=(2,2),name='l2_2',padding='same')(l3_1)
    up2_2=keras.layers.concatenate([up2_2,l2_0,l2_1],axis=3)
    l2_2=standard_unit(up2_2,'l2_2',16)
    up2_3=keras.layers.Conv2DTranspose(16,(2,2),strides=(2,2),name='l2_3',padding='same')(l3_2)
    up2_3=keras.layers.concatenate([up2_3,l2_0,l2_1,l2_2],axis=3)
    l2_3=standard_unit(up2_3,'l2_3',16)
    up1_1=keras.layers.Conv2DTranspose(16,(2,2),strides=(2,2),name='l1_1',padding='same')(l2_0)
    up1_1=keras.layers.concatenate([up1_1,l1_0],axis=3)
    l1_1=standard_unit(up1_1,'l1_1',16)
    up1_2=keras.layers.Conv2DTranspose(16,(2,2),strides=(2,2),name='l1_2',padding='same')(l2_1)
    up1_2=keras.layers.concatenate([up1_2,l1_0,l1_1],axis=3)
    l1_2=standard_unit(up1_2,'l1_2',16)
    up1_3=keras.layers.Conv2DTranspose(16,(2,2),strides=(2,2),name='l1_3',padding='same')(l2_2)
    up1_3=keras.layers.concatenate([up1_3,l1_0,l1_1,l1_2],axis=3)
    l1_3=standard_unit(up1_3,'l1_3',16)
    up1_4=keras.layers.Conv2DTranspose(16,(2,2),strides=(2,2),name='l1_4',padding='same')(l2_3)
    up1_4=keras.layers.concatenate([up1_4,l1_0,l1_1,l1_2,l1_3],axis=3)
    l1_4=standard_unit(up1_4,'l1_4',16)

    up0_1=keras.layers.Conv2DTranspose(32,(2,2),strides=(2,2),name='l0_1',padding='same')(l1_0)
    up0_1=keras.layers.concatenate([up0_1,l0_0],axis=3)
    l0_1=final_unit(up0_1,'l0_1',1)
    up0_2=keras.layers.Conv2DTranspose(16,(2,2),strides=(2,2),name='l0_2',padding='same')(l1_1)
    up0_2=keras.layers.concatenate([up0_2,l0_0,l0_1],axis=3)
    l0_2=final_unit(up0_2,'l0_2',1)
    up0_3=keras.layers.Conv2DTranspose(16,(2,2),strides=(2,2),name='l0_3',padding='same')(l1_2)
    up0_3=keras.layers.concatenate([up0_3,l0_0,l0_1,l0_2],axis=3)
    l0_3=final_unit(up0_3,'l0_3',1)
    up0_4=keras.layers.Conv2DTranspose(16,(2,2),strides=(2,2),name='l0_4',padding='same')(l1_3)
    up0_4=keras.layers.concatenate([up0_4,l0_0,l0_1,l0_2,l0_3],axis=3)
    l0_4=final_unit(up0_4,'l0_4',1)
    up0_5=keras.layers.Conv2DTranspose(16,(2,2),strides=(2,2),name='l0_5',padding='same')(l1_4)
    up0_5=keras.layers.concatenate([up0_5,l0_0,l0_1,l0_2,l0_3,l0_4],axis=3)
    l0_5=final_unit(up0_5,'l0_5',1)







    model=keras.models.Model(inputs=img_input,outputs=[l0_1,l0_2,l0_3,l0_4,l0_5])
    return model
def test():
    img_input=keras.layers.Input(shape=(None,None,3),name='main_input')
    x = keras.layers.Conv2D(12, (3, 3),activation=act, name='conv', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=keras.regularizers.l2(1e-4))(img_input)
    return keras.models.Model(inputs=img_input,outputs=x)
def diceloss(y_true,y_pred):

    numerator=2*keras.backend.sum(y_true*y_pred)+0.0001
    denominator=keras.backend.sum(y_true**2)+keras.backend.sum(y_pred**2)+0.0001
    return 1-numerator/denominator/2
def iou(y_true,y_pred):
    y_true=(y_true>0.5)
    y_true=keras.backend.cast(y_true, dtype='float32')
    return keras.backend.sum(y_true*y_pred)/(keras.backend.sum(y_pred),keras.backend.sum(y_true)-keras.backend.sum(y_true*y_pred)+0.0001)
def picture(pre_numpy,img_numpy,mask_numpy):
    #pre_numpy has shape num,height,width,channel
    voc_colormap=np.array([[0, 0, 0], [245,222,179]])
    num=len(img_numpy)
    target=(pre_numpy>0.5).squeeze().astype(int)
    mean,std=np.array((0.485, 0.456, 0.406)),np.array((0.229, 0.224, 0.225))
    img=img_numpy*std+mean
    img=img.squeeze()
    mask=voc_colormap[mask_numpy.squeeze().astype(int)]
    tar=voc_colormap[target]/255.
    tmp=np.concatenate((img,tar,mask),axis=0)
    for i,j in enumerate(tmp,1):
        plt.subplot(3,num,i)
        plt.imshow(j)
    plt.show()

batch_size=2
train_data=keras_data(batch_size=batch_size)
val_data=keras_data(image_set='test')
optim=keras.optimizers.Adam()
steps = math.ceil(len(glob.glob('data/mask/'+ '*.png')) / batch_size)
losses={
    'convl0_1':diceloss,
    'convl0_2':diceloss,
    'convl0_3':diceloss,
    'convl0_4':diceloss,
    'convl0_5':diceloss,

}
metrics={
    'convl0_1':iou,
    'convl0_2':iou,
    'convl0_3':iou,
    'convl0_4':iou,
    'convl0_5':iou,
}
model=nest()
model.compile(optimizer=optim,loss=[diceloss,diceloss,diceloss,diceloss,diceloss,] ,loss_weights=[0.2,0.2,0.2,0.2,0.2])
model.fit_generator(train_data,steps_per_epoch=steps,epochs=20,use_multiprocessing=True, verbose=2,workers=2)
keras.models.save_model(model, 'unet_.h5',include_optimizer=False)
# # #
#
def test(model):
    img=[]
    pred=[]
    mask=[]
    l1_list=[]
    l2_list=[]
    l3_list=[]
    l4_list=[]
    l5_list=[]
    for image,mask_img in val_data:
        l1,l2,l3,l4,\
        l5=model.predict(image)
        label=(l5>0.5).squeeze().astype(int)
        l1_list.append(l1)
        l2_list.append(l2)
        l3_list.append(l3)
        l4_list.append(l4)
        l5_list.append(l5)
        pred.append(label)
        img.append(image)
        mask.append(mask_img)
    return np.concatenate(img,axis=0),np.concatenate(pred,axis=0),np.concatenate(mask,axis=0),[l1_list,l2_list,l3_list,l4_list,l5_list]

model=keras.models.load_model('unet_.h5')
img,pred,mask,l=test(model)
ap,iou,hist,tmp=label_acc_score(mask,pred,2)
# converter=tf.lite.TFLiteConverter.from_keras_model_file('unet.h5',input_shapes={'main_input':[1,320,240,3]})
# tflite_model = converter.convert()
# open("unet_second.tflite", "wb").write(tflite_model)
#


















# https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter
# https://stackoverflow.com/questions/50581883/how-do-i-export-a-tensorflow-model-as-a-tflite-file
# https://stackoverflow.com/questions/52400043/how-to-get-toco-to-work-with-shape-none-24-24-3?rq=1
# https://medium.com/@xianbao.qian/convert-keras-model-to-tflite-e2bdf28ee2d2


# Step 1: Create a Keras model (which you might already have)
# model = create_my_keras_model()
# model.compile(loss, optimizer)
# model.fit_generator(dataset)
# Step 2: Convert inference model
# output_names = [node.op.name for node in model.outputs]
# sess = tf.keras.backend.get_session()
# frozen_def = tf.graph_util.convert_variables_to_constants(
#     sess, sess.graph_def, output_names)
# Step 3: Create tflite model
# # You might want to do some hack to add port number to
# # output_names here, e.g. convert add to add:0
# tflite_model = tf.contrib.lite.toco_convert(
#     frozen_def, [inputs], output_names)
# with tf.gfile.GFile(tflite_graph, 'wb') as f:
#     f.write(tflite_model)




#https://github.com/qubvel/segmentation_models
#if step_per_encoch=0 or train data give 0 image will casuse attribute error progbarlogger object has no attribute log values
# mask sure this is ok

# >>> b={layer.name:num for num ,layer in enumerate( mobile.layers)}
# >>> b[block_14_add]
#
# import tensorflow as tf
# from keras.layers import Lambda, Input
# from keras.models import Model
# import numpy as np
#
# def duplicate_last_row(tensor):
#     return tf.concat((tensor, tf.expand_dims(tensor[:, -1, ...], 1)), axis=1)
#
# def duplicate_last_col(tensor):
#     return tf.concat((tensor, tf.expand_dims(tensor[:, :, -1, ...], 2)), axis=2)
#
# # --------------
# # Demonstrating with TF:
#
# x = tf.convert_to_tensor([[[1, 2, 3], [4, 5, 6]],
#                           [[10, 20, 30], [40, 50, 60]]])
#
# x = duplicate_last_row(duplicate_last_col(x))
# with tf.Session() as sess:
#     print(sess.run(x))
# # [[[ 1  2  3  3]
# #   [ 4  5  6  6]
# #   [ 4  5  6  6]]
# #
# #  [[10 20 30 30]
# #   [40 50 60 60]
# #   [40 50 60 60]]]
#
#
# # --------------
# # Using as a Keras Layer:
#
# inputs = Input(shape=(5, 5, 3))
# padded = Lambda(lambda t: duplicate_last_row(duplicate_last_col(t)))(inputs)
#
# model = Model(inputs=inputs, outputs=padded)
# model.compile(optimizer="adam", loss='mse', metrics=['mse'])
# batch = np.random.rand(2, 5, 5, 3)
# x = model.predict(batch, batch_size=2)
# print(x.shape)

#
#
# CameraActivity
# package com.example.android.tflitecamerademo;
#
# package com.example.android.tflitecamerademo;
#
# import android.app.Activity;
# import android.content.Context;
# import android.content.res.AssetManager;
# import android.graphics.Bitmap;
# import android.graphics.BitmapFactory;
# import android.os.Bundle;
# import android.util.Log;
# import android.widget.ImageView;
# import android.widget.TextView;
#
# import java.io.IOException;
# import java.io.InputStream;
#
# /** Main {@code Activity} class for the Camera app. */
# public class CameraActivity extends Activity {
#   private AutoFitTextureView textureView;
#   private ImageClassifier classifier;
#   private TextView textView;
#   private static final String TAG = "TfLiteCameraDemo";
#   @Override
#   protected void onCreate(Bundle savedInstanceState) {
#     super.onCreate(savedInstanceState);
#     setContentView(R.layout.camera);
#     ImageView imageView=findViewById(R.id.texture);
#     textView = (TextView) findViewById(R.id.text);
#     Bitmap bitmap=getBitmapFromAsset(this,"mouse.png");
#     imageView.setImageBitmap(bitmap);
#     try {
#       // create either a new ImageClassifierQuantizedMobileNet or an ImageClassifierFloatInception
#       classifier = new ImageClassifierQuantizedMobileNet(this);
# //      classifier = new ImageClassifierFloatInception(this);
# //      classifier= new  ImageClassifierResnet(this);
# //      classifier=new ImageClassifiersqueeze(this);
#     } catch (IOException e) {
#       Log.e(TAG, "Failed to initialize an image classifier.", e);
#     }
# //    for (int i=0;i<3;i++) {
# //      classifyFrame();
# //     Log.e(TAG,"test");
# //    }
#     classifyFrame();
#
#
# //    if (null == savedInstanceState) {
# //      getFragmentManager()
# //          .beginTransaction()
# //          .replace(R.id.container, Camera2BasicFragment.newInstance())
# //          .commit();
# //    }
#   }
#   private void classifyFrame() {
# //    if (classifier == null || getActivity() == null || cameraDevice == null)
#     if(classifier==null)
#     {
#       showToast("Uninitialized Classifier or invalid context.");
#       return;
#     }
#
# //    Bitmap bitmap = textureView.getBitmap(classifier.getImageSizeX(), classifier.getImageSizeY());
#     Bitmap bitmap=getBitmapFromAsset(this,"mouse.png");
#     bitmap=Bitmap.createScaledBitmap(bitmap,classifier.getImageSizeX(),classifier.getImageSizeY(),false);
#     String textToShow = classifier.classifyFrame(bitmap);
# //    bitmap.recycle();
#     showToast(textToShow);
#   }
#   private void showToast(final String text) {
#
#                   textView.setText(text);
#
#
#   }
#   public static Bitmap getBitmapFromAsset(Context context, String filePath) {
#     AssetManager assetManager = context.getAssets();
#
#     InputStream istr;
#     Bitmap bitmap = null;
#     try {
#       istr = assetManager.open(filePath);
#       bitmap = BitmapFactory.decodeStream(istr);
#     } catch (IOException e) {
#       // handle exception
#     }
#
#     return bitmap;
#   }
# }
#
#
#
#
#
#
#
# ImageClassifierResnet
#
# package com.example.android.tflitecamerademo;
#
# import android.app.Activity;
#
# import java.io.IOException;
#
# public class ImageClassifierResnet extends ImageClassifier {
#
#     /**
#      * The inception net requires additional normalization of the used input.
#      */
#     private static final int IMAGE_MEAN = 128;
#     private static final float IMAGE_STD = 128.0f;
#
#     /**
#      * An array to hold inference results, to be feed into Tensorflow Lite as outputs.
#      * This isn't part of the super class, because we need a primitive array here.
#      */
#     private float[][] labelProbArray = null;
#
#     /**
#      * Initializes an {@code ImageClassifier}.
#      *
#      * @param activity
#      */
#     ImageClassifierResnet(Activity activity) throws IOException {
#         super(activity);
#         labelProbArray = new float[1][getNumLabels()];
#     }
#
#     @Override
#     protected String getModelPath() {
#         // you can download this file from
#         // https://storage.googleapis.com/download.tensorflow.org/models/tflite/inception_v3_slim_2016_android_2017_11_10.zip
#         return "squeezenet.tflite";
# //    return "inceptionv3_non_slim_2015.tflite";
#     }
#
#     @Override
#     protected String getLabelPath() {
#
#         return "squeeze.txt";
# //      return "imagenet_2015_label_strings.txt";
#
#     }
#
#     @Override
#     protected int getImageSizeX() {
#         return 224;
#     }
#
#     @Override
#     protected int getImageSizeY() {
#         return 224;
#     }
#
#     @Override
#     protected int getNumBytesPerChannel() {
#         // a 32bit float value requires 4 bytes
#         return 4;
#     }
#
#     @Override
#     protected void addPixelValue(int pixelValue) {
# //        imgData.putFloat(((pixelValue >> 16) & 0xFF));
# //        imgData.putFloat (((pixelValue >> 8) & 0xFF));
# //        imgData.putFloat ((pixelValue & 0xFF));
#         imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
#         imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
#         imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
#     }
#
#     @Override
#     protected float getProbability(int labelIndex) {
#         return labelProbArray[0][labelIndex];
#     }
#
#     @Override
#     protected void setProbability(int labelIndex, Number value) {
#         labelProbArray[0][labelIndex] = value.floatValue();
#     }
#
#     @Override
#     protected float getNormalizedProbability(int labelIndex) {
#         // TODO the following value isn't in [0,1] yet, but may be greater. Why?
#         return getProbability(labelIndex);
#     }
#
#     @Override
#     protected void runInference() {
#         tflite.run(imgData, labelProbArray);
#     }
# }
