import tensorflow as tf
from tensorflow import keras
# import keras
from data_keras import label_acc_score,keras_data
import numpy as np
# keras.backend.set_session(tf.Session(config=tf.ConfigProto(device_count={'cpu':3})))
# keras.backend.set_session(tf.Session(config=tf.ConfigProto(intra_op_‌​parallelism_threads=‌1)))
interpreter=tf.lite.Interpreter('converted_model.tflite')
interpreter.allocate_tensors()

input_detail=interpreter.get_input_details()
output_detail = interpreter.get_output_details()

# a,b=next(iter(keras_data(image_set='test')))
for a,b in iter(keras_data(image_set='test',batch_size=1)):
    # interpreter.set_tensor(input_detail[0]['index'],a[0][None,:])
    interpreter.set_tensor(input_detail[0]['index'],a)
    interpreter.invoke()
    output=interpreter.get_tensor(output_detail[0]['index'])
    #