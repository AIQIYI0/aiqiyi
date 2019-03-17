"""
Adapted from keras example cifar10_cnn.py
Train ResNet-18 on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10.py
"""
from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping,ModelCheckpoint

import numpy as np
import resnet
import pandas as pd
import tensorflow as tf
import os
import keras
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model
from keras.callbacks import TensorBoard,LearningRateScheduler
import random
# def read_pickle_test():
#     dir_test= '/home/w/competition/aiqiyi/data/classify/feats_test.pickle'
#     data_test=pd.read_pickle(dir_test)
#     x_test=[]
#     y_test=[]
#     frame_ids=[]
#     det_scores=[]
#     qua_scores=[]
#     video_names=[]
#     for video_name in data_test.keys():
#         tmp_df = pd.DataFrame(data_test[video_name], columns=['frame_id','box','det_score','qua_score','feat_arr'])
#         frame_id=np.array(tmp_df['frame_id'])
#         det_score=np.array(tmp_df['det_score'])
#         qua_score=np.array(tmp_df['qua_score'])
#         feat_arr=np.array(tmp_df['feat_arr'])
#         for i in range(len(feat_arr)):
#             x_test.append(feat_arr[i].reshape(32, 16))
#             frame_ids.append(frame_id[i])
#             det_scores.append(det_score[i])
#             qua_scores.append(qua_score[i])
#             video_names.append(video_name)
#         # print(x_test,frame_ids,det_scores,qua_scores)
#     y_test.append(np.array(frame_ids))
#     y_test.append(np.array(video_names))
#     y_test.append(np.array(det_scores))  ############################################################添加人脸置信度分数
#     y_test.append(np.array(qua_scores))  ############################################################添加人脸图片质量评估分数
#     y_test = np.array(y_test)
#     x_test=np.array(x_test)
#     print(x_test.shape,y_test.shape)
#     np.save("./x_testall.npy",x_test)
#     np.save("./y_testall.npy",y_test)
#     print('########数据加载完成！###########')
# read_pickle_test()
def load_data_train():
    x_train=np.load("./x_trainall3.npy")
    y_train=np.load("./y_trainall3.npy")
    print(x_train.shape,y_train.shape)
    img_rows, img_cols = 32, 16  # 将每张图片的特征转为32*16
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    return x_train,y_train
def mean_train(x_train,y_train):
    y_label = y_train[0].tolist()
    y_video_name = y_train[2].tolist()
    a = y_video_name[0]
    b = np.zeros(shape=(32, 16,1))
    train_mean = []
    label = []
    video_name=[]
    y=[]
    j=0
    for i in range(len(y_video_name)): #遍历视频名，对属于一个视频的所有图片取32*16特征均值
        if a != y_video_name[i]:
            if j==0:
                j=1
            train_mean.append(b / float(j))
            label.append(y_label[i - 1])
            video_name.append(y_video_name[i-1])
            a = y_video_name[i]
            b = np.zeros(shape=(32, 16,1))
            j=0
            b = b + x_train[i]
            # j+=1
        else:
            b = b + x_train[i]
            j+=1
        print(i)
    if j==0:
        j=1
    # train_mean.append(b / float(j))
    # label.append(y_label[i])  对于最后一个视频
    # video_name.append(y_video_name[i])
    y.append(label)
    y.append(video_name)
    np.save('./x_trainmeannew',np.array(train_mean))
    np.save('./y_trainmeannew',np.array(y))
# x_train,y_train=load_data_train()
# mean_train(x_train,y_train)

def load_data_val():
    x_test = np.load("./x_valall.npy")
    y_test = np.load("./y_valall.npy")
    if K.image_data_format() == 'channels_first':
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    x_test = x_test.astype('float32')
    return x_test, y_test, input_shape
# def data_generator(data, targets, batch_size,num_classes):
#     batches = (len(data) + batch_size - 1)//batch_size
#     while(True):
#          for i in range(batches):
#               X = data[i*batch_size : (i+1)*batch_size]
#               Y = targets[i*batch_size : (i+1)*batch_size]
#               Y=keras.utils.to_categorical(Y,num_classes)
#               yield (X, Y)
def generate_batch_data_random(train, batch_size,num_classes):
        ylen = len(train)
        loopcount = ylen // batch_size
        while (True):
          for i in range(loopcount):
             a=train[i * batch_size:(i + 1) * batch_size]
             x=[]
             y=[]
             for j in range(batch_size):
               # c=random.random()
               # if c<=0.1:
               #     b=a[j][0][:,::-1]
               # elif 0.15<c<=0.3:
               #     b = a[j][0][::-1]
               # else:
               #     b=a[j][0]
                x.append(a[j][0])
                y.append(a[j][1])
             X=np.array(x)
             Y = keras.utils.to_categorical(np.array(y), num_classes)
             yield X, Y

def generate_val_data_random(val, batch_size,num_classes):
    ylen = len(val)
    loopcount = ylen // batch_size
    while (True):
        for i in range(loopcount):
            a = val[i * batch_size:(i + 1) * batch_size]
            x = []
            y = []
            for j in range(batch_size):
                x.append(a[j][0])
                y.append(a[j][1])
            X = np.array(x)
            Y = keras.utils.to_categorical(np.array(y), num_classes)
            yield X, Y   #y 128*4394
def get_session(gpu_fraction=0.9):#使用90%的GPU
    num_threads = os.environ.get('OMP_NUM_THREADS')#执行是可用的线程数
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
if __name__ == "__main__":
    batch_size = 128
    num_classes =4935#有4934个人物+1
    epochs =30
    img_rows, img_cols = 32, 16 # 将每张图片的特征转为32*16
    img_channels = 1
    # lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=, min_lr=0.5e-6)
    # earlystop = EarlyStopping(min_delta=0.001, patience=10)
    # csv_logger = CSVLogger('6.txt')
    # earlystop = EarlyStopping(monitor='val_acc', patience=10, verbose=1)
    lr_schedule = lambda epoch: 0.0005 * 0.4     ** epoch
    learning_rate = np.array([lr_schedule(i) for i in range(epochs)])
    print(learning_rate)
    changelr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))
    checkpoint = ModelCheckpoint(filepath='./models/weights-densenet-{epoch:02d}-{val_acc:.3f}.h5', monitor='val_acc',save_best_only=False, save_weights_only=True)
    tensorboard = TensorBoard(log_dir='./models/logs', write_graph=True)
    earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    x_train = np.load("./x_trainmean11.npy")
    y_train = np.load("./y_trainmean11.npy")
    # y_train=np.array(y_train[0])
    print(len(set(y_train.tolist())))
    x_val = np.load("./x_valmean.npy")
    y_val = np.load("./y_valmean.npy")
    y_val = np.array(y_val[0])
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    train=[]
    val=[]
    for i in range(len(y_train)):
        a=[]
        a.append(x_train[i])
        a.append(y_train[i])
        train.append(a)

    random.shuffle(train)
    for i in range(len(y_val)):
        a=[]
        a.append(x_val[i])
        a.append(y_val[i])
        val.append(a)

    random.shuffle(val)
    train.extend(val[:87032])
    val=val[87032:]
    print(len(train))
    print(len(val))
    K.set_session(get_session())  # 使用gpu训练
    model = resnet.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols),num_classes)
    # model.load_weights('./weights-densenet-10-0.79.h5')
    model.load_weights('./models/weights-densenet-04-0.800.h5')
    # model=load_model('./0.9250.636.h5')
    # model = Sequential()
    # model.add(Conv2D(32, kernel_size=(3, 3),
    #                  activation='relu',
    #                  input_shape=input_shape))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    hist=model.fit_generator(generator=generate_batch_data_random(train,batch_size,num_classes),
              steps_per_epoch=len(train) // batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=generate_val_data_random(val,batch_size,num_classes),
              validation_steps=len(val) // batch_size,
              max_q_size=100,
              callbacks=[checkpoint,tensorboard,changelr,earlystop])
    with open('./models/hist2.txt', 'w') as f:
        f.write(str(hist.history))








