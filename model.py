from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import Convolution2D, MaxPooling2D
from keras.preprocessing.image import img_to_array, load_img, flip_axis, random_shift
from keras.utils import np_utils
from keras.regularizers import l2, activity_l2
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
import cv2
import random

#load train data from csv file
def load_data():
    #we should use left camera and right camera to recover, so we need add a shift to steering angle
    adjust_steering = 0.45
    data = pd.read_csv("./data/driving_log.csv")
    data = data.loc[data['throttle'] >= 0.2]
    y_left = list(map(lambda x: x + adjust_steering, data.ix[:,'steering']))
    x_left = (data.ix[:, 'left']).values
    y_right = list(map(lambda x: x - adjust_steering, data.ix[:,'steering']))
    x_right = (data.ix[:, 'right']).values
    
    data = data.loc[abs(data['steering']) > 0.1]
    y_center = (data.ix[:,'steering']).values
    x_center = (data.ix[:, 'center']).values
    
    #x_ = np.append(x_left , x_right)
    #y_ = np.append(y_left,  y_right)
    x_ = np.append(np.append(x_center, x_left) , x_right)
    y_ = np.append(np.append(y_center, y_left),  y_right)
    return x_, y_

def shuffle_and_split(x_,y_):
    x_data, y_data = shuffle(x_,y_)
    #return x_data, y_data
    x_train, x_validation, y_train, y_validation = train_test_split(x_data, y_data, test_size=0.1, random_state=42)
    return x_train, y_train, x_validation, y_validation

    
def _generator(batch_size, x_, y_, shape):
    while 1:
        batch_x, batch_y = [],[]
        for i in range(batch_size):
            index = random.randint(0, len(x_)-1)
            keep_flag = 0
            while keep_flag == 0:
                image, label = load_image(x_, y_, index, shape)
                if abs(label) < 0.2:
                    if random.random() < 0.2:
                        keep_flag = 1
                else :
                    keep_flag = 1
            batch_x.append(image)
            batch_y.append(label)
        yield np.array(batch_x), np.array(batch_y)

def _validation_generator(batch_size, x_, y_, shape):
    while 1:
        batch_x, batch_y = [],[]
        for i in range(batch_size):
            index = random.randint(0, len(x_)-1)
            image, label = load_image(x_, y_, index, shape, augment=False)
            batch_x.append(image)
            batch_y.append(label)
        yield np.array(batch_x), np.array(batch_y)



def load_image(x_, y_, index, shape, augment=True):
    path = './data/' + x_[index].strip()
    
    image = cv2.imread(path)
    image = cv2.resize(image, (shape[1], shape[0]))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)   
    label = y_[index]
    
    if augment:
        image = add_random_shadow(image)
        image = augment_brightness_camera_images(image)
        image,label = trans_image(image,label,100, shape)
    image = np.array(image)    
    if augment:    
        #image = random_shift(image, 0, 0.2, 0, 1,2)
        image = flip_axis(image, 1)
        label = -label
    return image, label

def augment_brightness_camera_images(image):

    image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .20+np.random.uniform()
    image[:,:,2] = image[:,:,2]*random_bright
    image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
    return image

def trans_image(image,steer,trans_range, shape):
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 40*np.random.uniform()-40/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(shape[1],shape[0]))
    
    return image_tr,steer_ang

def add_random_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]

    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)

    return image


def behavioral_clone_model(shape):
    model = Sequential()
    
    model.add(Lambda(lambda x: x/255.-0.5, input_shape=shape))

    model.add(Convolution2D(32, 3, 3, activation='elu', init='he_normal'))
    model.add(MaxPooling2D())

    #model.add(Convolution2D(32, 3, 3, activation='elu'))
    #model.add(MaxPooling2D())
    #model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, activation='elu', init='he_normal'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, 3, 3, activation='elu', init='he_normal'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, 3, 3, activation='elu', init='he_normal'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, init='he_normal'))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))

    model.add(Dense(512, init='he_normal'))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))

    model.add(Dense(256, init='he_normal'))
    model.add(Activation('elu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='linear', init='he_normal'))
    model.compile(loss='mse',
              optimizer='adam',
              metrics=['mean_absolute_error'])


    return model

def save_model(model):
    #model_json = model.to_json()
    #open('model.json', 'w').write(model_json)
    #model.save_weights('model.h5')
    model.save('model.h5')

shape = (80,160,3)
x,y = load_data()
x_train, y_train, x_validation, y_validation = shuffle_and_split(x,y)
#x_train, y_train = shuffle_and_split(x,y)
print(len(x_train), len(y_train))
my_net = behavioral_clone_model(shape)
my_net.fit_generator(_generator(128, x_train, y_train, shape), samples_per_epoch=24064, nb_epoch=12, validation_data=_validation_generator(128, x_validation, y_validation, shape),nb_val_samples=1920)
save_model(my_net)

