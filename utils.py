import numpy as np
from keras.preprocessing import image
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
from keras import callbacks
import time
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
import random

def get_current_directory():
    path = os.getcwd()
    return path


def get_time_stamp():
    timestr = time.strftime("%Y%m%d-%H%M%S")
    return timestr


def prepare_input_data(path, nr_of_examples):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    df = pd.DataFrame(columns=['age', 'gender', 'ethnic','file_name'])
    index = 0
    for item in onlyfiles:
        if index < nr_of_examples:
            a = item.split("_", 3)
            try:
                age, gender, ethnic, time = a[0], a[1], a[2], a[3]
            except:
                 pass
            df.loc[index, ['age', 'gender', 'ethnic', 'file_name']] = age, gender, ethnic, item
            index += 1
        else:
            df['age'] = df['age'].astype('float')
            return df
    df['age']=df['age'].astype('float')
    return df


def save_model(classifier, model_name):
    # serialize model to JSON
    model_name_json = model_name + '.json'
    model_json = classifier.to_json()
    with open(model_name_json, "w") as json_file:
        json_file.write(model_json)

    model_name_h5 = model_name + '.h5'
    # serialize weights to HDF5
    classifier.save_weights(model_name_h5)
    print("Saved model to disk")


def load_model(model_name):
    # load json and create model
    model_name_json = model_name + '.json'
    json_file = open(model_name_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    from keras.models import model_from_json


    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    model_name_h5 = model_name + '.h5'
    loaded_model.load_weights(model_name_h5)
    print("Loaded model from disk")
    return loaded_model

def callback_history():
    class LossHistory(callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.losses = []

        def on_epoch_end(self, epoch, logs={}):
            self.losses.append(logs.get('loss'))
    history = LossHistory()
    return history

def callbackEarlyStopping():
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2, min_delta=1)
    return es

def callbackCheckpoint(model_name):
    model_name_h5 = model_name + '_checkPoint.h5'
    mc = ModelCheckpoint(model_name, monitor='val_loss', mode='min', save_best_only=True)
    return mc

def callbackTensor():
    tb = TensorBoard(log_dir='/logs', histogram_freq=0, write_graph=True, write_images=True)
    return tb

def create_cv_logger():
    wd = get_current_directory()
    now = get_time_stamp()
    name = now + '_training.csv'
    name = wd + r'/log/' + name
    csv_logger = CSVLogger(name)
    return csv_logger


def create_trainingDataGenerator_instance():
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    return train_datagen

def create_testingDataGenerator_instance():
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    return test_datagen

def create_set(datagen, df, path, target_size, batch_size, target, color_mode, class_mode):
    set = datagen.flow_from_dataframe(
            dataframe = df,
            directory = path,
            x_col = 'file_name',
            y_col = target,
            target_size = (target_size, target_size),
            batch_size = batch_size,
            color_mode = color_mode,
            class_mode = class_mode)
    return set


def make_new_prediction(classifier, target):
    wd = get_current_directory()
    path1 = wd + '\part2\\'

    onlyfiles = [f for f in listdir(path1) if isfile(join(path1, f))]
    random_pic = random.choice(onlyfiles)
    path = wd + '\part2\\' + random_pic

    test_image = image.load_img(path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)

    prediction = mapper(result, target)
    return prediction , path

def mapper(result, target):
    if target == 'ethnic':
        if result[0][0] == 1:
            prediction = 'african'
        elif result[0][0] == 2:
            prediction = 'asian'
        elif result[0][0] == 3:
            prediction = 'indian'
        elif result[0][0] == 4:
            prediction = 'latino'
        else:
            prediction = 'white'

    elif target == 'gender':
        if result[0][0] == 1:
           prediction = 'woman'
        else:
           prediction = 'man'
    else:
        prediction = str(result[0][0])

    return prediction
