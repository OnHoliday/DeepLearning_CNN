import numpy as np
from keras.preprocessing import image
import os, shutil
from os import listdir
from os.path import isfile, join
import pandas as pd
from keras import callbacks
import time
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
import random
from keras.utils import to_categorical
from PIL import Image
from pathlib import Path, PureWindowsPath
import getpass
import os
from sklearn.metrics import confusion_matrix, classification_report
import glob
from pathlib import Path, PureWindowsPath # please check this medium article!! https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f


def get_path():
    if getpass.getuser() == 'Konrad':
        project_dir = Path(PureWindowsPath('D:\\DeepLearningProject'))
    elif getpass.getuser() == 'fruechtnicht':
        project_dir = Path('/Users/fruechtnicht/NOVA/M.Sc_Data_Science_and_Advanced_Analytics/Semester2/Deep_Learning/Project/project_dir')
    elif getpass.getuser() == 'dominika.leszko':
        project_dir = Path(r'C:\Users\dominika.leszko\Desktop\NOVAIMS\SEMESTER2\Deep Learinng\PROJECT\git_repo')
    elif getpass.getuser() == 'jojo':
        project_dir = Path(r'C:\Users\jojo\Documents\Uni\Second Semester\Deep Learning\Project\Master')
    else:
        raise ValueError('Check you own user name and add your project root with a proper elif statement !!!')
    return project_dir


def get_current_directory():
    path = os.getcwd()
    return path


def get_time_stamp():
    timestr = time.strftime("%Y%m%d-%H%M%S")
    return timestr


def prepare_input_data(path, nr_of_examples):
    lst = os.listdir(path)

    for item in lst:
        if not item.endswith(".jpg"):
            os.remove(path / item)

    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

    random.shuffle(onlyfiles)

    df = pd.DataFrame(columns=['age', 'gender', 'ethnic','file_name'])
    index = 0
    for item in onlyfiles:
        if index < nr_of_examples:
            a = item.split("_", 3)
            try:
                age, gender, ethnic, time = a[0], a[1], a[2], a[3]
            except:
                pass
            data = time[:8]
            df.loc[index, ['age', 'gender', 'ethnic', 'file_name']] = age, gender, ethnic, item
            index += 1
        else:
            df['age'] = df['age'].astype('float')

            return df
    df['age'] = df['age'].astype('float')
    return df

def organize_cropped_files(project_dir=get_path()):
    #create train & test directories
    train_dir = os.path.join(project_dir, 'UTKFace')
    os.mkdir(project_dir / 'UTKFace_test')
    test_dir = os.path.join(project_dir, 'UTKFace_test')
    os.mkdir(project_dir / 'UTKFace_pred')
    pred_dir=os.path.join(project_dir, 'UTKFace_pred')
    #move random 20% files to UTKFace_test
    onlyfiles = [f for f in listdir(project_dir / 'UTKFace') if isfile(join(project_dir / 'UTKFace', f))]
    random.shuffle(onlyfiles)
    t=int(0.8*len(onlyfiles))
    mv_files=onlyfiles[t:]
    for i in range(len(mv_files)):
        i_path_from=os.path.join(train_dir, mv_files[i])
        i_path_to=os.path.join(test_dir, mv_files[i])
        shutil.move(i_path_from, i_path_to)
    # move random 1% files to UTKFace_pred
    random.shuffle(mv_files)
    t2=int(0.99*len(mv_files))
    mv_files2=mv_files[t2:]
    for i in range(len(mv_files2)):
        i_path_from2=os.path.join(test_dir, mv_files2[i])
        i_path_to2=os.path.join(pred_dir, mv_files2[i])
        shutil.move(i_path_from2, i_path_to2)




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
    project_dir = get_path()
    # wd = get_current_directory()
    # print(wd)
    # load json and create model
    # model_name_json = model_name + '.json'
    # Uncoment to use Lucas CVS
    model_name_type = model_name + '.json'
    model_path = project_dir / model_name_type
    json_file = open(model_path, 'r')
    # json_file = open(model_name_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    from keras.models import model_from_json

    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    model_name_h5 = model_name + '.h5'
    model_path = project_dir / model_name_h5

    loaded_model.load_weights(model_path)
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

# def callbackEarlyStopping():
#     es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4, min_delta=0.0001)
#     return es

def callbackCheckpoint(model_name):
    model_name_h5 = model_name + '_checkPoint.h5'
    mc = ModelCheckpoint(model_name_h5, monitor='val_loss', mode='min', save_best_only=True)
    return mc

def callbackTensor():
    tb = TensorBoard(log_dir=get_path() / 'logs', histogram_freq=0, write_graph=True, write_images=True)
    return tb


def create_unique_cv_logger():
    wd = get_current_directory()
    now = get_time_stamp()
    name = Path(now + '_training.csv')
    name = project_dir / Path('log') / name
    csv_logger = CSVLogger(name)
    return csv_logger

def create_cv_logger(model_name):
    wd = get_current_directory()
    name = model_name + '_training.csv'
    name = wd + r'/log/' + name
    csv_logger = CSVLogger(name, append=True)
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


def gnerate_genarator_multi(datagen, df, path, target_size, batch_size, target1, target2, target3, color_mode):
    GENy1 = datagen.flow_from_dataframe(
            dataframe = df,
            directory = path,
            x_col = 'file_name',
            y_col = target1,
            target_size = (target_size, target_size),
            batch_size = batch_size,
            color_mode = color_mode,
            class_mode = 'binary',
            seed = 1)

    GENy2 = datagen.flow_from_dataframe(
            dataframe = df,
            directory = path,
            x_col = 'file_name',
            y_col = target2,
            target_size = (target_size, target_size),
            batch_size = batch_size,
            color_mode = color_mode,
            class_mode = 'categorical',
            seed = 1)


    GENy3 = datagen.flow_from_dataframe(
            dataframe = df,
            directory = path,
            x_col = 'file_name',
            y_col = target3,
            target_size = (target_size, target_size),
            batch_size = batch_size,
            color_mode = color_mode,
            class_mode = 'other',
            seed = 1)

    while True:
            y1 = GENy1.next()
            y2 = GENy2.next()
            y3 = GENy3.next()
            yield y1[0], {'gender_output': y1[1], 'race_output': y2[1], 'age_output': y3[1]}



def make_new_prediction(classifier, target, target_size, cropped=False):
    wd = get_current_directory()
    if cropped:
        path1 = wd + '\\UTKFace_pred\\'#for cropped
    else:
        path1 = wd + '\part2\\'#for non-cropped

    onlyfiles = [f for f in listdir(path1) if isfile(join(path1, f))]
    random_pic = Path(random.choice(onlyfiles))
    path = path1 / random_pic

    test_image = image.load_img(path, target_size=(target_size, target_size))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)
    print(result)
    prediction = mapper(result, target)
    return prediction , path

def voting_prediction(models, target, target_size, cropped=True, testAll = True):
    wd = get_current_directory()
    if cropped:
        path1 = wd + '\\UTKFace_pred\\'#for cropped
    else:
        path1 = wd + '\part2\\'#for non-cropped

    onlyfiles = [f for f in listdir(path1) if isfile(join(path1, f))]
    len_ = len(onlyfiles)
    if testAll == False:

        random_pic = Path(random.choice(onlyfiles))
        path = path1 / random_pic

        test_image = image.load_img(path, target_size=(target_size, target_size))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        i=0
        preds = []
        for model in models:
            result = model.predict(test_image)
            preds.append(result[0])

        pred = np.average(preds, axis=0)

        if np.argmax(pred) == 0:
            x='White'
        elif np.argmax(pred) == 1:
            x='Black'
        elif np.argmax(pred) == 2:
            x='Asian'
        elif np.argmax(pred) == 3:
            x='Indian'
        else:
            x='Other'

        return x, path

    else:
        ground_truth = []
        all_preds = []
        for i in range(len_):
            random_pic = Path(random.choice(onlyfiles))
            path = path1 / random_pic
            item = str(random_pic)
            a = item.split("_", 3)
            age, gender, ethnic, time = a[0], a[1], a[2], a[3]

            if target=='age':
                ground_truth.append(int(age))
            elif target=='gender':
                ground_truth.append(int(gender))
            else:
                ground_truth.append(int(ethnic))

            test_image = image.load_img(path, target_size=(target_size, target_size))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)

            i = 0
            preds = []
            for model in models:
                result = model.predict(test_image)
                preds.append(result[0])
            pred = np.average(preds, axis=0)
            all_preds.append(np.argmax(pred))

        if target == 'age':
            print(np.square(np.subtract(np.array(ground_truth), np.array(all_preds))))
            mse = np.mean(np.square(np.subtract(np.array(ground_truth), np.array(all_preds))))
            print(mse)
        else:
            cm = confusion_matrix(ground_truth, all_preds)
            print(classification_report(ground_truth, all_preds))
            print(cm)

def final_preds_age(models, target_size, cropped=True):
    wd = get_path()
    if cropped:
        path1 = wd / 'UTKFace_pred'#for cropped
    else:
        path1 = wd / 'part2'#for non-cropped

    onlyfiles = [f for f in listdir(path1) if isfile(join(path1, f))]
    len_ = len(onlyfiles)

    df_pred = pd.DataFrame(columns=['age_truth', 'age_preds'])

    for i in range(len_):
        pic = Path(onlyfiles[i])
        path = path1 / pic
        item = str(pic)
        a = item.split("_", 3)
        age, gender, ethnic, time = a[0], a[1], a[2], a[3]

        ground_truth = int(age)
        target_size = 128
        print(path)

        test_image = image.load_img(path, target_size=(target_size, target_size))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        preds = []
        for model in models:
            result = model.predict(test_image)
            print(result[0])
            res = int(result[0]/128)
            print(res)
            preds.append(res)
            print(preds)
        pred = np.average(preds, axis=0)


        df_pred.loc[i, ['age_truth', 'age_preds']] = ground_truth , pred

    path1 = wd / 'log' / 'final_results_ensemble_age.csv'
    df_pred.to_csv(path1)

def final_preds_gender(models, target_size, cropped=True):
    wd = get_path()
    if cropped:
        path1 = wd / 'UTKFace_pred'#for cropped
    else:
        path1 = wd / 'part2'#for non-cropped

    onlyfiles = [f for f in listdir(path1) if isfile(join(path1, f))]
    len_ = len(onlyfiles)

    df_pred = pd.DataFrame(columns=['gender_truth', 'gender_preds'])

    for i in range(len_):
        pic = Path(onlyfiles[i])
        path = path1 / pic
        item = str(pic)
        a = item.split("_", 3)
        age, gender, ethnic, time = a[0], a[1], a[2], a[3]

        ground_truth = int(gender)
        target_size = 128
        print(path)

        test_image = image.load_img(path, target_size=(target_size, target_size))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        preds = []
        for model in models:
            result = model.predict(test_image)
            print(result[0])
            preds.append(result[0])
            print(preds)
        pred = np.average(preds, axis=0)

        if pred < .5:
            pr = 0
        else:
            pr = 1

        df_pred.loc[i, ['gender_truth', 'gender_preds']] = ground_truth , pr

    path1 = wd / 'log' / 'final_results_ensemble_gender.csv'
    df_pred.to_csv(path1)


def final_preds_race(models, target_size, cropped=True):
    wd = get_path()
    if cropped:
        path1 = wd / 'UTKFace_pred'#for cropped
    else:
        path1 = wd / 'part2'#for non-cropped

    onlyfiles = [f for f in listdir(path1) if isfile(join(path1, f))]
    len_ = len(onlyfiles)

    df_pred = pd.DataFrame(columns=['race_truth', 'race_preds'])

    for i in range(len_):
        pic = Path(onlyfiles[i])
        path = path1 / pic
        item = str(pic)
        a = item.split("_", 3)
        age, gender, ethnic, time = a[0], a[1], a[2], a[3]

        ground_truth = int(ethnic)
        target_size = 128
        print(path)

        test_image = image.load_img(path, target_size=(target_size, target_size))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        preds = []
        for model in models:
            result = model.predict(test_image)
            print(result[0])
            res = result[0]
            preds.append(res)
            print(preds)

        pred = np.average(preds, axis=0)

        if np.argmax(pred) == 0:
            x = 0
        elif np.argmax(pred) == 1:
            x = 1
        elif np.argmax(pred) == 2:
            x = 2
        elif np.argmax(pred) == 3:
            x = 3
        else:
            x = 4
        # prediction['Ethnicity'] = x

        df_pred.loc[i, ['race_truth', 'race_preds']] = ground_truth , x

    path1 = wd / 'log' / 'final_results_ensemble_race.csv'
    df_pred.to_csv(path1)



def ensamble_median_prediction(models, target, target_size, cropped=True, testAll = True):
    wd = get_current_directory()
    if cropped:
        path1 = wd + '\\UTKFace_pred\\'#for cropped
    else:
        path1 = wd + '\part2\\'#for non-cropped

    onlyfiles = [f for f in listdir(path1) if isfile(join(path1, f))]
    len_ = len(onlyfiles)

    ground_truth = []
    all_preds = []
    for i in range(len_):
        random_pic = Path(random.choice(onlyfiles))
        path = path1 / random_pic
        item = str(random_pic)
        a = item.split("_", 3)
        age, gender, ethnic, time = a[0], a[1], a[2], a[3]

        ground_truth.append(int(age))

        test_image = image.load_img(path, target_size=(target_size, target_size))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        i = 0
        preds = []
        for model in models:
            result = model.predict(test_image)
            preds.append(result[0])
        pred = np.average(preds, axis=0)
        all_preds.append(np.argmax(pred))


    print(np.square(np.subtract(np.array(ground_truth), np.array(all_preds))))
    mse = np.mean(np.square(np.subtract(np.array(ground_truth), np.array(all_preds))))
    print(mse)



def classifier(n):
    return lambda: 0 if n < .5 else 0


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

def data_generator_cust(df,im_width, im_height, for_training, path, batch_size):
    images, ages, races, genders = [], [], [], []
    n_races = len(df['ethnic'].unique())
    df['age'] = df['age'].astype(float)
    df['gender'] = df['gender'].astype(float)

    max_age = 116


    while True:
        for i in range(len(df)):
            r = df.iloc[i]
            file, age, race, gender = Path(r['file_name']), r['age'], r['ethnic'], r['gender']
            im = Image.open(path/file)
            im = im.resize((im_width, im_height))
            rgb_im = im.convert('RGB')
            im = np.array(rgb_im) / 255.0
            images.append(im)
            ages.append(age / max_age)
            races.append(to_categorical(race, n_races))
            genders.append(gender)
            if len(images) == batch_size:
                yield np.array(images), [np.array(ages), np.array(races), np.array(genders)]
                images, ages, races, genders = [], [], [], []
        if not for_training:
            break

def make_new_p_multi(classifier):


    path1 = project_dir / Path('UTKFace_pred')
    onlyfiles = [f for f in listdir(path1) if isfile(join(path1, f))]
    random_pic = Path(random.choice(onlyfiles))
    path = path1 / random_pic

    test_image = image.load_img(path, target_size=(224, 224))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)
    print(result)
    prediction = mapper_result_multi(result)
    return prediction, path

def mapper_result_multi(result):
    prediction = dict()

    prediction['age'] = (int(result[0][0][0]*116))
    gen = lambda x: 'Male' if x < .5 else 'Female'
    prediction['gender'] = gen(result[2])

    if np.argmax(result[1]) == 0:
        x='White'
    elif np.argmax(result[1]) == 1:
        x='Black'
    elif np.argmax(result[1]) == 2:
        x='Asian'
    elif np.argmax(result[1]) == 3:
        x='Indian'
    else:
        x='Other'
    prediction['Ethnicity'] = x

    return str(prediction)

