# set the matplotlib backend so figures can be saved in the background
import matplotlib
from utils import *
# matplotlib.use("Agg")
import getpass
from pathlib import Path, PureWindowsPath # please check this medium article!! https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f

# import the necessary packages
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from MULTI_CNN import Group12Net
# from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle

import os

if getpass.getuser() == 'Konrad':
    project_dir = Path(PureWindowsPath('D:\\DeepLearningProject'))
elif getpass.getuser() == 'fruechtnicht':
    project_dir = Path('/Users/fruechtnicht/NOVA/M.Sc_Data_Science_and_Advanced_Analytics/Semester2/Deep Learning/Project/Git')
else:
    raise ValueError('Check you own user name and add proper elif statement !!!')
# if you have a windows computer please specify your project path as Konrad, if not as fruechtnicht
os.chdir(project_dir)


# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions


## Parameters
target_size = 64
batch_size = 32
target = ['ethnic', 'age']               # 'gender' or 'age'
color_mode = 'rgb'              #  'grayscale'
class_mode = 'categorical'      # 'binary'

## Training data

path1 = project_dir / 'UTKFace'
train_df = prepare_input_data(path1, 1000)

## Test data

path3 = project_dir / 'UTKFace_test'
test_df = prepare_input_data(path3, 300)

# train_datagen = create_trainingDataGenerator_instance()
# training_set1 = create_set(train_datagen, train_df, path1, target_size, batch_size, 'gender', color_mode, class_mode)
# training_set2 = create_set(train_datagen, train_df, path1, target_size, batch_size, 'ethnic', color_mode, class_mode)
#
# def join_generators(xgenerators, ygenerator):
#     while True: # keras requires all generators to be infinite
#         data = [next(g) for g in xgenerators]
#         x = [d[0] for d in data]
#
#         yield x, next(ygenerator)
#
# def join_generators2(xgenerators, ygenerator):
#     while True: # keras requires all generators to be infinite
#         data = [next(g) for g in xgenerators]
#         x = [d[0] for d in data]
#         yield x, next(ygenerator)
#
# join_generators(training_set1, [training_set1, training_set2])


# initialize our FashionNet multi-output network
gr12 = Group12Net('tryModel')
gr12.build(64, 64, numGender=2, numRace=5, finalAct="softmax")
# gr12.load_model()



input_imgen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   rotation_range=5.,
                                   horizontal_flip = True)

test_imgen = ImageDataGenerator(rescale = 1./255)
#
inputgenerator = gnerate_genarator_multi(input_imgen, train_df, path1, target_size, batch_size, 'gender', 'ethnic', 'age', color_mode, class_mode)
testgenerator = gnerate_genarator_multi(test_imgen, test_df, path3, target_size, batch_size, 'gender', 'ethnic', 'age', color_mode, class_mode)

tensor = callbackTensor()

callbacks = [tensor]
history = gr12.model.fit_generator(inputgenerator,
                              steps_per_epoch=3,
                              epochs=2,
                              validation_data=testgenerator,
                              validation_steps=10,
                              callbacks=callbacks,
                              shuffle=False)

gr12._save_model()

wd = get_current_directory()
path1 = wd + '\part2\\'

onlyfiles = [f for f in listdir(path1) if isfile(join(path1, f))]
random_pic = random.choice(onlyfiles)
path = wd + '\part2\\' + random_pic

test_image = image.load_img(path, target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
(categoryGender, categoryRace,categoryAge) = gr12.model.predict(test_image)

categoryGenderID = categoryGender[0].argmax()
categoryRaceID = categoryRace[0].argmax()
categoryAgeID = categoryAge[0]
print(categoryGenderID, categoryRaceID,categoryAgeID)

from plots import plot_new_pred
prediction = 'Pred in console'
# prediction, path = make_new_prediction(gr12.model.model, target)
plot_new_pred(prediction, path)