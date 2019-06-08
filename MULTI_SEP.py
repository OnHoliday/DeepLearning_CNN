# set the matplotlib backend so figures can be saved in the background
import matplotlib
from utils import *
from pathlib import Path, PureWindowsPath # please check this medium article!! https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f

from MULTI_CNN_Models import Group12Net
import numpy as np
import random

project_dir = get_path()

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions


## Parameters
target_size = 224
batch_size = 32
target = ['gender', 'ethnic', 'age']               # 'gender' or 'age'
color_mode = 'rgb'              #  'grayscale'
class_mode = 'categorical'      # 'binary'

## Training data
#
path1 = project_dir / 'UTKFace'
train_df = prepare_input_data(path1, 1000)

## Test data

path3 = project_dir / 'UTKFace_test'
test_df = prepare_input_data(path3, 300)

train_datagen = create_trainingDataGenerator_instance()
training_set1 = create_set(train_datagen, train_df, path1, target_size, batch_size, 'gender', color_mode, class_mode)
training_set2 = create_set(train_datagen, train_df, path1, target_size, batch_size, 'ethnic', color_mode, class_mode)
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
gr12 = Group12Net('multiModel')
gr12.build(target_size, target_size, numGender=1, numRace=5, finalAct="softmax")
# gr12.load_model()
# final_preds(gr12.model, target_size, cropped=True)
gr12 = Group12Net('multiModel_Konrad2')
# gr12.build(target_size, target_size, numGender=1, numRace=5, finalAct="softmax")
gr12.load_model()
# final_preds(gr12.model, target_size, cropped=True)

#
#inputgenerator = gnerate_genarator_multi(input_imgen, train_df, path1, target_size, batch_size, 'gender', 'ethnic', 'age', color_mode)
#testgenerator = gnerate_genarator_multi(test_imgen, test_df, path3, target_size, batch_size, 'gender', 'ethnic', 'age', color_mode)
#
input_imgen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   rotation_range=5.,
                                   horizontal_flip = True)

test_imgen = ImageDataGenerator(rescale = 1./255)
#
inputgenerator = gnerate_genarator_multi(input_imgen, train_df, path1, target_size, batch_size, 'gender', 'ethnic', 'age', color_mode)
testgenerator = gnerate_genarator_multi(test_imgen, test_df, path3, target_size, batch_size, 'gender', 'ethnic', 'age', color_mode)


# tensor = callbackTensor()

checkPoint = callbackCheckpoint('tryModel')
tensor = callbackTensor()
callbacks = [checkPoint, tensor]

history = gr12.model.fit_generator(inputgenerator,
                              steps_per_epoch=3,
                              epochs=2,
                              validation_data=testgenerator,
                              validation_steps=10,
                              callbacks=callbacks,
                              shuffle=False)
#
# tensor = callbackTensor()

csv_logger = create_cv_logger(gr12.model_name)
history = callback_history()
# earlyStop = callbackEarlyStopping()
checkPoint = callbackCheckpoint(gr12.model_name)
tensor = callbackTensor()
callbacks = [csv_logger, history, checkPoint, tensor]

history = gr12.model.fit_generator(inputgenerator,
                              steps_per_epoch=1,
                              epochs=1,
                              validation_data=testgenerator,
                              validation_steps=1,
                              callbacks=callbacks,
                              use_multiprocessing=False,
                              workers=1,
                              shuffle=True)

# df_history = pd.DataFrame(history)
# df_history.to_csv('backup_multioutput.csv')

# gr12._save_model()
#
# wd = get_current_directory()
# path1 = wd + '\part2\\'
#
# onlyfiles = [f for f in listdir(path1) if isfile(join(path1, f))]
# random_pic = random.choice(onlyfiles)
# path = wd + '\part2\\' + random_pic
#
# test_image = image.load_img(path, target_size=(64, 64))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis=0)
# (categoryGender, categoryRace,categoryAge) = gr12.model.predict(test_image)
#
# categoryGenderID = categoryGender[0].argmax()
# categoryRaceID = categoryRace[0].argmax()
# categoryAgeID = categoryAge[0]
# print(categoryGenderID, categoryRaceID,categoryAgeID)
#
# from plots import plot_new_pred
# prediction = 'Pred in console'
# # prediction, path = make_new_prediction(gr12.model.model, target)
# plot_new_pred(prediction, path)