import os
from utils import *
from cnn import CnnSolver
from plots import plot_new_pred
import getpass
from pathlib import Path, PureWindowsPath # please check this medium article!! https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f



if getpass.getuser() == 'Konrad':
    project_dir = Path(PureWindowsPath('D:\\DeepLearningProject'))
elif getpass.getuser() == 'fruechtnicht':
    project_dir = Path('/Users/fruechtnicht/NOVA/M.Sc_Data_Science_and_Advanced_Analytics/Semester2/Deep_Learning/Project/project_dir')
elif getpass.getuser() == 'dominika.leszko':
    project_dir = Path(r'C:\Users\dominika.leszko\Desktop\NOVAIMS\SEMESTER2\Deep Learinng\PROJECT\git_repo')
else:
    raise ValueError('Check you own user name and add proper elif statement !!!')
# if you have a windows computer please specify your project path as Konrad, if not as fruechtnicht
os.chdir(project_dir)

#### Input data preprocessing => creating training and test set

#Organize cropped files
#organize_cropped_files(project_dir)#<-------------execute only once after you moved UTKFace folder to your project_dir


## Parameters
target_size = 128
batch_size = 32
target =      'ethnic'          # 'ethnic' or 'age' or 'gender'
color_mode = 'rgb'              #  'grayscale'
class_mode = 'categorical' #'sparse'      # 'binary','categorical, 'other'

# target = ['ethnic', 'gender']

## Training data

#cropped setup:
path1 = project_dir / 'UTKFace'
train_df = prepare_input_data(path1, 18)
#non-cropped setup:
#path1 = project_dir / 'part1' # see who easy we can join paths? no need for anything extra regardless your operating system!
#train_df = prepare_input_data(path1, 10000)

train_datagen = create_trainingDataGenerator_instance()
training_set = create_set(train_datagen, train_df, path1, target_size, batch_size, target, color_mode, class_mode)

## Test data

#cropped setup:
path3 = project_dir / 'UTKFace_test'
test_df = prepare_input_data(path3, 46)
#non-cropped setup:
#path3 = project_dir / 'part3'
#test_df = prepare_input_data(path3, 3000)

test_datagen = create_testingDataGenerator_instance()
test_set = create_set(test_datagen, test_df, path3, target_size, batch_size, target, color_mode, class_mode)


# #### Build model
#
#
#
# params = {
#     'kernel_size': 3,
#     'stride': 1,
#     'pooling_size': 2,
#     'padding': "same",
#     'nr_of_channel': 32,
#     'pooling_type': 'Max',
#     'number_of_convPool_layer': 2,
#     'dropout_rate': 0.4,
#     'activation_function': 'relu',
#     'input_size': target_size,
#     'hidden_neurons': 128,
#     'color_scale': 'rgb',
# }
#
# model = CnnSolver(class_mode, 'model_fancy_6')
# model.build_model(params)
#
#
# #### Load Model
#
# # model = CnnSolver(class_mode, 'model_fancy_6')
# # model.load_model()
#
# #### Train Model
#
#
# nr_of_epochs = 10
# steps_per_epoch = 20
#
# model.train(training_set, test_set,  nr_of_epochs, steps_per_epoch, iFcallbacks=True, do_plots=True)
#
#
# #### Make prediction
#
#
# prediction, path = make_new_prediction(model.model, target, target_size)
# plot_new_pred(prediction, path)
#

# Comparing 3 different archiecture

# 2x Con => Max // 3x Con => Max // 2x Con => Con => Max

params = {
    'kernel_size': 3,
    'stride': 1,
    'pooling_size': 2,
    'padding': "same",
    'nr_of_channel': 32,
    'pooling_type': 'Max',
    'number_of_convPool_layer': 5,
    'dropout_rate': 0.4,
    'activation_function': 'relu',
    'input_size': target_size,
    'hidden_neurons': 256,
    'color_scale': 'rgb',
}

model = CnnSolver(class_mode, 'model_fancy_2lay')
model.build_model(params)

nr_of_epochs = 2
steps_per_epoch = 5

model.train(training_set, test_set,  nr_of_epochs, steps_per_epoch, iFcallbacks=True, do_plots=False)


# params = {
#     'kernel_size': 3,
#     'stride': 1,
#     'pooling_size': 2,
#     'padding': "same",
#     'nr_of_channel': 64,
#     'pooling_type': 'Max',
#     'number_of_convPool_layer': 4,
#     'dropout_rate': 0.4,
#     'activation_function': 'relu',
#     'input_size': target_size,
#     'hidden_neurons': 1024,
#     'color_scale': 'rgb',
# }
#
# model = CnnSolver(class_mode, 'model_fancy_3lay')
# model.build_model(params)
#
#
# nr_of_epochs = 20
# steps_per_epoch = 100
#
# model.train(training_set, test_set,  nr_of_epochs, steps_per_epoch, iFcallbacks=True, do_plots=False)


#### Load Model

# nr_of_epochs = 20
# steps_per_epoch = 100
#
# model = CnnSolver(class_mode, 'model_fancy_2lay')
# model.load_model()
# model.train(training_set, test_set,  nr_of_epochs, steps_per_epoch, iFcallbacks=True, do_plots=False)

#### Train Model


# nr_of_epochs = 1#5
# steps_per_epoch = 2#21
#
# model.train(training_set, test_set,  nr_of_epochs, steps_per_epoch, iFcallbacks=True, do_plots=False)


#### Make prediction


prediction, path = make_new_prediction(model.model, target, target_size, cropped=True)
plot_new_pred(prediction, path)



