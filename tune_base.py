import os
from utils import *
from cnn import CnnSolver
from plots import plot_new_pred
import getpass
from pathlib import Path, PureWindowsPath # please check this medium article!! https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f



project_dir = get_path()

#### Input data preprocessing => creating training and test set


## Parameters
target_size = 128
batch_size = 32
target = 'gender'          # 'ethnic' or 'age' or 'gender'
color_mode = 'rgb'              #  'grayscale'
class_mode = 'binary'      # 'categorical', 'other', 'binary'


for i in range(3):#average over 3 runs

    path1 = project_dir / 'UTKFace'  # see who easy we can join paths? no need for anything extra regardless your operating system!
    train_df = prepare_input_data(path1, 18966)
    # print(train_df[['ethnic', 'gender']].head())
    train_datagen = create_trainingDataGenerator_instance()
    training_set = create_set(train_datagen, train_df, path1, target_size, batch_size, target, color_mode, class_mode)

    ## Test data

    path3 = project_dir / 'UTKFace_test'
    test_df = prepare_input_data(path3, 4694)
    test_datagen = create_testingDataGenerator_instance()
    test_set = create_set(test_datagen, test_df, path3, target_size, batch_size, target, color_mode, class_mode)

    df = pd.DataFrame

    params = {
        'kernel_size': 3,
        'stride': 1,
        'pooling_size': 2,
        'padding': "same",
        'nr_of_channel': 32,
        'pooling_type': 'Max',
        'number_of_convPool_layer': 4,
        'dropout_rate': 0.3,
        'activation_function': 'relu',
        'input_size': target_size,
        'hidden_neurons': 1024,
        'color_scale': 'rgb',
    }

    params2 = {
        'kernel_size': 3,
        'stride': 1,
        'pooling_size': 2,
        'padding': "same",
        'nr_of_channel': 32,
        'pooling_type': 'Max',
        'number_of_convPool_layer': 2,
        'dropout_rate': 0.3,
        'activation_function': 'relu',
        'input_size': target_size,
        'hidden_neurons': 1024,
        'color_scale': 'rgb',
    }

    params3 = {
        'kernel_size': 3,
        'stride': 1,
        'pooling_size': 2,
        'padding': "same",
        'nr_of_channel': 32,
        'pooling_type': 'Max',
        'number_of_convPool_layer': 3,
        'dropout_rate': 0.3,
        'activation_function': 'relu',
        'input_size': target_size,
        'hidden_neurons': 1024,
        'color_scale': 'rgb',
    }

    modelName = 'testing' + '_' + target + '_ConvConvPool_' + str(i)
    model1 = CnnSolver(class_mode, modelName)
    model1.build_model_cnn_cnn_pool(params3)

    modelName = 'testing' + '_' + target + '__4ConvPool_' + str(i)
    model2 = CnnSolver(class_mode, modelName)
    model2.build_model(params)

    modelName = 'testing' + '_' + target + '__2ConvPool__' + '_' + str(i)
    model3 = CnnSolver(class_mode, modelName)
    model3.build_model(params2)
    #### Train Model

    nr_of_epochs = 30
    steps_per_epoch = 50

    model1.train(training_set, test_set,  nr_of_epochs, steps_per_epoch, iFcallbacks=True, do_plots=False)
    model2.train(training_set, test_set,  nr_of_epochs, steps_per_epoch, iFcallbacks=True, do_plots=False)
    model3.train(training_set, test_set,  nr_of_epochs, steps_per_epoch, iFcallbacks=True, do_plots=False)






