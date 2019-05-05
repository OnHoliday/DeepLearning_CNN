import os
from utils import *
from cnn import CnnSolver
from plots import plot_new_pred
os.chdir('D:\DeepLearningProject')


# #### Input data preprocessing => creating training and test set
#
#
# # Parameters
# target_size = 64
# batch_size = 32
# target = 'ethnic'               # 'gender' or 'age'
# color_mode = 'rgb'              #  'grayscale'
class_mode = 'categorical'      # 'binary'
#
# ## Training data
#
# path1 = get_current_directory() + '\part1\\'
# train_df = prepare_input_data(path1, 2000)
# train_datagen = create_trainingDataGenerator_instance()
# training_set = create_set(train_datagen, train_df, path1, target_size, batch_size, target, color_mode, class_mode)
#
# ## Test data
#
# path3 = get_current_directory() + '\part3\\'
# test_df = prepare_input_data(path3, 600)
# test_datagen = create_testingDataGenerator_instance()
# test_set = create_set(test_datagen, test_df, path3, target_size, batch_size, target, color_mode, class_mode)
#
#
#### Build model


# params = {
#     'kernel_size': 3,
#     'stride': 2,
#     'pooling_size': 2,
#     'padding': "valid",
#     'nr_of_channel': 32,
#     'pooling_type': 'Max',
#     'number_of_convPool_layer': 2,
#     'dropout_rate': 0.3,
#     'activation_function': 'relu',
#     'input_size': 64,
#     'hidden_neurons': 256,
#     'color_scale': 'rgb',
# }
#
# model = CnnSolver(class_mode)
# model.build_model(params)


#### Load Model


model = CnnSolver(class_mode, 'model_ethnic')
model.load_model()

#### Train Model


# nr_of_epochs = 5
# steps_per_epoch = 21
#
# model.train(training_set, test_set,  nr_of_epochs, steps_per_epoch, iFcallbacks=True, do_plots=True)


#### Make prediction


prediction, path = make_new_prediction_ethnicity(model.model)
plot_new_pred(prediction, path)



