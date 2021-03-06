import os
from utils import *
from cnn import CnnSolver
from plots import plot_new_pred
import getpass
from pathlib import Path, PureWindowsPath # please check this medium article!! https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f

project_dir = get_path()

#### Input data preprocessing => creating training and test set

#Organize cropped files
# organize_cropped_files(project_dir)#<-------------execute only once after you moved UTKFace folder to your project_dir

# 2531qawSZQ`1`1  azq\ XS2`
## Parameters
target_size = 128
batch_size = 32
target = 'ethnic'          # 'ethnic' or 'age' or 'gender'
color_mode = 'rgb'              #  'grayscale'
class_mode = 'categorical'  #'sparse'      # 'binary','categorical, 'other'

# target = ['ethnic', 'gender']

## Training data

#cropped setup:
# path1 = project_dir / 'UTKFace'
# train_df = prepare_input_data(path1, 18)
#non-cropped setup:
#path1 = project_dir / 'part1' # see who easy we can join paths? no need for anything extra regardless your operating system!
#train_df = prepare_input_data(path1, 10000)

# train_datagen = create_trainingDataGenerator_instance()
# training_set = create_set(train_datagen, train_df, path1, target_size, batch_size, target, color_mode, class_mode)
#
# ## Test data
#
# #cropped setup:
# path3 = project_dir / 'UTKFace_test'
# test_df = prepare_input_data(path3, 46)
# #non-cropped setup:
# #path3 = project_dir / 'part3'
# #test_df = prepare_input_data(path3, 3000)
#
# test_datagen = create_testingDataGenerator_instance()
# test_set = create_set(test_datagen, test_df, path3, target_size, batch_size, target, color_mode, class_mode)
#

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

# params = {
#     'kernel_size': 3,
#     'stride': 1,
#     'pooling_size': 2,
#     'padding': "same",
#     'nr_of_channel': 32,
#     'pooling_type': 'Max',
#     'number_of_convPool_layer': 5,
#     'dropout_rate': 0.4,
#     'activation_function': 'relu',
#     'input_size': target_size,
#     'hidden_neurons': 256,
#     'color_scale': 'rgb',
# }
#
# model = CnnSolver(class_mode, 'model_fancy_2lay')
# model.build_model(params)
#
# nr_of_epochs = 2
# steps_per_epoch = 5
#
# model.train(training_set, test_set,  nr_of_epochs, steps_per_epoch, iFcallbacks=True, do_plots=False)


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

#### ensemble AGE
model1 = CnnSolver(class_mode, 'testing_age__2ConvPool___0')
model1.load_model('mse', ['mae'])

model2 = CnnSolver(class_mode, 'testing_age__2ConvPool___1')
model2.load_model('mse', ['mae'])

model3 = CnnSolver(class_mode, 'testing_age__2ConvPool___2')
model3.load_model('mse', ['mae'])

model4 = CnnSolver(class_mode, 'testing_age__4ConvPool_0')
model4.load_model('mse', ['mae'])

model5 = CnnSolver(class_mode, 'testing_age__4ConvPool_1')
model5.load_model('mse', ['mae'])

model6 = CnnSolver(class_mode, 'testing_age__4ConvPool_2')
model6.load_model('mse', ['mae'])


models = [model1.model,model2.model, model3.model, model4.model,model5.model, model6.model]
final_preds_age(models, target, target_size)

#### ensemble GENDER
#
# model1 = CnnSolver(class_mode, 'testing_ethnic__2ConvPool_0')
# model1.load_model('mse', ['mae'])
#
# model2 = CnnSolver(class_mode, 'testing_ethnic__2ConvPool_1')
# model2.load_model('mse', ['mae'])
#
# model3 = CnnSolver(class_mode, 'testing_ethnic__2ConvPool_2')
# model3.load_model('mse', ['mae'])
#
# model4 = CnnSolver(class_mode, 'testing_ethnic__4ConvPool___0')
# model4.load_model('mse', ['mae'])
#
# model5 = CnnSolver(class_mode, 'testing_ethnic__4ConvPool___1')
# model5.load_model('mse', ['mae'])
#
# model6 = CnnSolver(class_mode, 'testing_ethnic__4ConvPool___2')
# model6.load_model('mse', ['mae'])
#
#
# models = [model1.model,model2.model, model3.model, model4.model,model5.model, model6.model]
# final_preds_race(models, target, target_size)


#### ensemble GENDER
# model1 = CnnSolver(class_mode, 'testing_gender__2ConvPool___0')
# model1.load_model('mse', ['mae'])
#
# model2 = CnnSolver(class_mode, 'testing_gender__2ConvPool___1')
# model2.load_model('mse', ['mae'])
#
# model3 = CnnSolver(class_mode, 'testing_gender__2ConvPool___2')
# model3.load_model('mse', ['mae'])
#
# model4 = CnnSolver(class_mode, 'testing_gender__4ConvPool_0')
# model4.load_model('mse', ['mae'])
#
# model5 = CnnSolver(class_mode, 'testing_gender__4ConvPool_1')
# model5.load_model('mse', ['mae'])
#
# model6 = CnnSolver(class_mode, 'testing_gender__4ConvPool_2')
# model6.load_model('mse', ['mae'])
#
# models = [model1.model,model2.model, model3.model, model4.model,model5.model, model6.model]
# final_preds_gender(models, target, target_size)

# df = pd.read_csv(r'D:\DeepLearningProject\DL_CNN\final_results_ensemble.csv')
# print(df.head())
#
# from sklearn.metrics import mean_absolute_error
# import  seaborn as sns
# import matplotlib.pyplot as plt
# print(mean_absolute_error(df['age_truth'], df['age_preds']))
# # sns.distplot([df['age_truth'], df['age_preds']])
# # plt.show()
#
# df_plot = df[['age_truth', 'age_preds']]
# df_plot.plot(kind='kde', figsize=[12, 6], alpha=.4, legend=True)
# # df['age_preds'].plot(kind='hist', bins=12, figsize=[12, 6], alpha=.4, legend=True, color='n')
# plt.show()

# df[['age_truth', 'age_preds']].plot(kind='hist', bins=12, figsize=[12, 6], alpha=.4, legend=True)  # alpha for transparency
# plt.show()


#### Train Model

# nr_of_epochs = 1#5
# steps_per_epoch = 2#21
#
# model.train(training_set, test_set,  nr_of_epochs, steps_per_epoch, iFcallbacks=True, do_plots=False)


#### Make prediction

#
# prediction, path = make_new_prediction(model.model, target, target_size, cropped=True)
# # plot_new_pred(prediction, path)
#


####################################
#####  Ensembling -> Voting    #####
####################################
#
# nr_of_epochs = 10
# steps_per_epoch = 50
#
# model1 = CnnSolver(class_mode, 'age1')
# model1.build_model(params)
# model1.train(training_set, test_set,  nr_of_epochs, steps_per_epoch, iFcallbacks=True, do_plots=False)
#
# # model1.load_model( loss = 'categorical_crossentropy', metrics = ['accuracy'])
#
# model2 = CnnSolver(class_mode, 'age2')
# model2.build_model(params)
# model2.train(training_set, test_set,  nr_of_epochs, steps_per_epoch, iFcallbacks=True, do_plots=False)
#
# # model2.load_model( loss = 'categorical_crossentropy', metrics = ['accuracy'])
#
# model3 = CnnSolver(class_mode, 'age3')
# model3.build_model(params)
# model2.train(training_set, test_set,  nr_of_epochs, steps_per_epoch, iFcallbacks=True, do_plots=False)
#
# # model3.load_model( loss = 'categorical_crossentropy', metrics = ['accuracy'])
#
# models = [model1.model,model2.model, model3.model]
#
# voting_prediction(models, target, target_size)
#

####################################
#####  Testing different architecture    #####
####################################
#
# params1 = {
#     'kernel_size': 3,
#     'stride': 1,
#     'pooling_size': 2,
#     'padding': "same",
#     'nr_of_channel': 32,
#     'pooling_type': 'Max',
#     'number_of_convPool_layer': 1,
#     'dropout_rate': 0.3,
#     'activation_function': 'relu',
#     'input_size': target_size,
#     'hidden_neurons': 256,
#     'color_scale': 'rgb',
# }
# params2 = {
#     'kernel_size': 3,
#     'stride': 1,
#     'pooling_size': 2,
#     'padding': "same",
#     'nr_of_channel': 32,
#     'pooling_type': 'Max',
#     'number_of_convPool_layer': 4,
#     'dropout_rate': 0.3,
#     'activation_function': 'relu',
#     'input_size': target_size,
#     'hidden_neurons': 256,
#     'color_scale': 'rgb',
# }
# params3 = {
#     'kernel_size': 3,
#     'stride': 1,
#     'pooling_size': 2,
#     'padding': "same",
#     'nr_of_channel': 32,
#     'pooling_type': 'Max',
#     'number_of_convPool_layer': 3,
#     'dropout_rate': 0.3,
#     'activation_function': 'relu',
#     'input_size': target_size,
#     'hidden_neurons': 256,
#     'color_scale': 'rgb',
# }
#
#
#
# nr_of_epochs = 2
# steps_per_epoch = 4
#
# model1 = CnnSolver(class_mode, 'ethnic1')
# model1.build_model(params1)
# model1.train(training_set, test_set,  nr_of_epochs, steps_per_epoch, iFcallbacks=True, do_plots=False)
#
# # model1.load_model( loss = 'categorical_crossentropy', metrics = ['accuracy'])
#
# model2 = CnnSolver(class_mode, 'ethnic2')
# model2.build_model(params2)
# model2.train(training_set, test_set,  nr_of_epochs, steps_per_epoch, iFcallbacks=True, do_plots=False)
#
# # model2.load_model( loss = 'categorical_crossentropy', metrics = ['accuracy'])
#
# model3 = CnnSolver(class_mode, 'ethnic3')
# model3.build_model_cnn_cnn_pool(params3)
# model2.train(training_set, test_set,  nr_of_epochs, steps_per_epoch, iFcallbacks=True, do_plots=False)
#
# # model3.load_model( loss = 'categorical_crossentropy', metrics = ['accuracy'])
#
# models = [model1.model,model2.model, model3.model]
#
# voting_prediction(models, target, target_size)
#
