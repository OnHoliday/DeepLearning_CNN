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
else:
    raise ValueError('Check you own user name and add proper elif statement !!!')
# if you have a windows computer please specify your project path as Konrad, if not as fruechtnicht
os.chdir(project_dir)


#### Input data preprocessing => creating training and test set


## Parameters
target_size = 128
batch_size = 32
target = 'ethnic'          # 'ethnic' or 'age' or 'gender'
color_mode = 'rgb'              #  'grayscale'
class_mode = 'categorical'      # 'binary'

params_to_tune={'kernel_size':[2,3],
                'stride': [1,2,3],
                'nr_of_channel': [48,64,78],
                'hidden_neurons': 1024,
                }

df = pd.DataFrame
for parameter, values in params_to_tune:

    params = {
        'kernel_size': 3,  #
        'stride': 1,  #
        'pooling_size': 2,
        'padding': "same",
        'nr_of_channel': 64,  #
        'pooling_type': 'Max',
        'number_of_convPool_layer': 4,
        'dropout_rate': 0.4,
        'activation_function': 'relu',
        'input_size': target_size,
        'hidden_neurons': 1024,  #
        'color_scale': 'rgb',
    }

    for value in values:
        params[parameter] = value

        path1 = project_dir / 'part1' # see who easy we can join paths? no need for anything extra regardless your operating system!
        train_df = prepare_input_data(path1, 10000)
        # print(train_df[['ethnic', 'gender']].head())
        train_datagen = create_trainingDataGenerator_instance()
        training_set = create_set(train_datagen, train_df, path1, target_size, batch_size, target, color_mode, class_mode)

        ## Test data

        path3 = project_dir / 'part3'
        test_df = prepare_input_data(path3, 3000)
        test_datagen = create_testingDataGenerator_instance()
        test_set = create_set(test_datagen, test_df, path3, target_size, batch_size, target, color_mode, class_mode)

        modelName = target + '_' + parameter + value
        model = CnnSolver(class_mode, modelName)
        model.build_model(params)

        #### Train Model

        nr_of_epochs = 5
        steps_per_epoch = 21

        model.train(training_set, test_set,  nr_of_epochs, steps_per_epoch, iFcallbacks=True, do_plots=True)


#### Make prediction


prediction, path = make_new_prediction(model.model, target, target_size)
plot_new_pred(prediction, path)


