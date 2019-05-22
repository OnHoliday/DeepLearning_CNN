import os
from utils import *
from MULTI_CNN import *
from cnn import CnnSolver
from plots import plot_new_pred
import getpass
from pathlib import Path, PureWindowsPath # please check this medium article!! https://medium.com/@ageitgey/python-3-quick-tip-the-easy-way-to-deal-with-file-paths-on-windows-mac-and-linux-11a072b58d5f
from keras import backend as K
import pickle


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
target_size = 64
batch_size = 64
im_width = im_height = 198


# target = ['ethnic', 'gender']

## Training data

path = project_dir / 'crop_part1' # see who easy we can join paths? no need for anything extra regardless your operating system!
df = prepare_input_data(path, 10000)
n_races = len(df['ethnic'].unique())
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.3)

train_gen = data_generator_cust(train,im_width,im_width, True, path, batch_size)
test_gen = data_generator_cust(test,im_width,im_width, True, path, batch_size)





## building the model ##

model = built_multi(n_races, im_width)
model.summary()
from keras.callbacks import ModelCheckpoint



csv_logger = create_cv_logger()
callbacks = [csv_logger]

#K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_‌​parallelism_threads=‌​32, inter_op_parallelism_threads=32)))


model.fit_generator(train_gen,
                              steps_per_epoch=len(train) // batch_size,
                              epochs=10,
                              callbacks=callbacks,
                              workers=6,
                              use_multiprocessing=True,
                              validation_data=test_gen,
                              validation_steps=len(test) // batch_size)


save_model(model, 'lukas_multi_crop.h5')

# model = load_model_('lukas_multi.h5')

prediction, path = make_new_prediction_multi(model)
plot_new_pred(prediction, path)
