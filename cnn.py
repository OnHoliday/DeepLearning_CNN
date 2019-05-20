
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from utils import *
from plots import plot_accuracy, plot_loss

class CnnSolver():
    def __init__(self, problem_type, model_name):
        self.problem_type = problem_type
        self.model_name = model_name


    def unzip_parameters(self, kernel_size, stride, pooling_size, padding, nr_of_channel, pooling_type,
                         number_of_convPool_layer, dropout_rate, activation_function, input_size, hidden_neurons, color_scale):
        self.kernel_size = kernel_size
        self.stride = stride
        self.pooling_size = pooling_size
        self.padding = padding
        self.nr_of_channel = nr_of_channel
        self.pooling_type = pooling_type
        self.number_of_convPool_layer = number_of_convPool_layer
        self.dropout_rate = dropout_rate
        self.activation_function = activation_function
        self.input_size = input_size
        self.hidden_neurons = hidden_neurons
        self.color_scale = color_scale


    def build_model(self, params):
        self.unzip_parameters(**params)
        if self.color_scale == 'grayscale':
            dimmesionality = 1
        else:
            dimmensionality = 3

        classifier = Sequential()

        iteration = self.number_of_convPool_layer
        while iteration > 0:
            classifier.add(
                Conv2D(self.nr_of_channel, (self.kernel_size, self.kernel_size), strides=(self.stride, self.stride),
                       padding=self.padding, input_shape=(self.input_size, self.input_size, dimmensionality),
                       activation=self.activation_function))
            # classifier.add(MaxPooling2D(pool_size=(2, 2)))
            classifier.add(MaxPooling2D(pool_size=(self.pooling_size, self.pooling_size)))
            iteration -= 1

        classifier.add(Flatten())
        classifier.add(Dense(units=self.hidden_neurons, activation=self.activation_function))
        if self.problem_type == 'binary':
            classifier.add(Dense(units=1, activation='sigmoid'))
            classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        elif self.problem_type == 'regression':
            classifier.add(Dense(units=1, activation='sigmoid'))
            classifier.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        else:
            classifier.add(Dense(units=5, activation='softmax'))
            classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        print(classifier.summary())
        self.model = classifier


    def load_model(self):
        self.model = load_model(self.model_name)
        self.model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        print(self.model.summary())


    def _save_model(self):
        save_model(self.model, self.model_name)


    def train(self, training_set, test_set,  nr_of_epochs, steps_per_epoch, iFcallbacks=False, do_plots=False):

        if do_plots :
            if not iFcallbacks:
                raise ValueError('Wrong parameters: if wanna make plots iFcallbacks mast be True but is : ' + str(iFcallbacks))

        if iFcallbacks:
            csv_logger = create_cv_logger()
            history = callback_history()
            earlyStop = callbackEarlyStopping()
            checkPoint = callbackCheckpoint(self.model_name)
            tensor = callbackTensor()
            callbacks = [csv_logger, history, earlyStop, checkPoint, tensor]

            history = self.model.fit_generator(training_set,
                                     steps_per_epoch=steps_per_epoch,
                                     epochs=nr_of_epochs,
                                     validation_data=test_set,
                                     validation_steps=steps_per_epoch/3,
                                     #  use_multiprocessing=True,
                                     workers=12,
                                     callbacks=callbacks
                                     )
        else:
            self.model.fit_generator(training_set,
                                     steps_per_epoch=steps_per_epoch,
                                     epochs=nr_of_epochs,
                                     validation_data=test_set,
                                     validation_steps=steps_per_epoch/3,
                                     #  use_multiprocessing=True,
                                     workers=12,
                                     )
        print('Training is Done!!')

        if do_plots:
            plot_accuracy(history)
            plot_loss(history)

        self._save_model()
        print('Model is saved')

# cnn.add(conv.ZeroPadding2D( (1,1), input_shape = (1,28,28), ))


### additional parameters for keras model:
# activation = None,
# use_bias = True,
# kernel_initializer = 'glorot_uniform',
# bias_initializer = 'zeros',
# kernel_regularizer = None,
# bias_regularizer = None,
# activity_regularizer = None,
# kernel_constraint = None,
# bias_constraint = None

# from keras import backend as K
# from keras.layers import Layer
#
# class MyLayer():#Layer):
#
#     def __init__(self, output_dim, **kwargs):
#         self.output_dim = output_dim
#         super(MyLayer, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         # Create a trainable weight variable for this layer.
#         self.kernel = self.add_weight(name='kernel',
#                                       shape=(input_shape[1], self.output_dim),
#                                       initializer='uniform',
#                                       trainable=True)
#         super(MyLayer, self).build(input_shape)  # Be sure to call this at the end
#
#     def call(self, x):
#         return K.dot(x, self.kernel)
#
#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], self.output_dim)
#
# def min_max_pool2d(x):
#     max_x =  K.pool2d(x, pool_size=(2, 2), strides=(2, 2))
#     min_x = -K.pool2d(-x, pool_size=(2, 2), strides=(2, 2))
#     return K.concatenate([max_x, min_x], axis=1) # concatenate on channel
#
# def min_max_pool2d_output_shape(input_shape):
#     shape = list(input_shape)
#     shape[1] *= 2
#     shape[2] /= 2
#     shape[3] /= 2
#     return tuple(shape)
#
# # replace maxpooling layer
# # cnn.add(Lambda(min_max_pool2d, output_shape=min_max_pool2d_output_shape))