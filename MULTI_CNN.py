
# import the necessary packages
from keras.engine.saving import save_model, load_model
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Lambda
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
import tensorflow as tf


class Group12Net:

    def build_gender_branch(inputs, numGender, finalAct="softmax", chanDim=-1):

        x = inputs

        # CONV => RELU => POOL
        x = Conv2D(32, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)
        x = Dropout(0.25)(x)

        # (CONV => RELU) * 2 => POOL
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        # define a branch of output layers for the number of different
        # clothing categories (i.e., shirts, jeans, dresses, etc.)
        x = Flatten()(x)
        x = Dense(256)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.35)(x)
        x = Dense(numGender)(x)
        x = Activation(finalAct, name="gender_output")(x)

        # return the category prediction sub-network
        return x

    def build_race_branch(inputs, numRace, finalAct="softmax", chanDim=-1):

        x = inputs
        # CONV => RELU => POOL
        x = Conv2D(16, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)
        x = Dropout(0.25)(x)

        # CONV => RELU => POOL
        x = Conv2D(32, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        # define a branch of output layers for the number of different
        # colors (i.e., red, black, blue, etc.)
        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.35)(x)
        x = Dense(numRace)(x)
        x = Activation(finalAct, name="race_output")(x)

        # return the color prediction sub-network
        return x

    def build(width, height, numGender, numRace, finalAct="softmax"):

        # initialize the input shape and channel dimension (this code
        # assumes you are using TensorFlow which utilizes channels
        # last ordering)
        inputShape = (height, width, 3)
        chanDim = -1

        # construct both the "category" and "color" sub-networks
        inputs = Input(shape=inputShape)
        gender_branch = Group12Net.build_gender_branch(inputs, numGender, finalAct=finalAct, chanDim=chanDim)
        race_branch = Group12Net.build_race_branch(inputs, numRace, finalAct=finalAct, chanDim=chanDim)

        # create the model using our input (the batch of images) and
        # two separate outputs -- one for the clothing category
        # branch and another for the color branch, respectively
        model = Model(
            inputs=inputs,
            outputs=[gender_branch, race_branch],
            name="Group12Net")

        # return the constructed network architecture
        return model



def built_multi(n_races,im_width):
    from keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPool2D, GlobalMaxPool2D, Dropout
    from keras.optimizers import SGD
    from keras.models import Model

    def conv_block(inp, filters=32, bn=True, pool=True):
        _ = Conv2D(filters=filters, kernel_size=3, activation='relu')(inp)
        if bn:
            _ = BatchNormalization()(_)
        if pool:
            _ = MaxPool2D()(_)
        return _

    input_layer = Input(shape=(im_width, im_width, 3))
    _ = conv_block(input_layer, filters=32, bn=False, pool=False)
    _ = conv_block(_, filters=32 * 2)
    _ = conv_block(_, filters=32 * 3)
    _ = conv_block(_, filters=32 * 4)
    _ = conv_block(_, filters=32 * 5)
    _ = conv_block(_, filters=32 * 6)
    bottleneck = GlobalMaxPool2D()(_)

    # for age calculation
    _ = Dense(units=128, activation='relu')(bottleneck)
    age_output = Dense(units=1, activation='sigmoid', name='age_output')(_)

    # for race prediction
    _ = Dense(units=128, activation='relu')(bottleneck)
    race_output = Dense(units=n_races, activation='softmax', name='race_output')(_)

    # for gender prediction
    _ = Dense(units=128, activation='relu')(bottleneck)
    gender_output = Dense(units=2, activation='softmax', name='gender_output')(_)

    model = Model(inputs=input_layer, outputs=[age_output, race_output, gender_output])
    model.compile(optimizer='rmsprop',
                  loss={'age_output': 'mse', 'race_output': 'categorical_crossentropy',
                        'gender_output': 'categorical_crossentropy'},
                  loss_weights={'age_output': 2., 'race_output': 1.5, 'gender_output': 1.},
                  metrics={'age_output': 'mae', 'race_output': 'accuracy', 'gender_output': 'accuracy'})

    return model

def load_model_(model_name):
    model = load_model(model_name)
    model.compile(optimizer='rmsprop',
                  loss={'age_output': 'mse', 'race_output': 'categorical_crossentropy',
                        'gender_output': 'categorical_crossentropy'},
                  loss_weights={'age_output': 2., 'race_output': 1.5, 'gender_output': 1.},
                  metrics={'age_output': 'mae', 'race_output': 'accuracy', 'gender_output': 'accuracy'})

    print(model.summary())
    return model




