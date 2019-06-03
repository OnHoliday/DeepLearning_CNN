
# import the necessary packages
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

# source: https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/?fbclid=IwAR1CTwo64m-YJW4L6L2ab9v2Jm0xajlFrdCj16HF8N7pILWp8WKQLB9WgZI

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

        # construct both sub-networks
        inputs = Input(shape=inputShape)
        gender_branch = Group12Net.build_gender_branch(inputs, numGender, finalAct=finalAct, chanDim=chanDim)
        race_branch = Group12Net.build_race_branch(inputs, numRace, finalAct=finalAct, chanDim=chanDim)

        # create the model using our input (the batch of images) and
        # two separate outputs for the gender and the race branch
        model = Model(
            inputs=inputs,
            outputs=[gender_branch, race_branch],
            name="Group12Net")

        # return the constructed network architecture
        return model

