
# import the necessary packages
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.optimizers import Adam
from keras.layers.core import Lambda
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
import tensorflow as tf

from utils import *

class Group12Net:

    def __init__(self, model_name):
        self.model_name = model_name

    def build_gender_branch(self, inputs, numGender, finalAct="softmax", chanDim=-1):

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

    def build_race_branch(self, inputs, numRace, finalAct="softmax", chanDim=-1):

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

    def build_age_branch(self, inputs, chanDim=-1):

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
        x = Dense(64)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dense(1, name="age_output", kernel_initializer='normal', activation='linear')(x)
        return x


    def build(self, width, height, numGender, numRace, finalAct="softmax"):

        # initialize the input shape and channel dimension (this code
        # assumes you are using TensorFlow which utilizes channels
        # last ordering)
        inputShape = (height, width, 3)
        chanDim = -1

        # construct both the "category" and "color" sub-networks
        inputs = Input(shape=inputShape)
        gender_branch = self.build_gender_branch(inputs, numGender, finalAct=finalAct, chanDim=chanDim)
        race_branch = self.build_race_branch(inputs, numRace, finalAct=finalAct, chanDim=chanDim)
        age_branch = self.build_age_branch(inputs, chanDim=chanDim)

        # create the model using our input (the batch of images) and
        # two separate outputs -- one for the clothing category
        # branch and another for the color branch, respectively
        self.model = Model(
            inputs=inputs,
            outputs=[gender_branch, race_branch, age_branch],
            name=self.model_name)

        self._compile_model()

        # initialize the optimizer and compile the model
        # print("[INFO] compiling model...")
        # print(self.model.summary())
        # print("[INFO] Model compiled ...")

        # return the constructed network architecture
        self.model

    def load_model(self):
        self.model = load_model(self.model_name)
        self._compile_model()
        print(self.model.summary())


    def _save_model(self):
        save_model(self.model, self.model_name)

    def _compile_model(self):
        EPOCHS = 50
        INIT_LR = 1e-3
        BS = 32
        IMAGE_DIMS = (64, 64, 3)
        losses = {
            "gender_output": "binary_crossentropy",
            "race_output": "categorical_crossentropy",
            "age_output": "mean_absolute_error",
        }
        lossWeights = {"gender_output": 1.0, "race_output": 1.0, "age_output": 1.0}
        opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

        self.model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights, metrics=["accuracy", "mean_absolute_error"])
