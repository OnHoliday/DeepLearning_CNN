import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, GlobalMaxPool2D, GlobalMaxPooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam



#Preprocessing
def built_transfer(Unfreeze_some=False):

    base_model = MobileNet(weights='imagenet',include_top=False)

    _ = base_model.output
    bottleneck = GlobalMaxPooling2D()(_)

    # for age calculation
    _ = Dense(units=512, activation='relu')(bottleneck)
    _ = Dropout(0.3)(_)
    _ = Dense(units=256, activation='relu')(_)
    _ = Dropout(0.3)(_)
    _ = Dense(units=128, activation='relu')(_)
    _ = Dropout(0.3)(_)
    age_output = Dense(units=1, activation='sigmoid', name='age_output')(_)

    # for race prediction
    _ = Dense(units=256, activation='relu')(bottleneck)
    _ = Dropout(0.3)(_)
    _ = Dense(units=128, activation='relu')(_)
    _ = Dropout(0.3)(_)
    race_output = Dense(units=5, activation='softmax', name='race_output')(_)

    # for gender prediction
    _ = Dense(units=128, activation='relu')(bottleneck)
    _ = Dropout(0.3)(_)
    gender_output = Dense(units=1, activation='sigmoid', name='gender_output')(_)

    model = Model(inputs=base_model.input,outputs=[age_output, race_output, gender_output])

    if Unfreeze_some:

        for layer in model.layers[:80]:
            layer.trainable=False
        for layer in model.layers[80:]:
            layer.trainable=True

    else:
        for layer in model.layers[:88]:
            layer.trainable=False
        for layer in model.layers[88:]:
            layer.trainable=True

    model.compile(optimizer='rmsprop',
                  loss={'age_output': 'mse', 'race_output': 'categorical_crossentropy',
                        'gender_output': 'binary_crossentropy'},
                  loss_weights={'age_output': 2., 'race_output': 1.5, 'gender_output': 1.},
                  metrics={'age_output': 'mae', 'race_output': 'accuracy', 'gender_output': 'accuracy'})

    model.summary()

    return model



