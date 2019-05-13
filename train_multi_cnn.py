# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")

# import the necessary packages
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from MULTI_CNN import Group12Net
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os


# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 50
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (96, 96, 3)



# initialize our FashionNet multi-output network
model = Group12Net.build(96, 96, numGender=2, numRace=5, finalAct="softmax")

# define two dictionaries: one that specifies the loss method for
# each output of the network along with a second dictionary that
# specifies the weight per loss
losses = {
    "gender_output": "binary_crossentropy",
    "race_output": "categorical_crossentropy",
}
lossWeights = {"gender_output": 1.0, "race_output": 1.0}

# initialize the optimizer and compile the model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights, metrics=["accuracy"])



## Parameters
target_size = 64
batch_size = 32
target = 'ethnic'               # 'gender' or 'age'
color_mode = 'rgb'              #  'grayscale'
class_mode = 'categorical'      # 'binary'

## Training data

path1 = get_current_directory() + '\part1\\'
train_df = prepare_input_data(path1, 2000)
train_datagen = create_trainingDataGenerator_instance()
training_set = create_set(train_datagen, train_df, path1, target_size, batch_size, target, color_mode, class_mode)

## Test data

path3 = get_current_directory() + '\part3\\'
test_df = prepare_input_data(path3, 600)
test_datagen = create_testingDataGenerator_instance()
test_set = create_set(test_datagen, test_df, path3, target_size, batch_size, target, color_mode, class_mode)




def generate_data_generator(generator, X, Y1, Y2):
    genX = generator.flow(X, seed=7)
    genY1 = generator.flow(Y1, seed=7)
    while True:
            Xi = genX.next()
            Yi1 = genY1.next()
            Yi2 = function(Y2)
            yield Xi, [Yi1, Yi2]

flow(x, y=None, batch_size=32, shuffle=True, sample_weight=None, seed=None, save_to_dir=None, save_prefix='', save_format='png', subset=None)

model.fit_generator(generate_data_generator(generator, X, Y1, Y2), epochs=EPOCHS)




# train the network to perform multi-output classification
H = model.fit(trainX,
              {"category_output": trainCategoryY, "color_output": trainColorY},
              validation_data=(testX,
                               {"category_output": testCategoryY, "color_output": testColorY}),
              epochs=EPOCHS,
              verbose=1)

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# save the category binarizer to disk
print("[INFO] serializing category label binarizer...")
f = open(args["categorybin"], "wb")
f.write(pickle.dumps(categoryLB))
f.close()

# save the color binarizer to disk
print("[INFO] serializing color label binarizer...")
f = open(args["colorbin"], "wb")
f.write(pickle.dumps(colorLB))
f.close()

# plot the total loss, category loss, and color loss
lossNames = ["loss", "category_output_loss", "color_output_loss"]
plt.style.use("ggplot")
(fig, ax) = plt.subplots(3, 1, figsize=(13, 13))

# loop over the loss names
for (i, l) in enumerate(lossNames):
    # plot the loss for both the training and validation data
    title = "Loss for {}".format(l) if l != "loss" else "Total loss"
    ax[i].set_title(title)
    ax[i].set_xlabel("Epoch #")
    ax[i].set_ylabel("Loss")
    ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
    ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + l],
               label="val_" + l)
    ax[i].legend()

# save the losses figure and create a new figure for the accuracies
plt.tight_layout()
plt.savefig("{}_losses.png".format(args["plot"]))
plt.close()

# create a new figure for the accuracies
accuracyNames = ["category_output_acc", "color_output_acc"]
plt.style.use("ggplot")
(fig, ax) = plt.subplots(2, 1, figsize=(8, 8))

# loop over the accuracy names
for (i, l) in enumerate(accuracyNames):
    # plot the loss for both the training and validation data
    ax[i].set_title("Accuracy for {}".format(l))
    ax[i].set_xlabel("Epoch #")
    ax[i].set_ylabel("Accuracy")
    ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
    ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + l],
               label="val_" + l)
    ax[i].legend()

# save the accuracies figure
plt.tight_layout()
plt.savefig("{}_accs.png".format(args["plot"]))
plt.close()