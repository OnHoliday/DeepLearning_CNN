import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_accuracy(history):
    # Plot training & validation loss values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

def plot_new_pred(prediction, path):
    # path1 = r'D:\DeepLearningProject\part2\24_0_1_20170113133945657.jpg'
    img = mpimg.imread(path)
    plt.imshow(img)
    plt.title('This is a: ' + prediction)
    plt.show()