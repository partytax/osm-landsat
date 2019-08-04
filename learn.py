from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
import numpy as np
from PIL import Image
from sklearn.preprocessing import MinMaxScaler, minmax_scale
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.optimizers import SGD
import json
import os
import matplotlib.pyplot as plt


def prepare(grid_path='data/grid/grid.npy', train_folder='data/train', validate_folder='data/validate', test_folder='data/test', scale=64):
    """
    File all usable data in arrays. Split into train and test sets.
    """

    print('\nPREPARING DATA FOR MODELING')

    #instantiate label and sample lists
    X = []
    y = []
    print('label and sample lists instantiated')

    #iterate through geographic grid and file data in lists
    grid = np.load(grid_path, allow_pickle=True)
    for row in range(grid.shape[0]):
        for tile in range(grid.shape[1]):
            if grid[row][tile].get('status') == 4:
                X.append(np.array(Image.open(grid[row][tile].get('prepared_path'))))
                y.append(grid[row][tile].get('building_category'))
    print('grid data copied to lists for modeling')

    #convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    print('lists converted to numpy arrays')

    #one-hot encode labels
    y = to_categorical(y)

    #specify single frequency band
    X = np.reshape(X, (X.shape[0],scale,scale,1))

    #min/max scale images
    scaled_X = X / X.max()
    print('samples scaled between zero and one')

    #split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.1)
    X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.1)
    print('data split into training and testing sets')

    #define file paths for training label and sample arrays
    X_train_path = train_folder + '/' + 'X.npy'
    np.save(X_train_path, X_train)
    y_train_path = train_folder + '/' + 'y.npy'
    np.save(y_train_path, y_train)

    #define file paths for validation label and sample arrays
    X_validate_path = validate_folder + '/' + 'X.npy'
    np.save(X_validate_path, X_validate)
    y_validate_path = validate_folder + '/' + 'y.npy'
    np.save(y_validate_path, y_validate)

    #define file paths for test label and sample arrays
    X_test_path = test_folder + '/' + 'X.npy'
    np.save(X_test_path, X_test)
    y_test_path = test_folder + '/' + 'y.npy'
    np.save(y_test_path, y_test)
    print('numpy arrays saved to disk')

    print('COMPLETED DATA PREPARATION FOR MODELING')


def model(epochs=10, model_folder='data/models', train_folder='data/train', validate_folder='data/validate', test_folder='data/test', scale=64, output_categories=4):
    """
    Model relationship between satellite imagery and osm building counts.
    """

    print('\nDEFINING MODEL AND FITTING DATA')

    #load train label and sample data and shuffle it with ImageDataGenerator
    X_train = np.load(train_folder + '/' + 'X.npy', allow_pickle=True)
    y_train = np.load(train_folder + '/' + 'y.npy', allow_pickle=True)
    train_generator = ImageDataGenerator().flow(X_train, y_train, batch_size=X_train.shape[0])
    X_train_batch, y_train_batch = next(train_generator)

    #load validate label and sample data and shuffle it with ImageDataGenerator
    X_validate = np.load(validate_folder + '/' + 'X.npy', allow_pickle=True)
    y_validate = np.load(validate_folder + '/' + 'y.npy', allow_pickle=True)
    validate_generator = ImageDataGenerator().flow(X_validate, y_validate, batch_size=X_validate.shape[0])
    X_validate_batch, y_validate_batch = next(validate_generator)

    #load test label and sample data and shuffle it with ImageDataGenerator
    X_test = np.load(test_folder + '/' + 'X.npy', allow_pickle=True)
    y_test = np.load(test_folder + '/' + 'y.npy', allow_pickle=True)
    test_generator = ImageDataGenerator().flow(X_test, y_test, batch_size=X_test.shape[0])
    X_test_batch, y_test_batch = next(test_generator)
    print('data loaded')

    #define sequential TensorFlow model
    model = Sequential()
    #first convolutional layer
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(scale,scale,1)))
    model.add(MaxPooling2D((2, 2)))
    #second convolutional layer
    model.add(Conv2D(64, (4, 4), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    #two dense layers to mix things up
    model.add(Dense(27, activation='sigmoid'))
    model.add(Dense(54, activation='relu'))
    #third convolutional layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((3, 3)))
    #dropout layer to prevent overfit
    model.add(Dropout(0.25))
    #experimental dense layer
    model.add(Dense(70, activation='tanh'))
    #fourth convolutional layer
    model.add(Conv2D(64, (2, 2), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    #flattening layer for output
    model.add(Flatten())
    #final dense layers
    model.add(Dense(32, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    print('model defined')
    print(model.summary())

    #compile model with categorical loss function
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    print('model compiled')

    #fit and validate model
    #started with batch size of 32, but switched to 64 after StackExchange advice.
    history = model.fit(X_train_batch,
                    y_train_batch,
                    epochs=epochs,
                    batch_size=48,
                    validation_data=(X_validate_batch, y_validate_batch))

    #store model results
    #model_data = {'model':model, 'history':history}
    #model_name = input('enter name for model: ')
    #model_path = model_folder + '/' + model_name
    #with open(model_path, 'w') as f:
    #    json.dump(model_data, f)
    print('model fitted')

    #extract relevant metrics from model history
    training_accuracy = history.history['acc']
    validation_accuracy = history.history['val_acc']
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']
    epochs = range(len(training_accuracy))

    #plot data
    plt.plot(epochs, training_accuracy, label='Training Accuracy')
    plt.plot(epochs, validation_accuracy, label='Validation Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()

    plt.figure()
    plt.plot(epochs, training_loss, label='Training Loss')
    plt.plot(epochs, validation_loss, label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.show()

    print('MODELING COMPLETE')

def stats(model_folder='data/models'):
    pass

def test():
    pass
