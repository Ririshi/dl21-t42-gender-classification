#!/usr/bin/env python3

import argparse
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from datetime import date
from glob import glob
from sklearn.metrics import classification_report
from tensorflow.keras import Sequential, layers, optimizers
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from typing import Tuple


class GenderClassifier:

    def __init__(self, data_path, plot_path, epochs, batch_size):
        self.data_path = data_path
        self.plot_path = plot_path
        self.epochs = epochs
        self.batch_size = batch_size

        matplotlib.use('Agg')

    def __load_dataset(self, part: str) -> Tuple[np.ndarray, np.ndarray]:
        data = []
        labels = []
        imagePaths = glob(f'{self.data_path}/{part}/**/*.jpg', recursive=True)
        random.seed(42)
        random.shuffle(imagePaths)

        for imagePath in imagePaths:
            image = cv2.imread(imagePath)
            image = cv2.resize(image, (60, 80))
            data.append(image)

            # Binary encode where male = 0 and female = 1
            labels.append(0 if imagePath.split(os.path.sep)[-2] == 'male' else 1)

        return np.array(data), np.array(labels)

    def __create_model(self, reg_layer: layers.Layer, opt: optimizers.Optimizer) -> tf.keras.Sequential:
        model = Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(80, 60, 3)))
        model.add(layers.MaxPooling2D((3, 3)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((3, 3)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((3, 3)))
        model.add(layers.Flatten())
        model.add(reg_layer)
        model.add(layers.Dense(256, activation='sigmoid'))
        model.add(layers.Dense(1, activation='sigmoid', name='output'))
        model.compile(
            loss='binary_crossentropy',
            optimizer=opt,
            metrics=['accuracy']
        )
        return model

    def __plot_history(self, history, plot_path):
        # plot the training loss and accuracy
        N = np.arange(0, len(history.history['loss']))
        plt.style.use('ggplot')
        plt.figure()
        plt.plot(N, history.history['loss'], label='train_loss')
        plt.plot(N, history.history['accuracy'], label='train_acc')
        if 'val_loss' in history.history:
            plt.plot(N, history.history['val_loss'], label='val_loss')
            plt.plot(N, history.history['val_accuracy'], label='val_acc')
        plt.title('Training Loss and Accuracy')
        plt.xlabel('Epoch #')
        plt.ylabel('Loss/Accuracy')
        plt.legend()
        plt.savefig(plot_path)

    def __evaluate_model(self, model: tf.keras.Sequential, x_test: np.ndarray, y_test: np.ndarray) -> None:
        print('[INFO] evaluating network')
        predictions = model.predict(x=x_test, batch_size=1024)
        print(classification_report(y_test,
            predictions.round(), target_names=['Male', 'Female']))

    def tune_model(self) -> None:
        x_train, y_train = self.__load_dataset('Training')
        model = self.__create_model(
            layers.Dropout(0.2),
            optimizers.Adam()
        )
        print('[INFO] fitting network')
        es = EarlyStopping(monitor='val_accuracy', mode='max', patience=10, min_delta=0.005)
        history = model.fit(
            x_train, y_train, validation_split=0.1,
            epochs=self.epochs, batch_size=self.batch_size,
            callbacks=[es], verbose=2
        )
        self.__plot_history(history, self.plot_path)

    def save_model(self, model_path: str) -> None:
        x_train, y_train = self.__load_dataset('Training')

        model = self.__create_model(
            layers.Dropout(0.2),
            optimizers.SGD()
        )
        print('[INFO] fitting network')
        es = EarlyStopping(monitor='val_accuracy', mode='max', patience=10, min_delta=0.005)
        history = model.fit(
            x_train, y_train, epochs=self.epochs,
            batch_size=self.batch_size, 
            callbacks=[es], verbose=2
        )
        model.save(model_path, save_format="h5")
        self.__plot_history(history, self.plot_path)

    def test_model(self, model_path: str) -> None:
        x_test, y_test = self.__load_dataset('Validation')
        self.__evaluate_model(load_model(model_path), x_test, y_test)

    def compare_models(self) -> None:
        x_train, y_train = self.__load_dataset('Training')
        x_test, y_test = self.__load_dataset('Validation')

        regs = {
            'dropout': layers.Dropout(0.2),
            'batchnorm': layers.BatchNormalization() 
        }
        opts = {
            'sgd': optimizers.SGD(),
            'sgdmom': optimizers.SGD(momentum=0.5),
            'adam': optimizers.Adam(),
            'nadam': optimizers.Nadam(),
            'rmsprop': optimizers.RMSprop()
        }
        for reg_name, reg in regs.items():
            for opt_name, opt in opts.items():
                print(f'[INFO] fitting {reg_name} {opt_name} model', flush=True)
                clear_session()
                es = EarlyStopping(monitor='val_accuracy', mode='max', patience=10, min_delta=0.005)
                history = self.__create_model(reg, opt).fit(
                    x_train, y_train, validation_data=(x_test, y_test),
                    epochs=self.epochs, batch_size=self.batch_size,
                    callbacks=[es], verbose=0
                )
                self.__plot_history(
                    history, f'{self.plot_path}/{date.today().strftime("%Y%m%d")}'
                    f'_{reg_name}_{opt_name}_{len(history.history["loss"])}-epochs.png'
                )
                print(
                    f'Validation accuracy: {history.history["val_accuracy"][-1]}\n'
                    f'Validation loss: {history.history["val_loss"][-1]}'
                )


def parse_args():
    parser = argparse.ArgumentParser(description='Gender classifier DNN')
    parser.add_argument('-d', '--data', required=True,
        help='path to the gender classification dataset')
    parser.add_argument('-p', '--plot', required=True,
	    help='path to output accuracy/loss plot')

    sp = parser.add_subparsers(title='command', dest='command')

    sp.add_parser('tune',
        help='do a tuning run using 90% of the training data and validate on the remaining 10%')

    sp.add_parser('compare',
        help='compare models using different regularization layers and optimizers')

    save = sp.add_parser('save',
        help='train and save a model')
    save.add_argument('path',
	    help='output model path')

    test = sp.add_parser('test',
        help='do a testing run')
    test.add_argument('path',
        help='existing model path')

    return parser.parse_args()


def main():
    args = parse_args()

    clf = GenderClassifier(args.data, args.plot, 50, 128)

    if args.command == 'save':
        clf.save_model(args.path)
        clf.test_model(args.path)
        return
    elif args.command == 'test':
        clf.test_model(args.path)
        return
    elif args.command == 'compare':
        clf.compare_models()
        return

    clf.tune_model()


if __name__ == '__main__':
    main()
