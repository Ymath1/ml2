import datetime
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense

from classification.base import ImageClassifier


class KerasImageClassifier(ImageClassifier):

    def __init__(self, path, img_size, batch_size=16, train_data_gen=None, epochs=20, categorical=True):
        super(KerasImageClassifier, self).__init__()
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = Sequential()
        self.epochs = epochs
        if categorical:
            self.type = 'categorical'
        else:
            self.type = 'binary'
        self.train, self.valid, self.test = self.read_data(path, train_data_gen)

    def read_data(self, path, train_data_gen):
        if train_data_gen is None:
            train_data_gen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=40
            )
        test_data_gen = ImageDataGenerator(rescale=1. / 255)
        train_generator = train_data_gen.flow_from_directory(
            'classification/data/train',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode=self.type)
        validation_generator = test_data_gen.flow_from_directory(
            'classification/data/valid',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode=self.type,
            shuffle=False)
        test_generator = test_data_gen.flow_from_directory(
            'classification/data/test',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode=self.type,
            shuffle=False)
        return train_generator, validation_generator, test_generator

    def load_model(self, path):
        self.model = load_model(path+'.h5')
        try:
            self.history = pickle.load(open(f'{path}_history', "rb"))
        except:
            pass
        try:
            self.evaluation = pickle.load(open(f'{path}_evaluation', "rb"))
        except:
            pass

    def fit(self, save_results=True, results_dir='trained_models', model_name='model', save=True, date=False):
        fit = self.model.fit_generator(
            self.train,
            epochs=self.epochs,
            validation_data=self.valid,
            verbose=0
        )
        if date:
            model_name += f'_{datetime.datetime.now().strftime("%H_%M_%S_%m_%d_%Y")}'
        try:
            if save:
                with open(results_dir+f'/{model_name}_history', 'wb') as file_pi:
                    pickle.dump(self.model.history.history, file_pi)
        except:
            pass
        self.model.save(results_dir+f'/{model_name}.h5')
        return fit

    def evaluate(self, gen_data=None, save=True):
        if gen_data is None:
            evaluation = self.model.evaluate_generator(self.test)
        else:
            evaluation = self.model.evaluate_generator(gen_data)
        try:
            if save:
                with open(results_dir + f'/{model_name}_evaluation', 'wb') as file_pi:
                    pickle.dump(self.model.history, file_pi)
        except:
            pass
        return evaluation

    def predict(self, prob=False, gen_data=None):
        if gen_data is None:
            predictions = self.model.predict_generator(self.test, workers=0)
        else:
            predictions = self.model.predict_generator(gen_data, workers=0)
        if prob:
            return predictions
        return [[1 if max(small_list) == x else 0 for x in small_list] for small_list in predictions]

    def compile(self):
        raise NotImplementedError('Method must be override in derived classes')

    def build(self):
        raise NotImplementedError('Method must be override in derived classes')

    def plot(self):
        history = self.model.history.history
        plt.plot(history['acc'])
        plt.plot(history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()


class SimpleCNN(KerasImageClassifier):

    def __init__(self, *args, **kwargs):
        super(SimpleCNN, self).__init__(*args, **kwargs)

    def build(self):
        model = Sequential()
        model.add(Conv2D(64, (3, 3), input_shape=(self.img_size[0], self.img_size[1], 3), padding='same', activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(4))
        model.add(Activation('softmax'))

        self.model = model

    def compile(self):
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )


class CNN(KerasImageClassifier):

    def __init__(self, *args, **kwargs):
        super(CNN, self).__init__(*args, **kwargs)

    def build(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(self.img_size[0], self.img_size[1], 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2), padding='same'))

        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2), padding='same'))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2), padding='same'))
        model.add(Flatten())

        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(4))
        model.add(Activation('softmax'))

        self.model = model

    def compile(self):
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )


class NN(KerasImageClassifier):

    def __init__(self, *args, **kwargs):
        super(NN, self).__init__(*args, **kwargs)

    def build(self):
        model = Sequential()
        model.add(Flatten(input_shape=(self.img_size[0], self.img_size[1], 3)))
        model.add(Dense(300, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(4, activation='softmax'))

        self.model = model

    def compile(self):
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
