import argparse

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import GRU, CuDNNGRU
from keras.callbacks import CSVLogger, ModelCheckpoint
import h5py
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from coin import C


class ModelMaker:
    def __init__(self):
        self._init_env()

    def _init_env(self):
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

    # def update(
    #         self, hdf5_file_name
    # ):
    #     model = Sequential()
    #     model.add(GRU(units=units, activation=None, input_shape=(step_size, nb_features), return_sequences=False))
    #     model.add(Activation('tanh'))
    #     model.add(Dropout(0.2))
    #     model.add(Dense(output_size))
    #     model.add(Activation('relu'))
    #     model.load_weights(os.path.join(C['models_dir'], hdf5_file_name))

    def train(self, h5_file_name, model_opts=C['gru_opts']):
        with h5py.File(os.path.join(C['h5_dir'], h5_file_name), 'r') as hf:
            inputs = hf['inputs'].value
            labels = hf['outputs'].value
        model_file_name = os.path.splitext(h5_file_name)[0] + '_' + model_opts.get('arch')

        if model_opts.get('arch') == 'gru':
            self.train_gru(model_opts, inputs, labels, model_file_name)
        elif model_opts.get('arch') == 'lstm':
            pass

    def train_gru(self, model_opts, inputs, labels, model_file_name):
        data_size = inputs.shape[0]
        step_size = inputs.shape[1]
        features_size = inputs.shape[2]

        training_size = int(0.8 * data_size)
        training_data = inputs[:training_size, :]
        training_labels = labels[:training_size, :, 0]
        validation_data = inputs[training_size:, :]
        validation_labels = labels[training_size:, :, 0]

        model = Sequential()
        model.add(GRU(units=model_opts.get('units'), input_shape=(step_size, features_size), return_sequences=False))
        model.add(Activation('tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(C['output_size']))
        model.add(Activation('relu'))
        model.compile(loss='mse', optimizer='adam')
        model.fit(
            training_data, training_labels,
            batch_size=model_opts.get['batch_size'],
            validation_data=(validation_data, validation_labels),
            epochs=model_opts.get['epochs'],
            callbacks=[
                CSVLogger(os.path.join(C['logs_dir'], model_file_name + '.csv'), append=True),
                ModelCheckpoint(
                    os.path.join(C['models_dir'], model_file_name + '_{epoch:02d}_{val_loss:.5f}.hdf5'),
                    monitor='val_loss', verbose=1, mode='min'
                )
            ])


if __name__ == '__main__':
    maker = ModelMaker()
    maker.train(h5_file_name='USDT_BTC_180101_5_256_16.h5', model_opts=C['gru_opts'])
