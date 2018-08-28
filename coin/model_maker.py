from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import GRU, CuDNNGRU
from keras.callbacks import CSVLogger, ModelCheckpoint
import h5py
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from coin import C

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class ModelMaker:
    def __init__(self, h5_file_name, use_cuda=False):
        self._init_env()
        with h5py.File(os.path.join(C['h5_dir'], h5_file_name), 'r') as hf:
            self.inputs = hf['inputs'].value
            self.labels = hf['outputs'].value
        self.model_file_name, _ = os.path.splitext(h5_file_name)

    def _init_env(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

    def make_gru(
            self,
            units=C['gru_units'],
            batch_size=C['batch_size'],
            epochs=C['epochs'],
            output_size=C['output_size']
    ):
        model_file_name = self.model_file_name + '_gru'

        data_size = self.inputs.shape[0]
        step_size = self.inputs.shape[1]
        features_size = self.inputs.shape[2]

        training_size = int(0.8 * data_size)
        training_data = self.inputs[:training_size, :]
        training_labels = self.labels[:training_size, :, 0]
        validation_datas = self.inputs[training_size:, :]
        validation_labels = self.labels[training_size:, :, 0]

        model = Sequential()
        model.add(GRU(units=units, input_shape=(step_size, features_size), return_sequences=False))
        model.add(Activation('tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(output_size))
        model.add(Activation('relu'))
        model.compile(loss='mse', optimizer='adam')
        model.fit(
            training_data, training_labels,
            batch_size=batch_size, validation_data=(validation_datas, validation_labels), epochs=epochs,
            callbacks=[
                CSVLogger(os.path.join(C['logs_dir'], model_file_name + '.csv'), append=True),
                ModelCheckpoint(
                    os.path.join(C['checkpoints_dir'], model_file_name + '_{epoch:02d}_{val_loss:.5f}.hdf5'),
                    monitor='val_loss', verbose=1, mode='min'
                )
            ])


if __name__ == '__main__':
    maker = ModelMaker('USDT_BTC_20180801_300.h5')
    maker.make_gru()
