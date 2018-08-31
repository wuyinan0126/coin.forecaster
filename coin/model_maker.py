import argparse

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import GRU, CuDNNGRU
from keras.callbacks import CSVLogger, ModelCheckpoint
import h5py
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from coin import C, get_feature_size, get_file_name


class ModelMaker:
    def __init__(self, model_opts, trade_data_opts):
        self.model_opts = C[model_opts]
        self.trade_data_opts = C[trade_data_opts]

        file_name = get_file_name(self.trade_data_opts, self.model_opts)
        self.table_name = file_name
        self.h5_file_path = os.path.join(C['h5_dir'], file_name + '.h5')
        self.model_name_prefix = file_name + '_' + self.model_opts['arch']

        self.model = self.make(self.model_opts, self.trade_data_opts)

    @staticmethod
    def make(model_opts, trade_data_opts):
        """ 初始化模型 """

        def init_env():
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            set_session(tf.Session(config=config))

        def make_gru():
            model = Sequential()
            model.add(GRU(units=model_opts['units'],
                          input_shape=(model_opts['input_size'], get_feature_size(trade_data_opts)),
                          return_sequences=False))
            model.add(Activation('tanh'))
            model.add(Dropout(0.2))
            model.add(Dense(model_opts['output_size']))
            model.add(Activation('relu'))
            model.compile(loss='mse', optimizer='adam')
            return model

        init_env()
        # noinspection PyCallingNonCallable
        return locals().get('make_' + model_opts['arch'])()

    def train(self):
        """ 从头开始训练模型 """

        def train_gru():
            self.model.fit(
                training_data, training_labels,
                batch_size=self.model_opts['batch_size'],
                validation_data=(validation_data, validation_labels),
                epochs=self.model_opts['epochs'],
                callbacks=[
                    CSVLogger(os.path.join(C['logs_dir'], self.model_name_prefix + '.csv'), append=True),
                    ModelCheckpoint(
                        os.path.join(C['weights_dir'], self.model_name_prefix + '_{epoch:02d}_{val_loss:.5f}.hdf5'),
                        monitor='val_loss', verbose=1, mode='min'
                    )
                ])

        with h5py.File(self.h5_file_path, 'r') as hf:
            # inputs: [n*input_size*feature_size], labels: [n*output_size*1]
            inputs = hf['inputs'].value
            labels = hf['outputs'].value

        data_size = inputs.shape[0]
        step_size = inputs.shape[1]
        assert step_size == self.model_opts['input_size']
        features_size = inputs.shape[2]
        assert features_size == get_feature_size(self.trade_data_opts)

        training_size = int(0.8 * data_size)
        # training_data: [(n*0.8)*input_size*feature_size]
        training_data = inputs[:training_size, :]
        # training_labels: [(n*0.8)*output_size]
        training_labels = labels[:training_size, :, 0]
        validation_data = inputs[training_size:, :]
        validation_labels = labels[training_size:, :, 0]

        # noinspection PyCallingNonCallable
        locals().get('train_' + self.model_opts['arch'])()


if __name__ == '__main__':
    # python model_maker.py --trade-data-opts poloniex_btc_opts --model-opts gru_opts
    parser = argparse.ArgumentParser('data_maker', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--trade-data-opts', type=str, default=C['trade_data_opts'])
    parser.add_argument('--model-opts', type=str, default=C['model_opts'])
    args = parser.parse_args()

    maker = ModelMaker(model_opts=args.model_opts, trade_data_opts=args.trade_data_opts)
    maker.train()
