import os
import numpy as np

from datetime import datetime, timedelta

from sklearn.externals import joblib

from coin import C, get_file_name
from data_maker import DataMaker
from model_maker import ModelMaker


class Forecaster:
    def __init__(self, model_opts, trade_data_opts, weights_file_name):
        self.model_opts = C[model_opts]
        self.trade_data_opts = C[trade_data_opts]

        file_name = get_file_name(self.trade_data_opts, self.model_opts)
        self.scaler_dir = os.path.join(C['scaler_dir'], file_name)

        self.model = ModelMaker.make(self.model_opts, self.trade_data_opts)
        self.model.load_weights(os.path.join(C['weights_dir'], weights_file_name))

    def update(self):
        def update_gru():
            pass

        # noinspection PyCallingNonCallable
        locals().get('update_' + self.model_opts['arch'])()

    def forecast(self):
        df = self.__get_real_time_data()

        # features_df: [input_size*feature_size]
        features_df = df.loc[:, self.trade_data_opts['features']]

        # 归一化，最大为1
        for feature in features_df:
            scaler = joblib.load(os.path.join(self.scaler_dir, feature + '.scaler'))
            # df[column]: [input_size*1]
            features_df[feature] = scaler.transform(df[feature].values.reshape(-1, 1))

        # inputs: [input_size*feature_size] => [1*input_size*feature_size]
        inputs = np.array(features_df)[None, :, :]
        # scalered_predicted: [1*output_size]
        scalered_predict = self.model.predict(inputs)
        inverted_predict = []

        scaler = joblib.load(os.path.join(self.scaler_dir, 'close.scaler'))
        inverted_predict.append(scaler.inverse_transform(scalered_predict[:, :]))

        print(scalered_predict, inverted_predict)

    def __get_real_time_data(self):
        period = self.trade_data_opts['period']
        input_size = self.model_opts['input_size']
        start_time = (datetime.now() - timedelta(minutes=(input_size + 16) * period)).strftime('%s')
        df = DataMaker.get_trade_data(self.trade_data_opts, start_time)

        columns = ['close', 'date', 'high', 'low', 'open', 'volume']
        df = df.loc[len(df) - input_size:, columns]
        assert len(df) == input_size
        return df


if __name__ == '__main__':
    forecaster = Forecaster(
        model_opts='gru_opts',
        trade_data_opts='btc_opts',
        weights_file_name='usdt_btc_180101_p5_i256_o16_f2_gru_15_0.00002.hdf5'
    )
    # forecaster.update()
    forecaster.forecast()
