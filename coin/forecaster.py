import os

from datetime import datetime, timedelta

from coin import C
from data_maker import DataMaker
from model_maker import ModelMaker


class Forecaster:
    def __init__(self, model_opts, trade_data_opts, weights_file_name):
        self.model_opts = C[model_opts]
        self.trade_data_opts = C[trade_data_opts]

        self.model = ModelMaker.make(self.model_opts, self.trade_data_opts)
        self.model.load_weights(os.path.join(C['weights_dir'], weights_file_name))

    def update(self):
        def update_gru():
            pass

        # noinspection PyCallingNonCallable
        locals().get('update_' + self.model_opts['arch'])()

    def forecast(self):
        def forecast_gru():
            pass

        # noinspection PyCallingNonCallable
        locals().get('forecast_' + self.model_opts['arch'])()

    def get_real_time_data(self):
        period = self.trade_data_opts['period']
        input_size = self.model_opts['input_size']
        start_time = (datetime.now() - timedelta(minutes=(input_size + 16) * period)).strftime('%s')
        df = DataMaker.get_trade_data(self.trade_data_opts, start_time)


        print(len(df))


if __name__ == '__main__':
    forecaster = Forecaster(
        model_opts='gru_opts',
        trade_data_opts='btc_opts',
        weights_file_name='usdt_btc_180101_p5_i256_o16_f2_gru_15_0.00002.hdf5'
    )
    # forecaster.update()
    forecaster.get_real_time_data()
