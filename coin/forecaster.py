import logging
import os
from time import sleep

import numpy as np
import pandas as pd

from datetime import datetime, timedelta

from sklearn.externals import joblib

from coin import C, get_file_name
from data_maker import DataMaker
from db_maker import DbMaker
from model_maker import ModelMaker


class Forecaster:
    def __init__(self, model_opts, trade_data_opts, weights_file_name):
        self.model_opts = C[model_opts]
        self.trade_data_opts = C[trade_data_opts]

        file_name = get_file_name(self.trade_data_opts, self.model_opts)
        self.scaler_dir = os.path.join(C['scaler_dir'], file_name)
        self.table_name = file_name

        self.model = ModelMaker.make(self.model_opts, self.trade_data_opts)
        self.model.load_weights(os.path.join(C['weights_dir'], weights_file_name))

    def forecast(self):
        while True:
            try:
                # 获取input_size个实时数据
                real_time_df = self.__get_realtime_data()
                # 生成output_size个预测close价格
                predicts = self.__forecast_realtime(real_time_df)
                # input_size个实时数据和output_size个预测close价格入库
                self.__update_table(real_time_df, predicts)
                # 从数据库生成展示用的tsv文件
                self.__dump_table()
            except:
                logging.error('Get realtime trade data failed!')
                raise
            # 休眠period时间
            sleep(60)

    def __forecast_realtime(self, realtime_df):
        """
        :param realtime_df:             input_size * ['close', 'date', 'high', 'low', 'open', 'volume']
        :return: inverted_predicts:     output_size
        """
        # features_df: [input_size*feature_size]
        features_df = realtime_df.loc[:, self.trade_data_opts['features']]
        # 归一化，最大为1
        for feature in features_df:
            scaler = joblib.load(os.path.join(self.scaler_dir, feature + '.scaler'))
            # df[column]: [input_size*1]
            features_df[feature] = scaler.transform(realtime_df[feature].values.reshape(-1, 1))

        # inputs: [input_size*feature_size] => [1*input_size*feature_size]
        inputs = np.array(features_df)[None, :, :]
        # scalered_predicted: [1*output_size]
        scalered_predicts = self.model.predict(inputs)

        scaler = joblib.load(os.path.join(self.scaler_dir, 'close.scaler'))
        # scalered_predicted: [1*output_size]
        inverted_predicts = scaler.inverse_transform(scalered_predicts[:, :])
        # [output_size]
        return inverted_predicts[0, :]

    def __get_realtime_data(self):
        period = self.trade_data_opts['period']
        input_size = self.model_opts['input_size']
        start_time = (datetime.now() - timedelta(minutes=(input_size + 16) * period)).strftime('%s')
        df = DataMaker.get_trade_data(self.trade_data_opts, start_time)

        df = df.loc[len(df) - input_size:, :]
        assert len(df) == input_size
        return df

    def __update_table(self, realtime_df, predicts):
        realtime_df = realtime_df.loc[:, ['date', 'close']]
        realtime_df['date'] = realtime_df['date'].apply(lambda x: datetime.fromtimestamp(x))
        realtime_df['close_forecast'] = ''
        realtime_df.columns = ['time', 'close', 'close_forecast']

        latest_time = realtime_df.tail(1)['time'].values[0]
        logging.info(
            'Latest time of trade is {}'.format(pd.to_datetime(str(latest_time)).strftime('%Y-%m-%d %H:%M:%S')))
        period = self.trade_data_opts['period']
        # 计算real_time_df最后一个时间之后output_size个周期
        # 返回格式为: [[time1,'',predict1], [time2,'',predict2]...]
        predicts_data = [
            [latest_time + np.timedelta64(((x + 1) * period), 'm'), '', predicts[x]]
            for x in range(0, self.model_opts['output_size'])
        ]
        close_forecast_df = pd.DataFrame(predicts_data, columns=['time', 'close', 'close_forecast'])

        # 将real_time_df最后一个时间之后output_size个周期的预测数据append到df中
        # df = df.append(pd.DataFrame(predicts_data, columns=['time', 'close', 'close_forecast']), ignore_index=True)
        DbMaker.update_table(self.table_name, realtime_df, close_forecast_df)

    def __dump_table(self):
        DbMaker.dump_table(self.table_name)

    def __update_model(self):
        pass


if __name__ == '__main__':
    # forecaster = Forecaster(
    #     model_opts='gru_opts',
    #     trade_data_opts='poloniex_btc_opts',
    #     weights_file_name='poloniex_usdt_btc_150101_p5_i256_o16_f2_gru_52_0.00014.hdf5'
    # )
    forecaster = Forecaster(
        model_opts='gru_opts',
        trade_data_opts='bitfinex_btc_opts',
        weights_file_name='bitfinex_tbtcusd_150101_p5_i256_o16_f2_gru_61_0.00040.hdf5'
    )
    forecaster.forecast()
