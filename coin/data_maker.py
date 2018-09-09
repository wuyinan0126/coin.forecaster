import argparse
import ast
import logging
import shutil
from datetime import datetime

import h5py
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
import json
import pandas as pd
import numpy as np

import os
import time
import urllib.request

from stockstats import StockDataFrame

from coin import C, get_file_name
from db_maker import DbMaker


class DataMaker:
    def __init__(self, trade_data_opts='poloniex_btc_opts', model_opts='gru_opts'):
        self.trade_data_opts = C[trade_data_opts]
        self.model_opts = C[model_opts]

        file_name = get_file_name(self.trade_data_opts, self.model_opts)
        self.csv_file_path = os.path.join(C['csv_dir'], file_name + '.csv')
        self.h5_file_path = os.path.join(C['h5_dir'], file_name + '.h5')
        self.table_name = file_name
        self.scaler_dir = os.path.join(C['scaler_dir'], file_name)

    @staticmethod
    def get_trade_data(trade_data_opts, start_time):
        """ 从trade_data_opts['name']平台获取数据，从start_time开始到现在，unix时间戳(秒) """

        def get_data(api):
            logging.info('Getting trade data from {}'.format(api))
            proxy = urllib.request.ProxyHandler(
                # {'http': 'http://127.0.0.1:1087', 'https': 'http://127.0.0.1:1087'}
                {'http': 'http://10.2.2.153:8123', 'https': 'http://10.2.2.153:8123'}
            )
            opener = urllib.request.build_opener(proxy, urllib.request.HTTPHandler)
            urllib.request.install_opener(opener)
            data = urllib.request.urlopen(api).read()
            return data

        def get_poloniex_data():
            api = 'https://poloniex.com/public?command=returnChartData&start={start_time}&end=9999999999&period={period}&currencyPair={pair}'.format(
                start_time=start_time,
                period=trade_data_opts['period'] * 60,
                pair=trade_data_opts['pair']
            )
            data = get_data(api)
            poloniex_df = pd.DataFrame(json.loads(data))
            poloniex_df = poloniex_df.loc[:, ['close', 'date', 'high', 'low', 'open', 'volume']]
            return poloniex_df

        def get_bitfinex_data():
            period_str = {1: '1m', 5: '5m', 15: '15m', 30: '30m', 60: '1h', 180: '3h',
                          360: '6h', 720: '12h', 1440: '1D', 10080: '7D', 20160: '14D', -1: '1M'}
            period = trade_data_opts['period']
            limit = trade_data_opts['limit']

            bitfinex_df = pd.DataFrame([], columns=['CLOSE', 'MTS', 'HIGH', 'LOW', 'OPEN', 'VOLUME'])
            s = int(start_time)
            now = int(time.time())
            while s <= now and now - s > period * 60:
                e = now if now - s <= limit * period * 60 else s + limit * period * 60
                api = 'https://api.bitfinex.com/v2/candles/trade:{period}:{pair}/hist?limit={limit}&start={start_time}&end={end_time}&sort=1'.format(
                    start_time=s * 1000,
                    period=period_str[period],
                    pair=trade_data_opts['pair'],
                    end_time=e * 1000,
                    limit=limit,
                )
                # data = [[MTS, OPEN, CLOSE, HIGH, LOW, VOLUME]*100]
                data = get_data(api).decode('utf-8')
                data = ast.literal_eval(data)
                one_df = pd.DataFrame(data, columns=['MTS', 'OPEN', 'CLOSE', 'HIGH', 'LOW', 'VOLUME'])
                one_df = one_df[['CLOSE', 'MTS', 'HIGH', 'LOW', 'OPEN', 'VOLUME']]
                one_df['MTS'] = one_df['MTS'].apply(lambda x: x / 1000)
                bitfinex_df = bitfinex_df.append(one_df, sort=False)
                s = e
                now = int(time.time())
                # 防止被ban
                time.sleep(3)

            return bitfinex_df

        # noinspection PyCallingNonCallable
        df = locals().get('get_' + trade_data_opts['name'] + '_data')()
        df.columns = ['close', 'date', 'high', 'low', 'open', 'volume']
        return df

    @staticmethod
    def get_indicators(df):
        stock = StockDataFrame.retype(df)
        features = C['features']
        for feature in features:
            df[feature] = stock.get(feature)
        return df

    def collect(self):
        """ 从poloniex中收集原始数据，存到csv """
        logging.info('Raw trade data collecting...')

        start_time = datetime.strptime(self.trade_data_opts['start_date'], "%Y%m%d").strftime('%s')
        df = self.get_trade_data(self.trade_data_opts, start_time)
        date_df = df['date']
        # 计算各种指标
        df = self.get_indicators(df)
        df = df.assign(date=date_df.values)

        df.iloc[16:].to_csv(self.csv_file_path, index=None)
        logging.info('Raw trade data saved in {}'.format(self.csv_file_path))

    def transform(self):
        """ 从csv中读取原始数据，经过变换存到h5，用前input_size个价格预测之后的output_size个价格 """
        features = C['features']

        df = pd.read_csv(self.csv_file_path)
        # 获取时间列
        time_stamps = df['date']
        # 获取特征
        df = df.loc[:, features]
        ori_df = pd.read_csv(self.csv_file_path).loc[:, features]

        # 归一化
        if os.path.exists(self.scaler_dir):
            shutil.rmtree(self.scaler_dir)
        os.mkdir(self.scaler_dir)
        for feature in features:
            scaler = MinMaxScaler()
            # df[column]: [n*1]
            df[feature] = scaler.fit_transform(df[feature].values.reshape(-1, 1))
            # 保存scaler
            joblib.dump(scaler, os.path.join(self.scaler_dir, feature + '.scaler'))

        # [n*feature_size] => [n*1*feature_size], n是数据数，feature_size是使用的列数如只用一列'close'
        data = np.array(df)[:, None, :]
        ori_data = np.array(ori_df)[:, None, :]
        # [n] => [n*1*1]
        time_stamps = np.array(time_stamps)[:, None, None]

        inputs, outputs = self.__transform(data, False)
        input_times, output_times = self.__transform(time_stamps, False)
        ori_inputs, ori_outputs = self.__transform(ori_data, False)

        with h5py.File(self.h5_file_path, 'w') as f:
            f.create_dataset("inputs", data=inputs)
            f.create_dataset('outputs', data=outputs)
            f.create_dataset("input_times", data=input_times)
            f.create_dataset('output_times', data=output_times)
            f.create_dataset("ori_data", data=np.array(ori_df))
            f.create_dataset('ori_inputs', data=ori_inputs)
            f.create_dataset('ori_outputs', data=ori_outputs)
        logging.info('Transformed data saved in {}'.format(self.h5_file_path))

    def __transform(self, data, sliding_window=True):
        """
        :param sliding_window: True则数据index为[[0,1,2],[1,2,3],...], False则为[[0,1,2],[3,4,5]...]
        """
        # 每行数据大小
        row_size = self.model_opts['input_size'] + self.model_opts['output_size']
        # indexes
        if sliding_window:
            indices = np.arange(row_size) + np.arange(data.shape[0] - row_size + 1).reshape(-1, 1)
        else:
            if data.shape[0] % row_size == 0:
                indices = np.arange(row_size) + np.arange(0, data.shape[0], row_size).reshape(-1, 1)
            else:
                indices = np.arange(row_size) + np.arange(0, data.shape[0] - row_size, row_size).reshape(-1, 1)

        # matrix: [n*row_size*feature_size]
        matrix = data[indices].reshape(-1, row_size * data.shape[1], data.shape[2])
        # 每行数据中的输入数据大小
        i = self.model_opts['input_size'] * data.shape[1]
        # 输入和输出矩阵: [n*input_size*feature_size], [n*output_size*1]
        return matrix[:, :i], matrix[:, i:, :1]

    def make_table(self, load_history_trade_data=False):
        DbMaker.create_table(table_name=self.table_name)
        if load_history_trade_data:
            df = pd.read_csv(self.csv_file_path, index_col=None)
            df = df[['date', 'close']]
            # unix timestamp转datetime
            df['date'] = df['date'].apply(lambda x: datetime.fromtimestamp(x))
            df['close_forecast'] = ''
            df.columns = ['time', 'close', 'close_forecast']
            DbMaker.update_table(table_name=self.table_name, close_df=df, close_forecast_df=None)


if __name__ == '__main__':
    # python data_maker.py --trade-data-opts poloniex_btc_opts --model-opts gru_opts
    parser = argparse.ArgumentParser('data_maker', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--trade-data-opts', type=str, default=C['trade_data_opts'])
    parser.add_argument('--model-opts', type=str, default=C['model_opts'])
    args = parser.parse_args()

    maker = DataMaker(trade_data_opts=args.trade_data_opts, model_opts=args.model_opts)
    maker.collect()
    maker.transform()
    # maker.make_table(load_history_trade_data=False)
