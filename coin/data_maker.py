import argparse
import logging
import sqlite3
from datetime import datetime

import h5py
from sklearn.preprocessing import MinMaxScaler
import json
import pandas as pd
import numpy as np

import os

import urllib.request

from coin import C, get_file_name


class DataMaker:
    def __init__(self, trade_data_opts='btc_opts', model_opts='gru_opts'):
        self.trade_data_opts = C[trade_data_opts]
        self.model_opts = C[model_opts]

        file_name = get_file_name(self.trade_data_opts, self.model_opts)
        self.csv_file_path = os.path.join(C['csv_dir'], file_name + '.csv')
        self.h5_file_path = os.path.join(C['h5_dir'], file_name + '.h5')
        self.table_name = file_name

    @staticmethod
    def get_trade_data(trade_data_opts, start_time):
        api = trade_data_opts['api'].format(
            start_time=start_time,
            period=trade_data_opts['period'] * 60,
            pair=trade_data_opts['pair']
        )
        logging.info('Getting trade data from {}'.format(api))
        proxy = urllib.request.ProxyHandler(
            {'http': 'http://127.0.0.1:1087', 'https': 'http://127.0.0.1:1087'}
        )
        opener = urllib.request.build_opener(proxy, urllib.request.HTTPHandler)
        urllib.request.install_opener(opener)

        data = urllib.request.urlopen(api).read()
        df = pd.DataFrame(json.loads(data))
        return df

    def collect(self):
        """ 从poloniex中收集原始数据，存到csv """
        logging.info('Raw trade data collecting...')

        start_time = datetime.strptime(self.trade_data_opts['start_date'], "%Y%m%d").strftime('%s')
        df = self.get_trade_data(self.trade_data_opts, start_time)

        columns = ['close', 'date', 'high', 'low', 'open', 'volume']
        df = df.loc[:, columns]
        df.to_csv(self.csv_file_path, index=None)
        logging.info('Raw trade data saved in {}'.format(self.csv_file_path))

    def transform(self):
        """ 从csv中读取原始数据，经过变换存到h5，如用前input_size个价格预测之后的output_size个价格 """
        features = self.trade_data_opts['features']

        df = pd.read_csv(self.csv_file_path)
        # 获取时间列
        time_stamps = df['date']
        # 获取close价格
        df = df.loc[:, features]
        ori_df = pd.read_csv(self.csv_file_path).loc[:, features]

        scaler = MinMaxScaler()
        # 归一化，最大为1
        for feature in features:
            # df[column]: [n*1]
            df[feature] = scaler.fit_transform(df[feature].values.reshape(-1, 1))

        # [n*feature_size] => [n*1*feature_size], n是数据数，feature_size是使用的列数如只用一列'Close'
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
            f.create_dataset("ori_data", data=np.array(ori_data))
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

    def make_db(self):
        def init_data():
            df = pd.read_csv(self.csv_file_path, index_col=None)
            df = df[['date', 'close']]
            # unix timestamp转datetime
            df['date'] = df['date'].apply(lambda x: datetime.fromtimestamp(x))
            df['forecast'] = ''
            df.columns = ['time', 'truth', 'forecast']
            df.to_sql(name=self.table_name, con=conn, if_exists='replace', chunksize=1000, index=False)

        conn = sqlite3.connect(C['db_path'])
        init_data()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('data_maker', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--trade-data-opts', type=str, default='btc_opts')
    parser.add_argument('--model-opts', type=str, default='gru_opts')
    args = parser.parse_args()

    maker = DataMaker(trade_data_opts=args.trade_data_opts, model_opts=args.model_opts)
    # maker.collect()
    # maker.transform()
    # maker.make_db()
