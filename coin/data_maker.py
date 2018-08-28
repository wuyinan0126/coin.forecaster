import argparse
from datetime import datetime

import h5py
from sklearn.preprocessing import MinMaxScaler
import json
import pandas as pd
import numpy as np

import time
import os

import urllib.request

from coin import C


class DataMaker:
    def __init__(
            self, pair, start_date,
            period=C['period'], input_size=C['input_size'], output_size=C['output_size']
    ):
        """
        :param pair: poloniex中的货币对
        :param start_date: 数据开始采集时间，"%Y%m%d"
        :param period: 数据采集周期，默认为每5分钟
        :param input_size: 输入数据周期 = period * input_size ~ 21.3h
        :param output_size: 输出数据周期 = period * output_size ~ 1.3h
        """
        start_time = time.mktime(datetime.strptime(start_date, "%Y%m%d").timetuple())
        file_name = '{pair}_{start_date}_{period}_{input_size}_{output_size}'.format(
            pair=pair, start_date=start_date[2:], period=period, input_size=input_size, output_size=output_size
        )

        self.url = 'https://poloniex.com/public?' \
                   'command=returnChartData' \
                   '&start={start_time}&end=9999999999' \
                   '&period={period}&currencyPair={pair}' \
            .format(start_time=start_time, period=period * 60, pair=pair)

        self.csv_file_path = os.path.join(C['csv_dir'], file_name + '.csv')
        self.h5_file_path = os.path.join(C['h5_dir'], file_name + '.h5')

        self.input_size = input_size
        self.output_size = output_size

    def collect(self):
        """ 从poloniex中收集原始数据，存到csv """
        proxy = urllib.request.ProxyHandler(
            # {'http': 'http://127.0.0.1:1087', 'https': 'http://127.0.0.1:1087'}
        )
        opener = urllib.request.build_opener(proxy, urllib.request.HTTPHandler)
        urllib.request.install_opener(opener)
        data = urllib.request.urlopen(self.url).read()
        df = pd.DataFrame(json.loads(data))
        ori_columns = ['close', 'date', 'high', 'low', 'open']
        new_columns = ['Close', 'Timestamp', 'High', 'Low', 'Open']
        df = df.loc[:, ori_columns]
        df.columns = new_columns
        df.to_csv(self.csv_file_path, index=None)

    def transform(self):
        """ 从csv中读取原始数据，经过变换存到h5，如用前input_size个价格预测之后的output_size个价格 """
        columns = ['Close']

        df = pd.read_csv(self.csv_file_path)
        time_stamps = df['Timestamp']
        df = df.loc[:, columns]
        ori_df = pd.read_csv(self.csv_file_path).loc[:, columns]

        scaler = MinMaxScaler()
        # normalization
        for c in columns:
            df[c] = scaler.fit_transform(df[c].values.reshape(-1, 1))

        data = np.array(df)[:, None, :]
        ori_data = np.array(ori_df)[:, None, :]
        time_stamps = np.array(time_stamps)[:, None, None]

        inputs, outputs = self._transform(data, False)
        input_times, output_times = self._transform(time_stamps, False)
        ori_inputs, ori_outputs = self._transform(ori_data, False)

        with h5py.File(self.h5_file_path, 'w') as f:
            f.create_dataset("inputs", data=inputs)
            f.create_dataset('outputs', data=outputs)
            f.create_dataset("input_times", data=input_times)
            f.create_dataset('output_times', data=output_times)
            f.create_dataset("ori_data", data=np.array(ori_data))
            f.create_dataset('ori_inputs', data=ori_inputs)
            f.create_dataset('ori_outputs', data=ori_outputs)

    def _transform(self, data, sliding_window=True):
        # Number of samples per row (sample + target)
        row_size = self.input_size + self.output_size
        # indexes
        if sliding_window:
            indices = np.arange(row_size) + np.arange(data.shape[0] - row_size + 1).reshape(-1, 1)
        else:
            if data.shape[0] % row_size == 0:
                indices = np.arange(row_size) + np.arange(0, data.shape[0], row_size).reshape(-1, 1)

            else:
                indices = np.arange(row_size) + np.arange(0, data.shape[0] - row_size, row_size).reshape(-1, 1)

        matrix = data[indices].reshape(-1, row_size * data.shape[1], data.shape[2])
        # Number of features per sample
        ci = self.input_size * data.shape[1]
        # Sample matrix, Target matrix
        return matrix[:, :ci], matrix[:, ci:]


if __name__ == '__main__':
    parser = argparse.ArgumentParser('data_maker', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pair', type=str, default='USDT_BTC')
    parser.add_argument('--start-date', type=str, default='20180101')
    parser.add_argument('--period', type=int, default=C['period'])
    parser.add_argument('--input-size', type=int, default=C['input_size'])
    parser.add_argument('--output-size', type=int, default=C['output_size'])
    args = parser.parse_args()

    maker = DataMaker(
        pair=args.pair, start_date=args.start_date,
        period=args.period, input_size=args.input_size, output_size=args.output_size
    )
    maker.collect()
    maker.transform()
