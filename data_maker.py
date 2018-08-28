from datetime import datetime

import h5py
from sklearn.preprocessing import MinMaxScaler
import json
import pandas as pd
import numpy as np

import time

import urllib.request


class DataMaker:
    def __init__(self, pair='USDT_BTC', start_date='20180101', period=300):
        """
        :param pair: poloniex中的货币对
        :param start_date: 数据开始采集时间，"%Y%m%d"
        :param period: 数据采集周期，默认为每5分钟
        """
        start_time = time.mktime(datetime.strptime(start_date, "%Y%m%d").timetuple())

        self.url = 'https://poloniex.com/public?' \
                   'command=returnChartData' \
                   '&start={start_time}&end=9999999999' \
                   '&period={period}&currencyPair={pair}' \
            .format(start_time=start_time, period=period, pair=pair)
        self.csv_file_path = 'data/csv/{pair}_{start_date}_{period}.csv'.format(
            pair=pair, start_date=start_date, period=period
        )
        self.h5_file_path = 'data/h5/{pair}_{start_date}_{period}.h5'.format(
            pair=pair, start_date=start_date, period=period
        )

    def collect(self):
        """ 从poloniex中收集原始数据，存到csv """
        proxy = urllib.request.ProxyHandler({'http': 'http://127.0.0.1:1087', 'https': 'http://127.0.0.1:1087'})
        opener = urllib.request.build_opener(proxy, urllib.request.HTTPHandler)
        urllib.request.install_opener(opener)
        data = urllib.request.urlopen(self.url).read()
        df = pd.DataFrame(json.loads(data))
        ori_columns = ['close', 'date', 'high', 'low', 'open']
        new_columns = ['Close', 'Timestamp', 'High', 'Low', 'Open']
        df = df.loc[:, ori_columns]
        df.columns = new_columns
        df.to_csv(self.csv_file_path, index=None)

    def transform(self, input_size=256, output_size=16):
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

        inputs, outputs = self.__transform(input_size, output_size, data, False)
        input_times, output_times = self.__transform(input_size, output_size, time_stamps, False)
        ori_inputs, ori_outputs = self.__transform(input_size, output_size, ori_data, False)

        with h5py.File(self.h5_file_path, 'w') as f:
            f.create_dataset("inputs", data=inputs)
            f.create_dataset('outputs', data=outputs)
            f.create_dataset("input_times", data=input_times)
            f.create_dataset('output_times', data=output_times)
            f.create_dataset("ori_data", data=np.array(ori_data))
            f.create_dataset('ori_inputs', data=ori_inputs)
            f.create_dataset('ori_outputs', data=ori_outputs)

    def __transform(self, input_size, output_size, data, sliding_window=True):
        # Number of samples per row (sample + target)
        row_size = input_size + output_size
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
        ci = input_size * data.shape[1]
        # Sample matrix, Target matrix
        return matrix[:, :ci], matrix[:, ci:]


if __name__ == '__main__':
    maker = DataMaker(pair='USDT_BTC', start_date='20180827', period=300)
    maker.collect()
    maker.transform(input_size=256, output_size=16)
