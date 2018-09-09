import logging

import os

P = '/home/wuyinan/Desktop/coin.forecaster/' if os.path.exists(
    '/home/wuyinan/Desktop/coin.forecaster/') else '/Users/wuyinan/Projects/se/projects/coin.forecaster/'

C = {
    # ----------------------------- args -----------------------------
    'model_opts': 'gru_opts',
    'trade_data_opts': 'bitfinex_btc_opts',
    # ----------------------------- data_maker -----------------------------
    'csv_dir': P + 'data/datasets/csv/',
    'h5_dir': P + 'data/datasets/h5/',
    'db_path': P + 'data/db/coin.sqlite3',
    'scaler_dir': P + 'data/scalers/',
    # ----------- trade_data_opts: btc -----------
    'poloniex_btc_opts': {
        'name': 'poloniex',
        'pair': 'USDT_BTC',
        'start_date': '20150101',
        'period': 5,  # 数据采样周期(min), 可以为5, 15, 30, 120, 240, 1440
        'features': ['close', 'volume']
    },
    'bitfinex_btc_opts': {
        'name': 'bitfinex',
        'pair': 'tBTCUSD',
        'start_date': '20150101',
        'period': 60,  # 数据采样周期(min), 可以为1m, 5m, 15m, 30m, 1h, 3h, 6h, 12h, 1D, 7D, 14D, 1M
        'limit': 1000,  # 每次请求最大为1000
    },
    # ----------------------------- model_maker -----------------------------
    'weights_dir': P + 'data/weights/',
    'logs_dir': P + 'data/logs/',
    'db_dir': P + 'data/db/',
    # https://github.com/jealous/stockstats
    'features': [
        'close', 'high', 'low', 'open', 'volume',
        # volume delta against previous day
        'volume_delta',
        # CR indicator, including 5, 10, 20 days moving average
        'cr', 'cr-ma1', 'cr-ma2', 'cr-ma3',
        # KDJ, default to 9 days
        'kdjk', 'kdjd', 'kdjj',
        # MACD, MACD signal line, MACD histogram
        'macd', 'macds', 'macdh',
        # bolling, including upper band and lower band
        'boll', 'boll_ub', 'boll_lb',
        # 6 days RSI, 12 days RSI
        'rsi_6', 'rsi_12',
        # 10 days WR, # 6 days WR
        'wr_6', 'wr_10',
        # CCI, default to 14 days, 20 days CCI
        'cci', 'cci_20',
        # TR (true range), ATR (Average True Range)
        'tr', 'atr',
        # DMA, difference of 10 and 50 moving average
        'dma',
        # DMI: +DI, default to 14 days; -DI, default to 14 days; DX, default to 14 days of +DI and -DI; ADX, 6 days SMA of DX; ADXR, 6 days SMA of ADX
        'pdi', 'mdi', 'dx', 'adx', 'adxr',
        # TRIX, default to 12 days; MATRIX is the simple moving average of TRIX
        'trix', 'trix_9_sma',
        # VR, default to 26 days; MAVR is the simple moving average of VR
        'vr', 'vr_6_sma',
    ],
    # ----------- model_opts: gru -----------
    'gru_opts': {
        'arch': 'gru',
        'input_size': 32,  # 输入数据周期 = period * input_size ~ 21.3h
        'output_size': 4,  # 输出数据周期 = period * output_size ~ 1.3h
        'units': 50,
        'batch_size': 8,
        'epochs': 100
    },
    # ----------------------------- forecaster -----------------------------
    'ui_data': P + 'coin_ui/data/'
}


def get_file_name(trade_data_opts, model_opts):
    file_name = '{name}_{pair}_{start_date}_p{period}_i{input_size}_o{output_size}_f{features_size}'.format(
        name=trade_data_opts['name'],
        pair=trade_data_opts['pair'],
        start_date=trade_data_opts['start_date'][2:],
        period=trade_data_opts['period'],
        input_size=model_opts['input_size'],
        output_size=model_opts['output_size'],
        features_size=len(C['features']),
    ).lower()
    return file_name


def __init__():
    def set_logger(log_file_path=None, checkpoint=None):
        logger.setLevel(logging.INFO)
        log_format = logging.Formatter('%(asctime)s: [ %(message)s ]', '%Y/%m/%d %H:%M:%S')
        console = logging.StreamHandler()
        console.setFormatter(log_format)
        logger.addHandler(console)
        if log_file_path:
            if checkpoint:
                file = logging.FileHandler(log_file_path, 'a')
            else:
                file = logging.FileHandler(log_file_path, 'w')
            file.setFormatter(log_format)
            logger.addHandler(file)

    logger = logging.getLogger()
    set_logger()


__init__()
