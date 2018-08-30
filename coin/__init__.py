import logging

P = '/home/wuyinan/Desktop/coin.forecaster/'
# P = '/Users/wuyinan/Projects/se/projects/coin.forecaster/'

C = {
    # ----------------------------- data_maker -----------------------------
    'csv_dir': P + 'data/datasets/csv/',
    'h5_dir': P + 'data/datasets/h5/',
    'db_path': P + 'data/db/coin.sqlite3',
    'scaler_dir': P + 'data/scalers/',
    # ----------- trade_data_opts: btc -----------
    'btc_opts': {
        'pair': 'USDT_BTC',
        'api': 'https://poloniex.com/public?command=returnChartData&start={start_time}&end=9999999999&period={period}&currencyPair={pair}',
        'start_date': '20150101',
        'period': 5,  # 数据采样周期(min), 可以为5, 15, 30, 120, 240, 1440
        'features': ['close', 'volume']
    },
    # ----------------------------- model_maker -----------------------------
    'weights_dir': P + 'data/weights/',
    'logs_dir': P + 'data/logs/',
    'db_dir': P + 'data/db/',
    # ----------- model_opts: gru -----------
    'gru_opts': {
        'arch': 'gru',
        'input_size': 256,  # 输入数据周期 = period * input_size ~ 21.3h
        'output_size': 16,  # 输出数据周期 = period * output_size ~ 1.3h
        'units': 50,
        'batch_size': 8,
        'epochs': 100,
    },

}


def get_feature_size(trade_data_opts=C['btc_opts']):
    return len(trade_data_opts['features'])


def get_file_name(trade_data_opts, model_opts):
    file_name = '{pair}_{start_date}_p{period}_i{input_size}_o{output_size}_f{features_size}'.format(
        pair=trade_data_opts['pair'],
        start_date=trade_data_opts['start_date'][2:],
        period=trade_data_opts['period'],
        input_size=model_opts['input_size'],
        output_size=model_opts['output_size'],
        features_size=get_feature_size(),
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
