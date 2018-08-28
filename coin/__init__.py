# P = '/Users/wuyinan/Projects/se/projects/coin.forecaster/'
P = '/home/wuyinan/Desktop/coin.forecaster/'

C = {
    # ----------------------------- data_maker -----------------------------
    'csv_dir': P + 'data/datasets/csv/',
    'h5_dir': P + 'data/datasets/h5/',
    'period': 5,  # 数据采样周期(min)
    'input_size': 256,  # 输入数据周期 = period * input_size ~ 21.3h
    'output_size': 16,  # 输出数据周期 = period * output_size ~ 1.3h

    # ----------------------------- model_maker -----------------------------
    'models_dir': P + 'data/models/',
    'logs_dir': P + 'data/logs/',
    # ----------- GRU -----------
    'gru_opts': {'arch': 'gru', 'units': 50, 'batch_size': 8, 'epochs': 100, }

}
