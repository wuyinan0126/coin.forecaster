# P = '/Users/wuyinan/Projects/se/projects/coin.forecaster/'
P = '/home/wuyinan/Desktop/coin.forecaster/'

C = {
    # ----------------------------- data_maker -----------------------------
    'csv_dir': P + 'data/datasets/csv/',
    'h5_dir': P + 'data/datasets/h5/',

    # ----------------------------- model_maker -----------------------------
    'models_dir': P + 'data/models/',
    'logs_dir': P + 'data/logs/',
    'checkpoints_dir': P + 'data/checkpoints/',

    'input_size': 256,
    'output_size': 16,
    'batch_size': 8,
    'epochs': 100,
    # ----------- GRU -----------
    'gru_units': 50,
}
