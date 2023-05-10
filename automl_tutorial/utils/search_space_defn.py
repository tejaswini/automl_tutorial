from ray import tune

def cnn_search_space():
    search_space = {
        "l1": tune.randint(2,8),
        "num_filters1": tune.randint(2,8),
        "num_filters2": tune.randint(2,8),
        "num_filters3": tune.randint(2,8),
        "lr": tune.loguniform(5e-3, 1e-2),
    }
    return search_space

def one_dim_search_space():
    search_space = {
        "x": tune.uniform(0.1, 5)
    }
    return search_space


