from sklearn.metrics import (
    r2_score,
    mean_squared_error,
)


def mean_squared_errors(Y_test, Y_pred):
    return mean_squared_error(Y_test, Y_pred) / 5
