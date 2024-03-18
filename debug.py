from backend.Supervised import Regressor
from backend.Preprocess import preprocess
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
)
from Super import mean_squared_errors
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# df = pd.read_csv("Datasets\\titanic.csv", encoding="latin-1")
# df = Preprocess.preprocess(df)
df = pd.read_csv("Datasets\\CarPrice_Assignment.csv", encoding="latin-1")
# pd.options.mode.use_inf_as_na = True
# df.dropna(how="all", inplace=True)

df = preprocess(df)
X = df.iloc[:, :-1]
Y = df.iloc[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)


# df.to_csv("Output.csv", index=False)
reg = Regressor(verbose=0, ignore_warnings=False, custom_metric=None)
models_train, predictions_train = reg.fit(X_train, X_train, Y_train, Y_train)
models_test, predictions_test = reg.fit(X_train, X_test, Y_train, Y_test)


def adjusted_rsquared(r2, n, p):
    return 1 - (1 - r2) * ((n - 1) / (n - p - 1))


rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, Y_train)
Y_pred = rf.predict(X_test)

r_squared = r2_score(Y_test, Y_pred)
print("R squared error = ", r_squared)
rmse = np.sqrt(mean_squared_errors(Y_test, Y_pred))
print("Root mean square error = ", rmse)
