from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from multivariate import *
from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True)

print(X.shape, y.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)


regressor = LinearRegression(learning_rate=1e-1, epochs=10000)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
sm_mse = mean_squared_error(y_test, y_pred)
sm_mae = mean_absolute_error(y_test, y_pred)

# _______________ COMPARING WITH scikit-learn Linear Regression model ______


from sklearn import linear_model

skReg = linear_model.LinearRegression()
skReg.fit(X_train, y_train)
sk_y_pred = skReg.predict(X_test)
sk_mse = mean_squared_error(y_test, sk_y_pred)
sk_mae = mean_absolute_error(y_test, sk_y_pred)

# _______________ MSE between both models __________________________________

print(f"Scratch model MSE: {sm_mse}")
print(f"Scratch model MAbsE: {sm_mae}")
print(f"Sklearn model MSE: {sk_mse}")
print(f"Sklearn model MAbsE: {sk_mae}")
