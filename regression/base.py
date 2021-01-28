import xgboost as xg
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error as mse, r2_score as r2, mean_absolute_error as mae,\
    mean_squared_log_error as msle


class RegressionModel:

    def __init__(self, data, dependent_variable, test_size=0.2):
        self.model = None
        self.prediction = None
        self.dependent_variable = data[dependent_variable]
        self.explanatory_variables = data.loc[:, data.columns != dependent_variable]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.explanatory_variables, self.dependent_variable, test_size=test_size, random_state=0, shuffle=False
        )

    def fit_cv(self, grid, cv=5):
        self.build()
        self.model = GridSearchCV(self.model, grid, cv=cv).fit(self.x_train, self.y_train).best_estimator_
        print(self.model)

    def build(self, *args, **kwargs):
        raise NotImplementedError()

    def fit(self):
        raise NotImplementedError()

    def predict(self):
        raise NotImplementedError()

    @staticmethod
    def results(x, y):
        return {
            'RMSE': round(np.sqrt(mse(x, y)), 3),
            'R2': round(r2(x, y), 3),
            'MSE': round(mse(x, y), 3),
            'MAE': round(mae(x, y), 3)
        }


class RegressionStack:

    def __init__(self):
        pass

    @staticmethod
    def run(list_of_models, method='linear'):
        if method == 'linear':
            df_train = pd.DataFrame.from_dict({i: model.predict(test=False) for i, model in enumerate(list_of_models)})
            df_test = pd.DataFrame.from_dict({i: model.predict(test=True) for i, model in enumerate(list_of_models)})
            lin_model = linear_model.Lasso(alpha=0)
            lin_model.fit(df_train, list_of_models[0].y_train)
            return RegressionModel.results(lin_model.predict(df_train), list_of_models[0].y_train),\
                RegressionModel.results(lin_model.predict(df_test), list_of_models[0].y_test)


class XGBoost(RegressionModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, *args, **kwargs):
        self.model = xg.XGBRegressor(*args, **kwargs)

    def fit(self):
        self.model.fit(self.x_train, self.y_train)

    def predict(self, test=True):
        if test:
            self.prediction = self.model.predict(self.x_test)
            return self.prediction
        return self.model.predict(self.x_train)


class DecisionTree(RegressionModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, *args, **kwargs):
        self.model = DecisionTreeRegressor(*args, **kwargs)

    def fit(self):
        self.model.fit(self.x_train, self.y_train)

    def predict(self, test=True):
        if test:
            self.prediction = self.model.predict(self.x_test)
            return self.prediction
        return self.model.predict(self.x_train)

    def plot(self):
        plot_tree(self.model)


class RandomForest(RegressionModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, *args, **kwargs):
        self.model = RandomForestRegressor(*args, **kwargs)

    def fit(self):
        self.model.fit(self.x_train, self.y_train)

    def predict(self, test=True):
        if test:
            self.prediction = self.model.predict(self.x_test)
            return self.prediction
        return self.model.predict(self.x_train)

    def plot(self):
        plot_tree(self.model)


class Lasso(RegressionModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, *args, **kwargs):
        self.model = linear_model.Lasso(*args, **kwargs)

    def fit(self):
        self.model.fit(self.x_train, self.y_train)

    def predict(self, test=True):
        if test:
            self.prediction = self.model.predict(self.x_test)
            return self.prediction
        return self.model.predict(self.x_train)


class Ridge(RegressionModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, *args, **kwargs):
        self.model = linear_model.Ridge(*args, **kwargs)

    def fit(self):
        self.model.fit(self.x_train, self.y_train)

    def predict(self, test=True):
        if test:
            self.prediction = self.model.predict(self.x_test)
            return self.prediction
        return self.model.predict(self.x_train)

