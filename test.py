from prophet_gridsearch import GridProphetCV
import pandas as pd
import yaml

data = pd.read_csv('./docs/tutorial/datasets/dataset.csv', parse_dates=['ds'])

with open('./docs/tutorial/datasets/model_config.yaml') as file:
    hyperparameters = yaml.load(file, Loader=yaml.FullLoader)

holidays = pd.read_csv('./docs/tutorial/datasets/holidays.csv', parse_dates=['ds'])


m = GridProphetCV(param_grid=hyperparameters, horizon='30 days', initial='365 days', period='180 days',
                  holidays=holidays, return_train_score=True, n_jobs=-1, verbose=True)
m.fit(df=data)

best_model = m.best_model(metric='mae')

params = m._grid_df.parameters.values[3]
selected_model = m.get_model(parameters=params)

best_params = m.best_params(metric='mae')
best_score = m.best_score(metric='mae')

cross_validation = m.cv_results
train_results = m.train_results

a = best_model.cv_metrics
b = selected_model.train_metrics
