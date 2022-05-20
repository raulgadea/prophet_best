from multiprocessing import Pool, cpu_count, Value
import logging
from time import time as ttime
import pandas as pd
from lazy_property import LazyProperty
from sklearn.model_selection import ParameterGrid
from typing import Union

from prophet_gridsearch import ProphetDecorator
from prophet_gridsearch.utils import MetricUnavailable

logging.getLogger('fbprophet').setLevel(logging.ERROR)
logger = logging.getLogger('GridProphetCV')

counter = Value('i', 0)


class GridProphetCV:
    def __init__(
            self,
            param_grid: dict,
            horizon: str,
            initial: str = None,
            period: str = None,
            holidays: Union[pd.DataFrame, None] = None,
            scoring: str = 'rmse',
            return_train_score: bool = False,
            n_jobs: Union[int, None] = None,
            verbose: bool = True
    ):
        """
        This class is responsible for executing a grid search cross validation for Facebook prophet objects.

        Parameters
        ----------
        param_grid : dict
                Dictionary containing all possible hyper parameters combinations.
        horizon : str
                String with pd.Timedelta compatible style, e.g., '5 days', '3 hours', '10 seconds'.
        initial : str
                String with pd.Timedelta compatible style. Simulated forecast will be done at every this period.
                If not provided, 0.5 * horizon is used.
        period : str
                String with pd.Timedelta compatible style. The first training period will begin here.
                If not provided, 3 * horizon is used.
        holidays : pandas.DataFrame
                pd.DataFrame with columns holiday (string) and ds (date type)
                and optionally columns lower_window and upper_window which specify a range of days around the date to be
                included as holidays. lower_window=-2 will include 2 days prior to the date as holidays. Also optionally
                can have a column prior_scale specifying the prior scale for that holiday.
        scoring : str
                Metric used to select the optimal Prophet model.
        return_train_score : bool
                Whereas or not computes training metrics.
        n_jobs : int
                Number of multi processes to distribute the grid search computation.
        verbose : bool
                Whereas or not logs all training information.
        """
        self.prophet_config = param_grid
        self.holidays = holidays
        self.initial = initial
        self.horizon = horizon
        self.period = period
        self.return_train_score = return_train_score
        if self.return_train_score:
            self.train_results = pd.DataFrame()
        else:
            self.train_results = None
        self._grid_df = pd.DataFrame()
        self.scoring = scoring
        if n_jobs is None:
            self.n_jobs = 1
        elif n_jobs == -1:
            self.n_jobs = None
        else:
            self.n_jobs = n_jobs
        self.df = None
        self.fit_kwargs = {}
        if verbose:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)

    @LazyProperty
    def real_n_jobs(self):
        """
        int : Real number of cores used in computation.
        """
        if self.n_jobs is None:
            return cpu_count()
        else:
            return min(cpu_count(), self.n_jobs)

    @LazyProperty
    def _shallow_grid(self):
        """
        list : Shallow parameter grid.
        """
        return [p for p in ParameterGrid(self.prophet_config)]

    @LazyProperty
    def deep_grid(self):
        """
        list : All possible combinations of hyper parameters.
        """
        deep_grid = list()
        for shallow_combination in self._shallow_grid:
            param = dict()
            for key, value in shallow_combination.items():
                if key.startswith('add_') and value is False:
                    continue
                elif isinstance(value, dict):
                    param[key] = [p for p in ParameterGrid(value)]
                elif key == 'holidays' and value is True:
                    param[key] = [self.holidays]
                else:
                    param[key] = [value]
            for p in ParameterGrid(param):
                deep_grid.append(p)
        return deep_grid

    @LazyProperty
    def n_combinations(self):
        """
        int : Number of possible combinations of hyper parameters.
        """
        n = len(self.deep_grid)
        return n

    def _cv_metrics(self, model, parameters):
        """
        Single model cross validation metrics
        Parameters
        ----------
        model : ProphetDecorator
                Trained prophet model.
        parameters : dict
                Prophet hyper parameters, regressors and seasonalities.

        Returns
        -------
                pandas.DataFrame : Cross validation metrics
        """
        df_performance = model.get_cv_metrics(horizon=self.horizon, initial=self.initial, period=self.period)
        df_parameters = pd.DataFrame([pd.Series(parameters)])
        df_performance = pd.concat([df_parameters, df_performance], axis=1)
        df_performance['parameters'] = [parameters]
        df_performance.insert(0, 'model', [model])
        return df_performance

    @staticmethod
    def _train_metrics(model, parameters):
        """
        Single model train metrics
        Parameters
        ----------
        model : ProphetDecorator
                Trained prophet model.
        parameters : dict
                Prophet hyper parameters, regressors and seasonalities.

        Returns
        -------
                pandas.DataFrame : Train metrics
        """
        df_performance = model.get_train_metrics()
        df_parameters = pd.DataFrame([pd.Series(parameters)])
        df_performance = pd.concat([df_parameters, df_performance], axis=1)
        return df_performance

    def pool_func(self, parameters):
        """
        Single Prophet object training and evaluation.
        Parameters
        ----------
        parameters : dict
                Prophet hyper parameters, regressors and seasonalities.

        Returns
        -------
                list : Contains CV performance and train performance.
        """
        start = ttime()
        model = ProphetDecorator(**parameters)
        model.fit(df=self.df, **self.fit_kwargs)
        df_train_performance = None

        if 'holidays' in parameters:
            if isinstance(parameters['holidays'], pd.DataFrame):
                parameters['holidays'] = True

        df_cv_performance = self._cv_metrics(model=model, parameters=parameters)
        if self.return_train_score:
            df_train_performance = self._train_metrics(model=model, parameters=parameters)
        with counter.get_lock():
            counter.value += 1
        logger.info(
            f'Combination {counter.value}/{self.n_combinations} for params {parameters} takes {ttime() - start} seconds')
        return [df_cv_performance, df_train_performance]

    def fit(self, df: pd.DataFrame, **kwargs):
        """
        Training method.
        Trains all hyper parameters combination models and distributes its computing over the specified cores.
        Parameters
        ----------
        df : pandas.DataFrame
                pd.DataFrame containing the history. Must have columns ds (date type) and y, the time series.
                If self.growth is 'logistic', then df must also have a column cap that specifies the capacity
                at each ds.
        kwargs: dict
                Additional arguments passed to the optimizing or sampling functions in Stan.

        Returns
        -------
                GridProphetCV : The fitted GridProphetCV object.

        """
        global counter
        start = ttime()
        self.df = df
        self.fit_kwargs = kwargs
        with Pool(processes=self.n_jobs) as pool:
            results = pool.map(self.pool_func, self.deep_grid)
        counter = Value('i', 0)
        for cv_results, train_results in results:
            self._grid_df = pd.concat([cv_results, self._grid_df])
            if self.return_train_score is not None:
                self.train_results = pd.concat([train_results, self.train_results])
        logger.info(f"Training ends. Taking {ttime() - start} seconds to complete.")
        return self

    @LazyProperty
    def cv_results(self):
        """
        pandas.DataFrame : Cross validation results of all trained models.
        """
        return self._grid_df.drop(columns=['parameters', 'model'])

    def best_model(self, metric=None):
        """
        Returns the best performing model giving an evaluation metric.
        Parameters
        ----------
        metric : str
                Metric used to select the optimal Prophet model.

        Returns
        -------
                ProphetDecorator : The best fitted ProphetDecorator object.
        """
        if metric is None:
            metric = self.scoring
        if metric not in self._grid_df.columns.tolist():
            raise MetricUnavailable
        best_df = self._grid_df[self._grid_df[metric] == self._grid_df[metric].min()]
        return best_df['model'].values[0]

    def best_params(self, metric=None):
        """
        Returns the best performing model hyper parameters giving an evaluation metric.
        Parameters
        ----------
        metric : str
                Metric used to select the optimal Prophet model.

        Returns
        -------
                dict : The best fitted hyper parameters.
        """
        if metric is None:
            metric = self.scoring
        if metric not in self._grid_df.columns.tolist():
            raise MetricUnavailable
        best_df = self._grid_df[self._grid_df[metric] == self._grid_df[metric].min()]
        return best_df['parameters'].values[0]

    def best_score(self, metric=None):
        """
        Returns the best performing model score giving an evaluation metric.
        Parameters
        ----------
        metric : str
                Metric used to select the optimal Prophet model.

        Returns
        -------
                float : The best score giving an evaluation metric.
        """
        if metric is None:
            metric = self.scoring
        if metric not in self._grid_df.columns.tolist():
            raise MetricUnavailable
        best_df = self._grid_df[self._grid_df[metric] == self._grid_df[metric].min()]
        return best_df[metric].values[0]

    def get_model(self, parameters: dict):
        """
        Returns a trained ProphetDecorator model given its hyper parameters.
        Parameters
        ----------
        parameters : dict
                Prophet hyper parameters, regressors and seasonalities.

        Returns
        -------
                ProphetDecorator : The selected ProphetDecorator object.
        """
        model_df = self._grid_df[self._grid_df.parameters == parameters]
        return model_df.model.values[0]
