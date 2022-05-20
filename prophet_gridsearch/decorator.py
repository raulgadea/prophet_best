import os
from contextlib import contextmanager

from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics

from prophet_gridsearch.utils import separate_keys


class ProphetDecorator(Prophet):
    """Prophet forecaster Decorator.

    Parameters
    ----------
    growth : str
            'linear' or 'logistic' to specify a linear or logistic trend.
    changepoints : list
            List of dates at which to include potential changepoints.
            If not specified, potential changepoints are selected automatically.
    n_changepoints : int
            Number of potential changepoints to include. Not used if input `changepoints` is supplied.
            If `changepoints` is not supplied, then n_changepoints potential changepoints are selected uniformly from
            the first `changepoint_range` proportion of the history.
    changepoint_range : float
            Proportion of history in which trend changepoints will be estimated.
            Defaults to 0.8 for the first 80%. Not used if `changepoints` is specified.
    yearly_seasonality : bool
            Fit yearly seasonality. Can be 'auto', True, False, or a number of Fourier terms to generate.
    weekly_seasonality : bool
            Fit weekly seasonality. Can be 'auto', True, False, or a number of Fourier terms to generate.
    daily_seasonality : bool
            Fit daily seasonality. Can be 'auto', True, False, or a number of Fourier terms to generate.
    holidays : pandas.DataFrame
            DataFrame with columns holiday (string) and ds (date type) and optionally columns lower_window and
            upper_window which specify a range of days around the date to be included as holidays.
            lower_window=-2 will include 2 days prior to the date as holidays. Also optionally can have a column
            prior_scale specifying the prior scale for that holiday.
    seasonality_mode : str
            'additive' (default) or 'multiplicative'.
    seasonality_prior_scale : float
            Parameter modulating the strength of the seasonality model. Larger values allow the model to fit larger
            seasonal luctuations, smaller values dampen the seasonality.
            Can be specified for individual seasonalities using add_seasonality.
    holidays_prior_scale : float
            Parameter modulating the strength of the holiday components model, unless overridden in the holidays input.
    changepoint_prior_scale : float
            Parameter modulating the flexibility of the automatic changepoint selection. Large values will allow many
            changepoints, small values will allow few changepoints.
    mcmc_samples : int
            if greater than 0, will do full Bayesian inference with the specified number of MCMC samples.
            If 0, will do MAP estimation.
    interval_width : float
            Width of the uncertainty intervals provided for the forecast. If mcmc_samples=0,
            this will be only the uncertainty in the trend using the MAP estimate of the extrapolated generative model.
            If mcmc.samples>0, this will be integrated over all model parameters, which will include uncertainty
            in seasonality.
    uncertainty_samples : int
            Number of simulated draws used to estimate uncertainty intervals. Settings this value to 0 or False
            will disable uncertainty estimation and speed up the calculation. uncertainty intervals.
    stan_backend : str
            string as defined in StanBackendEnum default: None - will try to iterate over all available backends and
            find the working one.
    kwargs: dict
            Contains Verbose mode and all additional regressors and seasonalities.
    """

    def __init__(
            self,
            growth='linear',
            changepoints=None,
            n_changepoints=25,
            changepoint_range=0.8,
            yearly_seasonality='auto',
            weekly_seasonality='auto',
            daily_seasonality='auto',
            holidays=None,
            seasonality_mode='additive',
            seasonality_prior_scale=10.0,
            holidays_prior_scale=10.0,
            changepoint_prior_scale=0.05,
            mcmc_samples=0, interval_width=0.80,
            uncertainty_samples=1000,
            stan_backend=None,
            verbose=False,
            **kwargs
    ):

        super().__init__(
            growth=growth,
            changepoints=changepoints,
            n_changepoints=n_changepoints,
            changepoint_range=changepoint_range,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            holidays=holidays,
            seasonality_mode=seasonality_mode,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
            changepoint_prior_scale=changepoint_prior_scale,
            mcmc_samples=mcmc_samples,
            interval_width=interval_width,
            uncertainty_samples=uncertainty_samples,
            stan_backend=stan_backend
        )
        self.verbose = verbose
        self.hyperparamters = kwargs
        self._initial = None
        self._horizon = None
        self._period = None
        self.cv_metrics = None
        self.train_metrics = None

    def fit(self, df, **kwargs):
        """
        Training method. Adds all aditional regressors and seasonalities and trains.
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
                ProphetDecorator : The fitted ProphetDecorator object.

        """
        season_params, regressor_params = separate_keys(dictionary=self.hyperparamters, key_1='seasonality',
                                                        key_2='regressor')
        for season_key, season_value in season_params.items():
            super().add_seasonality(**season_value)
        for regressor_key, regressor_value in regressor_params.items():
            super().add_regressor(**regressor_value)

        if self.verbose:
            super().fit(df, **kwargs)
        else:
            with self._suppress_stds():
                super().fit(df, **kwargs)
        return self

    def get_cv_metrics(self, horizon, initial=None, period=None, rolling_window=1):
        """
        Cross validation metrics over an expanding window cross validation.
        Further information on fbprophet cross_validation method and performance_metrics
        Parameters
        ----------
        horizon : str
                String with pd.Timedelta compatible style, e.g., '5 days', '3 hours', '10 seconds'.
        initial : str
                String with pd.Timedelta compatible style. Simulated forecast will be done at every this period.
                If not provided, 0.5 * horizon is used.
        period : str
                String with pd.Timedelta compatible style. The first training period will begin here.
                If not provided, 3 * horizon is used.
        rolling_window : float
                Proportion of data to use in each rolling window for computing the metrics.
                Should be in [0, 1] to average

        Returns
        -------
                pandas.DataFrame : Metrics mse, rmse, mae and (if possible) mape, over the cross validation.
        """
        self._initial = initial
        self._horizon = horizon
        self._period = period
        df_cv = cross_validation(self, initial=initial, period=period, horizon=horizon)
        df_performance = performance_metrics(df_cv, rolling_window=rolling_window)
        self.cv_metrics = df_performance
        return df_performance

    def get_train_metrics(self):
        """
        Training metrics
        Returns
        -------
                pandas.DataFrame : Metrics mse, rmse, mae and (if possible) mape, over the training period.
        """
        history_pred = self.predict(self.history)
        history_pred = history_pred.merge(self.history[['ds', 'y']], on='ds')
        history_pred['cutoff'] = history_pred.ds
        df_performance = performance_metrics(history_pred, rolling_window=1)
        self.train_metrics = df_performance
        return df_performance

    def get_prediction_metrics(self, df):
        """
        prediction metrics
        Returns
        -------
                pandas.DataFrame : Metrics mse, rmse, mae and (if possible) mape, over the training period.
        """
        future_pred = self.predict(df)
        future_pred = future_pred.merge(df[['ds', 'y']], on='ds')
        future_pred['cutoff'] = self.history.ds.max()
        df_performance = performance_metrics(future_pred, rolling_window=1)
        return df_performance

    @staticmethod
    @contextmanager
    def _suppress_stds():
        """
        Context manager that suppress Stan logs.
        """
        null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
        save_fds = [os.dup(1), os.dup(2)]

        os.dup2(null_fds[0], 1)
        os.dup2(null_fds[1], 2)
        try:
            yield
        finally:
            os.dup2(save_fds[0], 1)
            os.dup2(save_fds[1], 2)
            for fd in null_fds + save_fds:
                os.close(fd)
