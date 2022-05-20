def separate_keys(dictionary, key_1, key_2):
    """
    Extracts keys ending with the given strings.
    Parameters
    ----------
    dictionary : dict
            Dictionary to evaluate.
    key_1 : str
            Ends with key one.
    key_2 : str
            Ends with key two.

    Returns
    -------
    d1 : dict
            Dictionary with keys endig with key_1
    d2 : dict
            Dictionary with keys endig with key_2
    """
    d1 = {key: value for i, (key, value) in enumerate(dictionary.items()) if key.endswith(key_1)}
    d2 = {key: value for i, (key, value) in enumerate(dictionary.items()) if key.endswith(key_2)}
    return d1, d2


class MetricUnavailable(Exception):
    def __init__(self,
                 message='Metric not available. Only mse, rmse, mae always available and mape if prophet does not skip it.'):
        super().__init__(message)
