import pandas as pd
import numpy as np
import sklearn.mixture as mix


#######################################################################
# function for gmm
#######################################################################


def calc_quantile_var(data, alpha=0.05):
    """
    compute var by quantile
    """
    return data.quantile(alpha)


def calc_historical_var(data, alpha=0.05):
    """
    compute historical VAR
    """
    if isinstance(data, pd.DataFrame):
        data = data.squeeze()
    return calc_quantile_var(data, alpha=alpha)


def gmm(data, n_components, max_iter=150, random_state=0, **kwds):
    """
    gaussian mixture model by sklearn
    """
    model = mix.GaussianMixture(
        n_components, max_iter=max_iter, random_state=random_state, **kwds
    )
    model.fit(data)
    return model


def gmm_sample(
    data,
    n_components=2,
    max_iter=150,
    random_state=0,
    n_samples=1000,
    risky=True,
    **kwds
):
    """
    sample from the risky component
    """
    model = gmm(
        data,
        n_components=n_components,
        max_iter=max_iter,
        random_state=random_state,
        **kwds
    )
    X_s, y_s = model.sample(n_samples)
    df = pd.DataFrame(X_s, columns=data.columns).assign(component=y_s)
    if not risky:
        ser = pd.Series(X_s.ravel())
        ser.name = "gmm"
        return ser

    risky = df.groupby("component").mean().mean(1).argmin()
    ser = df.query("component==@risky").set_index("component").squeeze()
    ser.name = "gmm_risky"
    return ser


def calc_gmm_var(
    data,
    n_components=2,
    max_iter=150,
    random_state=0,
    n_samples=1000,
    risky=True,
    **kwds
):
    """
    compute quantile var for gmm risky component
    """
    gmm_samples = gmm_sample(data, n_components, risky=risky)
    return calc_quantile_var(gmm_samples)


#######################################################################
# updating historical timeseries dataframes
#######################################################################


def make_update_df(old, new, lookback):
    """combines and cleans numeric timeseries dataframes
       for updates

    # args
        old, new: pandas dataframes
        lookback: numeric

    # returns
        both: combined dataframe
    """
    # combine datasets
    both = pd.concat([old, new])
    # clean it up and keep only lookback period
    return both.drop_duplicates().sort_index().iloc[-lookback:]


#######################################################################
# order execution functions
#######################################################################


def get_open_order_secs(open_orders):
    """func to return list of symbols
        if open order list is populated
    """
    if open_orders:  # if list is populated
        open_order_secs = [order.Symbol for order in open_orders]
    else:
        open_order_secs = []
    return open_order_secs
