from pathlib import Path
import pandas as pd
import numpy as np
import sklearn.mixture as mix
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image, ImageOps

REPO_NAME = 'mixture_model_trading_public'
LOGO_LOC = "data/watermarks/Blackarbs LLC Watermark_rotate30.png"

# --------------------------------------------------------------------------------------
# general utils


def get_relative_project_dir(project_repo_name=None, partial=True):
    """helper fn to get local project directory"""
    current_working_directory = Path.cwd()
    cwd_parts = current_working_directory.parts
    if partial:
        while project_repo_name not in cwd_parts[-1]:
            current_working_directory = current_working_directory.parent
            cwd_parts = current_working_directory.parts
    else:
        while cwd_parts[-1] != project_repo_name:
            current_working_directory = current_working_directory.parent
            cwd_parts = current_working_directory.parts
    return current_working_directory


def check_path(fp):
    """fn: to create file directory if it doesn't exist"""
    if not Path(fp).exists():

        if len(Path(fp).suffix) > 0:  # check if file
            Path(fp).parent.mkdir(exist_ok=True, parents=True)

        else:  # or directory
            Path(fp).mkdir(exist_ok=True, parents=True)


def cprint(df, nrows=None):
    if not isinstance(df, (pd.DataFrame,)):
        try:
            df = df.to_frame()
        except:
            raise ValueError("object cannot be coerced to df")

    if not nrows:
        nrows = 5
    print("-" * 79)
    print("dataframe information")
    print("-" * 79)
    print(df.tail(nrows))
    print("-" * 50)
    print(df.info())
    print("-" * 79)
    print()


def get_column_range(series):
    """
    get min and max values

    :param series: array-like
    :return: float, float
    """
    return min(series), max(series)


def make_gmm(n_components, max_iter=150, random_state=None):
    """fn: wrapper for sklearn's Gaussian Mixture to create gmm object"""
    if not random_state:
        random_state = 7
    model_kwds = dict(
        n_components=n_components,
        max_iter=max_iter,
        n_init=100,
        init_params="random",
        random_state=random_state,
    )

    gmm = mix.GaussianMixture(**model_kwds)
    return gmm


def make_ic_series(list_of_tups, name=None):
    """fn: convert list of tuples for
            information criterion (aic, bic) into series
    # args
        list_of_tups : list() of tuples()
            tuple[0] is n_component, tuple[1] is IC
        name : str(), name of IC

    # returns
        s : pd.Series()
            index is n_components, values are IC's
    """
    s = (
        pd.DataFrame(list_of_tups)
        .rename(columns={0: "n_components", 1: name})
        .set_index("n_components")
        .squeeze()
    )
    return s

# --------------------------------------------------------------------------------------
def add_watermark(ax, scale=1.5, xo=0, yo=0):
    """
    add watermark to image

    See:
        https://stackoverflow.com/questions/43842567/matplotlib-automate-placement-of-watermark
        https://www.blog.pythonlibrary.org/2017/10/12/how-to-resize-a-photo-with-python/

    :param ax: axis object
    :param numeric scale: scale size up or down
    :param numeric xo: x-offset
    :param numeric yo: y-offset
    :return:
    """

    project_dir = get_relative_project_dir(REPO_NAME)
    logo_loc = Path(project_dir / LOGO_LOC).as_posix()
    img = Image.open(logo_loc)
    try:
        width, height = ax.figure.get_size_inches() * ax.figure.get_dpi()
    except:
        width, height = plt.gcf().get_size_inches() * plt.gcf().get_dpi()
    wm_width = int(width / scale)  # scale watermark
    scaling = wm_width / float(img.size[0])
    wm_height = int(float(img.size[1]) * float(scaling))
    img = ImageOps.fit(img, (wm_width, wm_height), Image.ANTIALIAS)
    plt.figimage(img, xo=xo, yo=yo, alpha=0.25, zorder=1)
    return
