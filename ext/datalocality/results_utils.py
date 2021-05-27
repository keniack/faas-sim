import numpy as np
from scipy.interpolate import make_interp_spline


def smooth_axis(x, y):
    y1 = np.array(y).squeeze()
    x1 = np.array(x).squeeze()
    x_smooth = np.linspace(x1.min(), x1.max(), 1000)
    X_Y_Spline = make_interp_spline(x1, y1)
    y_smooth = X_Y_Spline(x_smooth)
    return x_smooth, y_smooth
