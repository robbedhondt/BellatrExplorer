import os
import numpy as np

# User-changeable parameters
TOOLTIP_PREVIOUS_SPLIT = False
COLORMAP = "viridis"
QUANTILES = np.linspace(0, 1, 101)

# Constants
PATH_ASSETS = os.path.join(os.path.dirname(__file__), "assets")
PATH_TEMP   = os.path.join(os.path.dirname(__file__), "temp")
