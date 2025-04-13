import os
import numpy as np
PATH_ASSETS = os.path.join(os.path.dirname(__file__), "assets")
PATH_TEMP   = os.path.join(os.path.dirname(__file__), "temp")
TOOLTIP_PREVIOUS_SPLIT = False
QUANTILES = np.linspace(0, 1, 101)
