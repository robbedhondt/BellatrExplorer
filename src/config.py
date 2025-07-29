import os
import numpy as np

# User-changeable parameters
TOOLTIP_PREVIOUS_SPLIT = False
TOOLTIP_PARTIAL_RULE_PATH = True
COLORMAP = "viridis"
QUANTILES = np.linspace(0, 1, 101)
SLIDER_TRIM = 0.05 # 5 percent trim from slider background gradient
# BTREX_SCALE = 0.7 # Downscaler of the generated Bellatrex SVG
YSCALE_NEIGHBORHOOD_GLOBAL = True
# XSCALE_RULES_GLOBAL = True

# Constants
# TODO: random forest parameter ranges?
DEFAULT_N_TREES = 100
DEFAULT_MAX_DEPTH = 10
DEFAULT_MAX_FEATURES = "sqrt"
PATH_ASSETS = os.path.join(os.path.dirname(__file__), "assets")
PATH_TEMP   = os.path.join(os.path.dirname(__file__), "temp")

# Environment variables
IS_DEPLOYED = (os.getenv("DEPLOYED", default="False") == "True")

# Helper variables
last_cleanup_time = 0
