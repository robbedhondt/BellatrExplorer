# Bellatrexplorer: Random forest explainability toolbox based on Bellatrex

<img src="fig/screenshot_dashboard.jpeg" alt="Dashboard Screenshot" style="max-width: 800px;"/>

## Installation
First of all, clone this repository in your place of preference:
```bash
git clone https://github.com/robbedhondt/BellatrExplorer
cd BellatrExplorer
```
To run Bellatrex, we require Python 3.9. For now, it is recommended to clone Bellatrex from git instead of installing it with pip, as we need the latest development version for this explorer tool to work. The easiest way to install and run the visualisation tool is to configure a python virtual environment with [venv](https://docs.python.org/3/library/venv.html).
```bash
python3.9 -m venv venv_btrexplorer
source venv_btrexplorer/bin/activate
```

Clone and install the Bellatrex repository.
```bash
git clone https://github.com/Klest94/Bellatrex.git
cd Bellatrex
pip install -e .
```

Now go back to the root directory and install the remaining required packages.
```bash
cd ..
pip install -r requirements.txt
```

## Run the app
From the root directory of this project, run the following:
```bash
cd src
python web_visualisation.py
```

The app will now be available on http://127.0.0.1:8050/.
