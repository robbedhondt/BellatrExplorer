# Bellatrexplorer: Random forest explainability toolbox based on Bellatrex

<img src="src/assets/screenshot_dashboard.jpeg" alt="Dashboard Screenshot" style="max-width: 800px;"/>

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

## Future work
- General
    - Write a floating point number formatter: floats between 0.01 and 100.0 get rounded to 2 digits, otherwise scientific notation.
    - Add an optional UMAP visualisation that computes in the background. For each feature you can draw a line from the sample to where it would move in the visualisation if you change that feature in that way. You could also display the closest training sample somehow and indicate how far that one is from the current sample (to see how "out of distribution" you are).
    - Better handling of dataset input: autoprocess categorical features (OHE), impute missing values... The datatable at the bottom should contain the "cleaned up" data, so the user can verify and compare to the uploaded data.
    - Potential bug: user changes "learning task" after training; our program relies on value of "learning task" being associated to the currently trained random forest. Solution: stop using "learning task" beyond train_random_forest callback, just use `rf.task` in other places. Caution: any other variables that need to be double-checked for this problem? Like "target" for example...
    - Use the same numpy linspace of quantiles everywhere. Optimize for categorical variables as well (no need to create so many predictions).
- Modeling
    - Show more info: dataset descriptive statistics (n, p, ...), RF train and OOB performance
- Instance selection
    - Add a button to sort the features. Could be based on impurity (but doesn't work for survanal) or on permutation feature importances (based on out of bag error if that's possible?)
    - Sliders could each have a checkmark button to make it "exponentially scaled"? See https://dash.plotly.com/dash-core-components/slider "Non-Linear Slider and Updatemode". Alternatively, autodetect skewness? (cfr Jasper SurvivalLVQ)
    - Slider value with number formatter. Check out [docs](https://dash.plotly.com/dash-core-components/slider), can be achieved through `tooltip.transform` (maybe see if they have also the "g" formatting notation as in Python?)
    - Maybe show the delta on the slider tracks instead of absolute value?
    - Is the slider background even accurate? The color of the current sample doesn't seem to correspond across different features (while that should be the case)
    - Check if there are no problems with setting the sliders to the first sample in the dataset, since the sliders are constrained to the quantiles now
- All rules graph
    - [dcc.Tooltip](https://dash.plotly.com/dash-core-components/slider) is what you need to adapt the px lines tooltip 
    - set max rule depth: through callback with fig.update_yaxes(range=[None, value])
    - Add a legend and selector that would highlight all the rules based on one particular feature (similar to the feature selector in univariate feature effects graph)
    - Add colorbar (as a reference point for instance selection)
    - Change color of the lines to their final prediction
    - Use number formatter in the tooltip
    - Force graph xlim to the range of all the predictions made by the RF
    - Somehow indicate the current prediction (vertical dotted line, as in btrex)
- Univariate feature effects graph
    - Tooltip: show feature value along with quantile
- Bellatrex graph
    - Adapt generated figure size
    - Use older bellatrex graph implementation? That can be packaged along with this repo? (the alignment with the arrow at the bottom is not always perfect...)
- Data table
    - https://dash.plotly.com/datatable table click callback so if you click one of the rows in the datatable it's highlighted and the slider values are changed to it?
- Optimize flask-cache or see the most efficient way to deal with a model hanging around
    ```python
        # TODO: write a demo app with 3 buttons to truly test the difference of these
        #       three methods
        # MODEL HANGING AROUND (with the line `rf = defaults["model"]`)
        # - Total 365 ms (compute 196, network 169)
        # - Data transfer
        #   download: 2271298
        #   upload:   1856500
        # MODEL SERIALIZED
        # - Total 941 ms (compute 533, network 407)
        # - Data transfer
        #   download: 2328515
        #   upload:  29270904
        # MODEL FROM STORAGE CACHE
        # - Total 485 ms (compute 270, network 215)
        # - Data transfer
        #   download: 128057
        #   upload  : 1326
        # rf = defaults["model"]
    ```

References
- [Folder structure](https://community.plotly.com/t/structuring-a-large-dash-application-best-practices-to-follow/62739)
- [Devtools doc](https://dash.plotly.com/devtools)

## Paper guidelines
- Submissions should describe working systems based on state-of-the-art machine learning and data mining technology. Systems should demonstrate that they go beyond basic statistics and leverage machine learning techniques and knowledge discovery processes in a real setting.
- The paper must provide adequate information on the system's components and the way the system is operated, including, e.g., screenshots and a use case.
- Authors should remember that the description of a demo has inherently different content than a research paper submitted to the main conference. A successful demonstration paper should tackle the following questions
    - What are the innovative aspects, and in what way/area does it represent the state of the art?
    - Who are the target users, and why is the system interesting/useful to them?
    - If there are similar/related pieces of software, what are the advantages and disadvantages of the one presented?
- A demonstration submission can be up to four pages long, including references. 
- The paper should contain a URL linking to a demonstration video of at most 5 minutes. This video should show and explain the execution of the system as it will be done at the conference. It may be a mixture or combination of a demo, voice-over (Powerpoint, PDF slides, etc.), and screencast presentations showing and explaining what happens. It should include subtitles in English.

## Related work
- [An Interactive Visual Tool to Enhance Understanding of Random Forest Predictions](https://web.archive.org/web/20210312061825id_/https://publikationen.bibliothek.kit.edu/1000130424/105524939)
    - Manipulate selected features to explore "what-if" scenarios
    - Local surrogate tree as explanation for the prediction
    - Recommendation for reassignment of feature values of the example that leads to change in the prediction to a preferred class
    - This is probably the closest paper to our work
- [RfX: A Design Study for the Interactive Exploration of a Random Forest to Enhance Testing Procedures for Electrical Engines](https://onlinelibrary.wiley.com/doi/pdfdirect/10.1111/cgf.14452?download=true)
    - Global explainability
    - Icicle plots, investigate trees
    - User study
- [iForest: Interpreting Random Forests via Visual Analytics](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8454906)
    - User study
    - Both global and local
    - Local doesn't seem very useful? "decision path view"
- [Interactive Random Forests Plots](https://digitalcommons.usu.edu/cgi/viewcontent.cgi?article=1148&context=gradreports)
    - Master thesis
    - Aimed at larger datasets
    - Parallel coordinates and multidimensional scaling
- [Rfviz: An Interactive Visualization Package for Random Forests in R](https://digitalcommons.usu.edu/cgi/viewcontent.cgi?article=2360&context=gradreports)
    - Same as master thesis above
- [Visualisation of Random Forest classification](https://journals.sagepub.com/doi/full/10.1177/14738716241260745)
    - class distribution per feature and per depth in the tree? not following entirely what is shown in figure 1
    - figure 3 pyramid matrix for feature interaction looks quite cool
