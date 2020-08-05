# Metis Project 3: Stellar Classification
---
### By Edith Johnston

## Table of Contents
---
1. [Project Details](#project-details)
2. [Data Source and Structure](#data-source-and-structure)
3. [Modelling Process](#modelling-process)
4. [Main Tools Used](#main-tools-used)
5. [Deliverables](#deliverables)

## Project Details
---
### Scope
The goal of this project was to use stellar spectroscopic data to determine MK spectral type, in an effort to automate the process of classifying stars.  
[More about MK Spectral Types](https://starparty.com/topics/astronomy/stars/the-morgan-keenan-system/)
[More about stellar spectroscopy](http://spiff.rit.edu/classes/phys440/lectures/intro_spectra/intro_spectra.html)
### Success Metric
I wanted to be as accurate as possible for all classes - especially the minority ones. To that end, for much of the process, I used heatmaps of confusion matrices to estimate model success, and used balanced accuracy score when I need a single score metric.

## Data Source and Structure:
---
### Source
The data I used in this project was from Data Release 16 of the [Sloan Digital Sky Survey](https://www.sdss.org), accessed through the [CasJobs](https://skyserver.sdss.org/casjobs/) SQL interface. Specifically, the data I used was from two tables in DR16: [specObj](http://skyserver.sdss.org/dr16/en/help/browser/browser.aspx#&&history=description+SpecObj+V) and [sppLines](http://skyserver.sdss.org/dr16/en/help/browser/browser.aspx#&&history=description+sppLines+U).  
### Target
The specObj table contains a variety amount of spectroscopic data about stars, amongst which is the subclass column which I used as my target. This column contains both stellar type and subclass, (e.g. A0, G3, M5, etc). In order to reduce complexity of the model, I binned by stellar type.
### Features
The sppLines table contains the spectra line indices that I used as the features of my models.
These line index columns contain the following pieces of information for around 70 different element indices (indices are each a specific wavelength absorbed or emitted by an element):
- side: an approximation of the light intensity at that wavelength using a local fit
- cont: an approximation of the light intensity at that wavelength derived from a global 7th degree polynomial fit.
- err: line index error for that particular line band
- mask: a binary pixel quality indicator (0=good, 1=bad)
For more details about these approximations, see [here](https://www.sdss.org/dr16/spectro/sspp_lineindexmeas/).

## Modelling Process
---
## Sampling Data
The data I was working with had 370 features and over 500,000 samples, so to begin with, I took a random subset of the data containing 5% of all samples.  
## Model Selection
I was working on a cloud computing platform, so I tested a lot of initial models. After comparison, the best performing ones were XGBoost and HistGradientBoosting.
## Balancing Classes
My classes were very imbalanced (just 5 of the 9 classes contain over 96% of samples), so I resampled the data in order to better train the models. I randomly undersampled the 5 largest classes, then randomly oversampled the 4 smallest samples.
## Parameter Tuning
I used GridCV to tunes model parameters. After tuning, the best performing model was HistGradientBoosting.

## Main Tools Used
---
### Data Acquisition
- CasJobs
- SQL
- SciDrive
### Modelling
- SciServer
- Jupyter Notebooks
- scikit learn, imblearn, xgboost
- matplotlib, seaborn
- pandas, numpy

## Deliverables:
---
### Notebooks
0. [Acquiring Data](https://github.com/edithalice/stellar_classification/blob/master/0_Data_Acquisition.ipynb) (Notebook was not used to acquire data, but contains walkthrough)
1. [Sampling Data](https://github.com/edithalice/stellar_classification/blob/master/1_Sampling_Data.ipynb)
2. [Model Selection](https://github.com/edithalice/stellar_classification/blob/master/2_Model_Selection.ipynb)
3. [Balancing Classes](https://github.com/edithalice/stellar_classification/blob/master/3_Balancing_Classes.ipynb)
4. [Parameter Tuning](https://github.com/edithalice/stellar_classification/blob/master/4_Parameter_Tuning.ipynb)
### Python Modules
- [Sampling Processes](https://github.com/edithalice/stellar_classification/blob/master/sampling_process.py)
- [Modelling Processes](https://github.com/edithalice/stellar_classification/blob/master/model_processes.py)
### Presentation
- [Google Slides](https://docs.google.com/presentation/d/18RMsiepjmJ7rpHx3ZY8O1-9iLH5Xa2W5g2Orai8yh6s/edit?usp=sharing)
- [PDF]()
- [Powerpoint]()
 
