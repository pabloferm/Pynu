<img src="/resources/pynu_logo.png" width="150" />

# PyNuFit Framework

The aim of this software is to perform neutrino analysis in the most general and flexible way. There are three modules:
- PyNu: It is the core of the package handling simulations, data and fitting.
- Plot: A plotting toolkit to extract all the information from the analysis (UNDER CONSTRUCTION).
- Report: Automated module for preparing a report containing the detailed information of the analysis and the results (UNDER CONSTRUCTION).

For a more complete documentation, please open the ```docs/pynu.html``` folder with your browser.


# Dependencies
This code uses python3.10.
First, set the ```PYNU``` environment variable and add it to your ```PYTHONPATH```. On the top folder called ```Pynu``` do:
```
git clone https://github.com/pabloferm/Pynu.git
cd Pynu
export PYNU=$PWD/pynu
export PYTHONPATH=$PYTHONPATH:$PYNU
```

Some of the requirements can be installed directly with the following command:
```
pip install -r requirements.txt
```

Others, need to be installed by hand and can be a bit tricky. Please, let me know if you find any issues:
- nuSQuIDS (https://github.com/arguelles/nuSQuIDS): This package handles neutrino oscillations (https://arxiv.org/abs/2112.13804).
- nuflux (https://github.com/icecube/nuflux): This package is used for computing atmospheric neutrino fluxes, so may not be necessary.


# Usage

In the top directory there exist one simple python program for each of the modules. It is highly recommended that all input and output files of a given analysis is contained in a dedicated folder.
- ```analysis_main.py``` for ```PyNuFit```.
- ```plot_main.py``` for ```Plot```.

## PyNu

All the analysis information is passed to ```PyNuFit``` through an xml file. You can find examples for them at ```examples/AnalysisFiles/```.
For running the example ```analysis_main.py```:
```
usage: analysis_main.py [-h] [-p POINT [POINT ...]] [-rp RANGE_OF_POINTS [RANGE_OF_POINTS ...]] [-o [OUTFILE]] [--multi] [--cluster] [--mcmc] [xml_file]

positional arguments:
  xml_file              Input analysis file in xml format.

options:
  -h, --help            show this help message and exit
  -p POINT [POINT ...], --point POINT [POINT ...]
                        Specify analysis point or points (p0 p1 p2 p3) to run.
  -rp RANGE_OF_POINTS [RANGE_OF_POINTS ...], --range_of_points RANGE_OF_POINTS [RANGE_OF_POINTS ...]
                        Specify range (start and end) of analysis points to run, p0 p3 = p0 p0+1 p0+2 ... p3-1 p3. Edges are included.
  -o [OUTFILE], --outfile [OUTFILE]
                        Analysis output file.
  --multi               Option for running the analysis with multiprocessing (recommended locally).
  --cluster             Option for submitting jobs to a cluster.
  --mcmc                Option for sampling parameter space using Markov Chain Monte Carlo.
```

## Plot

You can produce the plots for the analysis using ```plot_main.py```:
```
usage: plot_main.py [-h] [-xml [XML_FILE]] [-dir [DIRECTORY]] [hdf5_file]

positional arguments:
  hdf5_file             Output analysis file in hdf5 format.

options:
  -h, --help            show this help message and exit
  -xml [XML_FILE], --xml_file [XML_FILE]
                        Input analysis file in xml format.
  -dir [DIRECTORY], --directory [DIRECTORY]
                        Path to folder to store the analysis.
```

## Report

Not yet...
