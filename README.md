![Pynu Logo](/resources/pynu_logo.png)

# Pynu

For the time being the code is contained under the src folder and structured in Analysis and Simulation folders.

## Analysis
Currently performs standard 3-flavor neutrino oscillation analyses for the implemented experiments (IceCube Upgrade and Super-Kamiokande) and their combination. The code also includes some systematic uncertainties associated to the neutrino source and each experiment. 

### Dependencies
- numpy  
- nuSQuIDS  
- nuSQUIDSTools  
- nuflux  
- h5py  
- matplotlib  
- pandas  
- scipy  


### Running
An analysis is specified to the code via a xml file which contains all the information about the experiments, the neutrino sources and the systematic errors. An example of this can be found in src/Analysis/xmlAnalysis/AnalysisTemplate.xml.  

In order to run some examples and get familiar with the code, the src/Analysis/run_example.sh contains a few quick examples. The main program for running the analysis is src/Analysis/runAnalysis.py and is run as follows:  
```
  usage: runAnalysis.py [-h] [-p [POINT]] [-o [OUTFILE]] [--multi] [--cluster] [xml_file]  

  positional arguments:  
    xml_file              Input analysis file in xml format  
  
  optional arguments:  
    -h, --help            show this help message and exit  
    -p [POINT], --point [POINT]  
                          Specify analysis point to run. Only if 'cluster' option is enabled  
    -o [OUTFILE], --outfile [OUTFILE]  
                          Analysis output file  
    --multi               Option for running the analysis with multiprocessing (recommended locally)  
    --cluster             Option for submitting jobs to a cluster
    --mcmc                Option for Markov Chain sampling over paraeter space
```

  
The results of the analysis are saved in <output_file> as a text file of columns (by deafult the output file is out.dat). If no running mode (--multi or --cluster) is specified it will run sequentially the list of points from the xml file.   
