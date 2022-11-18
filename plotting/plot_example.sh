#!/bin/bash

# Plots the sensitvity plots for the analyses done in run_example.sh.
python3 plotting/PlotGlobalSens.py SK SK/SK_T23_DM31_stats.dat
python3 plotting/PlotGlobalSens.py IC IC/IC_T23_DM31_stats.dat
python3 plotting/PlotGlobalSens.py IC+SK IC+SK/IC+SK_T23_DM31_stats.dat



