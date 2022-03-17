#!/bin/bash

# Computes the standard 3-flavor oscillation analysis for SK and IC atm. neutrinos and their combination assuming no systematics.
python3 runAnalysis.py xmlAnalysis/SK_T23_DM31_stats.xml -o SK/SK_T23_DM31_stats.dat --multi
python3 runAnalysis.py xmlAnalysis/IC_T23_DM31_stats.xml -o IC/IC_T23_DM31_stats.dat --multi
python3 runAnalysis.py xmlAnalysis/IC+SK_T23_DM31_stats.xml -o IC+SK/IC+SK_T23_DM31_stats.dat --multi
