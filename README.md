# Adaptive Game Design Study
This repository contains the data analysis and visualization code used in the master's thesis:

“An Adaptive Game Design Framework for Dynamically Interpolating Gameplay Experiences Based on Player Behavior.”

The scripts process experimental data collected from 21 participants, perform statistical analyses, and generate all figures used in the thesis.

Overview

This project analyzes player experience across three gameplay modes:

- Action
- Stealth
- Adaptive (dynamically interpolated between Action and Stealth)

The goal is to evaluate whether the adaptive system:

- Matches or improves player experience relative to preferred modes
- Produces more consistent experiences across participants
- Aligns with player behavioral tendencies

Repository Structure
.
├── Run_Analysis.py                       # Main script for statistical analysis
├── Analysis_Functions.py                 # Statistical tests and helper functions
├── Generate_Visualizations.py            # Script to generate figures
├── Plot_Functions.py                     # Plotting utilities
├── Adaptive_Game_Design_Study_Data.csv   # Collected Data after initial cleanup
└── graphs/                               # Output folder for generated figures


Requirements

Python 3.10+

Install dependencies:
</> pip install pandas numpy matplotlib scipy statsmodels pingouin </>


