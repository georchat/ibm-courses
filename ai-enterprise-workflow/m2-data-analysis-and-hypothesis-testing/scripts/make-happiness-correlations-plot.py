#!/usr/bin/env python

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## plot style, fonts and colors
plt.style.use('seaborn')

SMALL_SIZE = 12
MEDIUM_SIZE = 14
LARGE_SIZE = 16
COLORS = ["darkorange","royalblue","slategrey"]

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=LARGE_SIZE)   # fontsize of the figure title

DATA_DIR = os.path.join(".","data") 
IMAGE_DIR = os.path.join(".","images")

## allow script to be run from parent directory
if not os.path.exists(DATA_DIR):
    DATA_DIR = os.path.join("..","data") 
    IMAGE_DIR = os.path.join("..","images")
    
if not os.path.exists(DATA_DIR):
    raise Exception("cannot find DATA_DIR") 
if not os.path.exists(IMAGE_DIR):
    raise Exception("cannot find IMAGE_DIR")
    
sns.set(style="ticks", color_codes=True)

def ingest_data():
    """
    ready the data for EDA
    """
    print("... data ingestion")
    
    ## load the data and print the shape
    df = pd.read_csv(os.path.join(DATA_DIR, "world-happiness.csv"), index_col=0)

    ## clean up the column names
    df.columns = [re.sub("\s+","_",col) for col in df.columns.tolist()]

    ## drop the rows that have NaNs
    df.dropna(inplace=True)

    ## sort the data for more intuitive visualization
    df.sort_values(['Year', "Happiness_Score"], ascending=[True, False], inplace=True)

    return(df)

def create_correlations_gridplot(df):
    """
    create grid plot of pairwise correlations
    """
    print("... creating plot")
    
    # Compute the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    image_path = os.path.join(IMAGE_DIR, "pairwise-correlations.png")
    plt.savefig(image_path, bbox_inches='tight', pad_inches = 0, dpi=200)
    print("{} created.".format(image_path))
    
if __name__ == "__main__":
    df = ingest_data()
    create_correlations_gridplot(df)
    
