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

def save_fig(fig_id, tight_layout=True, image_path=IMAGE_DIR):
    path = os.path.join(image_path, fig_id + ".png")
    print("... saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

def load_clean_data():
    """
    load the clean dataset after data ingestion
    """
    print("... data loading")
    return pd.read_csv(os.path.join(DATA_DIR, "aavail-target.csv"))

def create_plots(df):
    """
    create plots for data visualization
    """
    
    print("... creating plots")
    
    # Analyze customer churn per market
    fig = plt.figure(figsize=(14,6))
    ax1 = fig.add_subplot(121)
    table_c = pd.pivot_table(df, index = ['country_name'], columns=["is_subscriber"], values = 'customer_id', aggfunc="count")
    table_c.plot(kind='bar', stacked=True, ax=ax1)
    ax1.set_xlabel("country")
    ax1.set_ylabel("num_of_customers")
    save_fig("market_size")
    
    # Analyze customer churn per market and subscription type
    fig = plt.figure(figsize=(12,5))

    ax1 = plt.subplot(121)
    table_1 = pd.pivot_table(df[df.country_name=="united_states"], index = ['subscriber_type'], columns=["is_subscriber"], values = 'customer_id', aggfunc="count")
    table_1.plot(kind='bar', stacked=True, ax=ax1, legend=False)
    ax1.set_xlabel("subscriber type")
    ax1.set_ylabel("num_of_customers")
    ax1.title.set_text("United States")

    ax2=plt.subplot(122)
    table_2 = pd.pivot_table(df[df.country_name=="singapore"], index = ['subscriber_type'], columns=["is_subscriber"], values = 'customer_id', aggfunc="count")
    table_2.plot(kind='bar', stacked=True, ax=ax2)
    ax2.set_xlabel("subscriber type")
    ax2.title.set_text("Singapore")
    
    save_fig("market_size_per_subscriber_type")
    
    # Focus only on the singapore customers
    df_singapore = df[df.country_name=="singapore"].copy()
    num_features = ["age", "num_streams"]
    
    ax1 = sns.pairplot(df_singapore, vars=num_features, hue="is_subscriber", palette="husl")
    save_fig("pairplot_churn")
    
    ax2 = sns.pairplot(df_singapore, vars=num_features, hue="subscriber_type", palette="husl")
    save_fig("pairplot_subscriber_type")
    
    # Analyze customer churn per age group in Singapore
    fig = plt.figure(figsize=(14,6))
    ax1 = fig.add_subplot(121)
    table = df_singapore.groupby(["is_subscriber", "subscriber_type"])["age"].describe()["mean"].unstack(0)
    table.plot(kind='barh', ax=ax1)
    ax1.set_xlabel("age")
    ax1.title.set_text("Average customer age per subscriber type")
    save_fig("average_customer_age_per_subscriber_type")
    
    
    df_singapore["age_rank_bins"] = pd.cut(df_singapore.age, 3, precision=0)
    fig = plt.figure(figsize=(15,5))
    ax1 = plt.subplot(131)

    sns.countplot(x="age_rank_bins", data=df_singapore[df_singapore.subscriber_type=="aavail_basic"], hue="is_subscriber", ax=ax1)
    ax1.set_xlabel("age groups")
    ax1.set_ylabel("num_of_customers")
    ax1.title.set_text("aavail_basic")

    ax2=plt.subplot(132)
    sns.countplot(x="age_rank_bins", data=df_singapore[df_singapore.subscriber_type=="aavail_premium"], hue="is_subscriber", ax=ax2)
    ax2.set_xlabel("age groups")
    ax2.set_ylabel("num_of_customers")
    ax2.title.set_text("aavail_premium")

    ax3=plt.subplot(133)
    sns.countplot(x="age_rank_bins", data=df_singapore[df_singapore.subscriber_type=="aavail_unlimited"], hue="is_subscriber", ax=ax3)
    ax3.set_xlabel("age groups")
    ax3.set_ylabel("num_of_customers")
    ax3.title.set_text("aavail_unlimited")
    
    save_fig("num_of_customers_per_age_group")
    
    # Analyze customer churn per streams number in Singapore
    df_singapore["streams_rank_bins"] = pd.cut(df_singapore.num_streams, 3, precision=0)
    fig = plt.figure(figsize=(15,5))
    ax1 = plt.subplot(131)

    sns.countplot(x="streams_rank_bins", data=df_singapore[df_singapore.subscriber_type=="aavail_basic"], hue="is_subscriber", ax=ax1)
    ax1.set_xlabel("streams group")
    ax1.set_ylabel("num_of_customers")
    ax1.title.set_text("aavail_basic")

    ax2=plt.subplot(132)
    sns.countplot(x="streams_rank_bins", data=df_singapore[df_singapore.subscriber_type=="aavail_premium"], hue="is_subscriber", ax=ax2)
    ax2.set_xlabel("streams groups")
    ax2.set_ylabel("num_of_customers")
    ax2.title.set_text("aavail_premium")

    ax3=plt.subplot(133)
    sns.countplot(x="streams_rank_bins", data=df_singapore[df_singapore.subscriber_type=="aavail_unlimited"], hue="is_subscriber", ax=ax3)
    ax3.set_xlabel("streams groups")
    ax3.set_ylabel("num_of_customers")
    ax3.title.set_text("aavail_unlimited")
    
    save_fig("num_of_customers_per_streams_group")
    
    
if __name__ == "__main__":
    df = load_clean_data()
    create_plots(df)
    
