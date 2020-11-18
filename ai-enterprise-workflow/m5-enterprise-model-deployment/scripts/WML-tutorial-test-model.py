#!/usr/bin/env python

"""
Watson machine learning tutorial

NOTE: this script requires scikit-learn==0.20.3

"""


import os
import sys
import json
import pickle
import pandas as pd
import numpy as np

from watson_machine_learning_client import WatsonMachineLearningAPIClient
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


## check for correct version of sklearn
SKLEARN_VERSION = "0.20.3"

from sklearn import __version__
if __version__ != SKLEARN_VERSION:
    raise Exception("sklearn version must be {}".format(SKLEARN_VERSION))

def load_wml_credentials():
    """
    fetch the saved watson machine learning credentials
    """

    wmlcreds_dir = os.path.join(".","ibm")
    wmlcreds_file = os.path.join(wmlcreds_dir,'wml.json')

    if not os.path.exists(wmlcreds_file):
        raise Exception("cannot find {}".format(wmlcreds_file))
    
    with open(wmlcreds_file, "r") as read_file:
        wmlcreds = json.load(read_file)

    return(wmlcreds)
        
def connect_wml_service():
    """
    Instantiate a client using credentials
    """

    wmlcreds = load_wml_credentials()
    wml_credentials = {
        "apikey": wmlcreds['apikey'],
        "instance_id": wmlcreds['instance_id'],
        "url": wmlcreds['url'],
    }

    client = WatsonMachineLearningAPIClient(wml_credentials)
    return(client)

def load_aavail_data():
    data_dir = os.path.join(".","data")
    df = pd.read_csv(os.path.join(data_dir,r"aavail-target.csv"))
       
    ## pull out the target and remove uneeded columns
    _y = df.pop('is_subscriber')
    y = np.zeros(_y.size)
    y[_y==0] = 1 
    df.drop(columns=['customer_id','customer_name'],inplace=True)
    df.head()
    X = df

    return(X,y)

if __name__ == "__main__":


    ## connect to the wml service with saved credentials
    client = connect_wml_service()
    instance_details = client.service_instance.get_details()
    print("connected")

    
    ## prepare the data
    X,y = load_aavail_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    print("list models")
    print(client.repository.list_models())
    print(client.deployments.get_uids())

    #sys.exit()
    
    ## use your model uid to get the model details
    model_uid = "c3c7f98c-32bd-4053-ac3d-904730b09d76"
    model_details = client.repository.get_model_details(model_uid)

    deployment_uid = "2f4934ec-4ba5-4a16-a22c-73b6e5c81c2d"
    deployment_details = client.deployments.get_details(deployment_uid)
    
    ## test model
    print("testing")
    scoring_endpoint = client.deployments.get_scoring_url(deployment_details)
    fields = X_test.columns.values.tolist()
    values = X_test.values[:3].tolist()
    playload_scoring = {'fields': fields, 'values': values}
    print(playload_scoring)
    predictions = client.deployments.score(scoring_endpoint,playload_scoring)
    predictions = pd.DataFrame(predictions["values"], columns=predictions["fields"])
    print(predictions)
    
 

