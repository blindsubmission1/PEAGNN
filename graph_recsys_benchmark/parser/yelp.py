import pandas as pd
import os
import json


def parse_yelp(dir):
    """
    Read the yelp dataset from .tar file
    :param dir: the path to raw tar file (yelp_dataset.tar)
    :return: yelp_business, yelp_review, yelp_user, yelp_checkin, yelp_tip, pandas.DataFrame
    """

    #Importing Yelp business data
    with open(os.path.join(dir, 'yelp_academic_dataset_business.json'), encoding='utf-8') as json_file:
      data = [json.loads(line) for line in json_file]    
      df_yelp_business = pd.DataFrame(data)
    
    #Importing Yelp review data
    with open(os.path.join(dir, 'yelp_academic_dataset_review.json'), encoding='utf-8') as json_file:
      data = [json.loads(line) for line in json_file]    
      df_yelp_review = pd.DataFrame(data)

 
    #Importing Yelp user data
    with open(os.path.join(dir, 'yelp_academic_dataset_user.json'), encoding='utf-8') as json_file:
      data = [json.loads(line) for line in json_file]    
      df_yelp_user = pd.DataFrame(data)
    
    #Importing Yelp checkin data
    with open(os.path.join(dir, 'yelp_academic_dataset_checkin.json'), encoding='utf-8') as json_file:
      data = [json.loads(line) for line in json_file]    
      df_yelp_checkin = pd.DataFrame(data)
        
    #Importing Yelp tip data
    with open(os.path.join(dir, 'yelp_academic_dataset_tip.json'), encoding='utf-8') as json_file:
      data = [json.loads(line) for line in json_file]    
      df_yelp_tip = pd.DataFrame(data) 
        
    return df_yelp_business, df_yelp_user, df_yelp_review, df_yelp_tip, df_yelp_checkin
