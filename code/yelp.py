import json
import pandas as pd
import numpy as np
import requests
from pprint import pprint

#https://www.yelp.com/developers/documentation/v3/all_category_list
with open('yelp_categories.json') as f:
    data = json.load(f)

#build a dict with restaurant type:type alias to loop through each type
restaurant_type = {}
for business in data:
    #business['parents'] is a list, check if 'restaurants' exists
    if 'restaurants' in business['parents'] :
        restaurant_type[business['title']] = business['alias']

api_key = ' ------ your api key -------------'
url = 'https://api.yelp.com/v3/businesses/search'
headers = {'Authorization': 'Bearer {}'.format(api_key),}
    

def search_restaurants(offset_num, r_type_alias): 
# This function launches the request for all restaurant in chicago by type.
    #set params for each category to be passed to the API
    url_params = { 
        "term": "restaurants, All",
    "categories": r_type_alias,
    "location":"Chicago",
    "state": "Illinoi",
    'offset': offset_num, # We are going to iterate the offset
     "limit":50 # Maximum return of results per request (ref: API documentation).
     }

    response = requests.get(url, headers=headers, params=url_params)
    return response.json()['businesses'] 


if __name__ == "__main__":
    grand_total = 0
    for r_type, r_type_alias in restaurant_type.items():
        for offset_num in np.arange(50,1000,50) :
            try:
                output_json = search_restaurants(offset_num,r_type_alias)
                data = output_json
                total = len(output_json)
                print("Category : %s -> %s" % (r_type,total))
                if grand_total == 0 :
                    df = pd.DataFrame.from_dict(data)
                    grand_total = total
                else:
                    _df = pd.DataFrame.from_dict(data)
                    df = df.append(_df, sort=True)
                    grand_total = grand_total + total

            except AttributeError:
                print("error at ", offset_num) # Helpful for debugging purposes  

            if total < 50 :
                break
                
                
    print("TOTAL COLLECTED : ", grand_total)
    #remove dups from dataframe
    df.drop_duplicates(['id'], keep=False, inplace=True)
    df.to_csv("yelp_data.csv", index = False)
