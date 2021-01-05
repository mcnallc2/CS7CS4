import json_lines
import csv
import ftfy
from google_trans_new import google_translator

translator = google_translator()    # create translator

with open('reviews.csv', 'w', newline='\n') as csvfile: # open new csv file
    writer = csv.writer(csvfile)    # create csv writer
    writer.writerow(["text", "voted_up", "early_access"])   # add header
    
    with open('reviews_178.jl', 'rb') as f: # open json file
        for item in json_lines.reader(f):   # for each line in json file
            
            text = ftfy.fix_text(item['text'])  # fix to standard unicode text
            X = translator.translate(text)      # translate text to english
            
            if item['voted_up']:     # if voted up set value to 1 otherwise 0
                y=1
            else:
                y=0
            
            if item['early_access']: # if early access set value to 1 otherwise 0
                z=1
            else:
                z=0
            
            writer.writerow([X, y, z])  # write processed line of json to csv