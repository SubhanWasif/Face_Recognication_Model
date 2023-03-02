
import io,os,csv
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/schneider/Desktop/MLProject/Assignment/mystical-melody-376814-b2a7836d24be.json"
from google.cloud import vision
from google.cloud.vision_v1 import types
client = vision.ImageAnnotatorClient()



directory1 = '/home/schneider/Desktop/MLProject/dataset/'
directory2 = '/home/schneider/Desktop/MLProject/mydatset/'
csv_files = [directory1,directory2]

def SmileApi(picture):
    expression = 'Not Smiling'
    image_url=(os.path.abspath(picture))
    with io.open(image_url, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.label_detection(image=image)
    labels = response.label_annotations
    for label in labels:
        if label.description == 'Smile':
            expression = 'Smiling'

    return expression

def FileRead():
    fields = ["Path","Name", "Expression", "Age"]
    with open("FinalDataSet.csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for i in csv_files:
            if i == os.path.basename(directory2):
                for picture in os.scandir(i):
                    expression = SmileApi(picture)
                    writer.writerow({"Path": os.path.abspath(picture), "Name": 'Me', "Expression": expression,'Age':20})
            else:
                for picture in os.scandir(i):
                    expression = SmileApi(picture)
                    writer.writerow({"Path": os.path.abspath(picture), "Name":'Unknown' , "Expression": expression,'Age':""})


def shuffle(final_file):
    
    df = pd.read_csv(final_file) # avoid header=None. 
    final = df.sample(frac=1)
    final.to_csv(final_file, index=False)


def SplitData():
    data = pd.read_csv('FinalDataSet.csv')
    X_train, X_test = train_test_split(data,test_size=0.2)
    print(X_train)






FileRead()
shuffle("FinalDataSet.csv")
SplitData()

