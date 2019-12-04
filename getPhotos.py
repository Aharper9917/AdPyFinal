'''
This simple script is used to download the images from the web 
using the links provided in the DataturksFaceDataset's .jason file

Run this to down load the images
'''
import requests
import shutil
import json
import re
import os
import time

IMG_FOLDER = "data/images"
DATA_PATH = "data/face_detection.json"
OUT_DATA = "data/face_detection_formatted.json"

def getData(filePath):
    with open(filePath) as json_file:
        data = json.load(json_file)
    json_file.close()
    return data

def saveData(data, filePath):
    with open(filePath, 'w') as outfile:
        json.dump(data, outfile, indent=2)
    outfile.close()

def downloadImages(data):
    img_num = 1

    for img_Data in data:
        url = img_Data['content']
        
        # regex for file extention
        regex = re.compile(r"\.[^.\\/:*?<>|\r\n]+$")
        extensions = regex.finditer(url)
        extension = [x.group(0) for x in extensions]

        img_name = "image_" + str(img_num).zfill(6) + extension[0]
        file_path = os.path.join(IMG_FOLDER, img_name)

        # get response form server
        response = requests.get(url, stream=True)
        if(response.status_code == 200): # Success!
            with open(file_path, 'wb') as f:
                response.raw.decode_content = True
                shutil.copyfileobj(response.raw, f)
            f.close()
        else:
            print("An error has occured")
        
        img_num += 1


if __name__ == '__main__':
    data = getData(DATA_PATH)

    since = time.time()
    downloadImages(data)
    print(time.time() - since)
