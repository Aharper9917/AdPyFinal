# run these commands to install
# $ python3.7 -m venv ./py37async
# $ pip install --upgrade pip aiohttp aiofiles

import os
import mmcv
import cv2
import torch
from facenet_pytorch import MTCNN
import numpy as np
from PIL import Image, ImageDraw
from IPython import display

import asyncio
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(keep_all=True, device=device)
mtcnn.to(device)

class webCam:
    def __init__(self):
        self.flag = True
        self.boxes = [[0,0,0,0]]
        self.run = True


async def show_webcam(web, mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if mirror: 
            img = cv2.flip(img, 1)
        if web.flag:
            web.img = img
            web.flag = False
        for box in web.boxes:
            print(box)
            cv2.rectangle(web.img, 
                        (int(box[0]), int(box[1])), 
                        (int(box[2]), int(box[3])), 
                        (255, 0, 0), 2)
        cv2.imshow('Facial Recognition', img) 
        await asyncio.sleep(0.001)
        if cv2.waitKey(1)== 27: 
            break  # esc to quit
    web.run = False
    cv2.destroyAllWindows()

async def box_stuff(web):
    while web.run:
        await asyncio.sleep(.01)
        # put funcion call here, save boxes as web.boxes
        frame = Image.fromarray(cv2.cvtColor(web.img, cv2.COLOR_BGR2RGB))
        web.boxes, _ = mtcnn.detect(frame)

        print(web.boxes)
        print(type(web.boxes))
        web.flag = True
        

async def main():
    web = webCam()

    await asyncio.gather(box_stuff(web), show_webcam(web))


if __name__ == '__main__':
    asyncio.run(main())
