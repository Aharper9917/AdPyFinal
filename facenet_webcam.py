import os
import mmcv
import cv2
import torch
from facenet_pytorch import MTCNN
import numpy as np
from PIL import Image, ImageDraw
from IPython import display
# from multiprocessing import Process

FILE_OUT = 'video.mp4'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(keep_all=True, device=device)
mtcnn.to(device)

def find_faces(frame):
    boxes, _ = mtcnn.detect(frame)
    frame_draw = frame.copy()
    draw = ImageDraw.Draw(frame_draw)

    for box in boxes:
        draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)

    open_cv_image = np.array(frame_draw)
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    return open_cv_image

def run():
    cap = cv2.VideoCapture(0)
    # size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    #         int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    while True:
        ret, frame = cap.read()

        if ret == True:
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            frame = find_faces(frame)

            cv2.imshow('Frame', frame)

        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# define MTCNN module

if __name__ == '__main__':
    run()

    
