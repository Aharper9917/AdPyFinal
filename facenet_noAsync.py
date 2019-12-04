import os, mmcv, cv2, torch
from facenet_pytorch import MTCNN
import numpy as np
from PIL import Image, ImageDraw
from IPython import display
# from multiprocessing import Process

FILE_OUT = 'video.mp4'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))


def saveVideo():
    if os.path.isfile(FILE_OUT):
        os.remove(FILE_OUT)

    cap = cv2.VideoCapture(0)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    out = cv2.VideoWriter(FILE_OUT, fourcc, 20.0, size)

    while True:
        ret, frame = cap.read()

        if ret == True:
            frame = cv2.flip(frame, 1)  # Mirror img

            out.write(frame)  # Save img

            cv2.imshow('Camera', frame)

        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


# define MTCNN module
mtcnn = MTCNN(keep_all=True, device=device)
mtcnn.to(device)

if __name__ == '__main__':
    saveVideo()

    video = mmcv.VideoReader(FILE_OUT)
    frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]
    
    display.Video(FILE_OUT, width=640)
    frames_tracked = []
    for i, frame in enumerate(frames):
        print('\rTracking frame: {}'.format(i + 1), end='')

        # Detect faces
        boxes, _ = mtcnn.detect(frame)

        # Draw faces
        frame_draw = frame.copy()
        draw = ImageDraw.Draw(frame_draw)
        for box in boxes:
            draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)

        frames_tracked.append(frame_draw.resize((640, 360), Image.BILINEAR))
    print('\nDone')

    dim = frames_tracked[0].size
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')    
    video_tracked = cv2.VideoWriter('video_tracked.mp4', fourcc, 25.0, dim)
    for frame in frames_tracked:
        video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    video_tracked.release()

    cap = cv2.VideoCapture('video_tracked.mp4')
    while True:
        ret, frame = cap.read()

        if ret == True:
            cv2.imshow('Video Tracked', frame)
        else:
            break
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()