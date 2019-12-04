'''

'''
import random
import asyncio
import mmcv
from engine import train_one_epoch, evaluate
import transforms as T
import torchvision.transforms as transforms
import utils
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import json
import cv2
import time

IMAGE_SIZE = 500
ROOT_DATA = "data/"
ROOT_DATA2 = "TransLearning_self/data/"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class webCam:
    def __init__(self):
        self.flag = True
        self.boxes = [[0, 0, 0, 0]]
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
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    web.run = False
    cv2.destroyAllWindows()


async def box_stuff(web):
    while web.run:
        await asyncio.sleep(.01)
        # put funcion call here, save boxes as web.boxes
        frame = Image.fromarray(cv2.cvtColor(web.img, cv2.COLOR_BGR2RGB))

        target = {}
        target["boxes"] = [[0,0,0,0]]

        transforms = get_transform(train=False)
        img, target = transforms(frame, target)

        model.eval()
        with torch.no_grad():
            prediction = model([img.to(device)])
            img = img.mul(255).permute(1, 2, 0).byte().numpy()

        web.boxes = prediction[0]['boxes'].cpu().numpy()
        web.flag = True


async def main():
    web = webCam()

    await asyncio.gather(box_stuff(web), show_webcam(web))


class DataturksFaceDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):  
        # root_of_data = TransLearning_self/data/
        self.root = root
        self.transforms = transforms
        self.data = getData(os.path.join(root, "face_detection.json"))
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))

    def __getitem__(self, idx):
        # load images
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        # TODO: normalize img size to fit NN

        # get bounding boxes for each face in img
        num_faces = 0
        boxes = []
        for label_Data in self.data[idx]['annotation']:
            num_faces += 1 # keeps track of number of labeled faces in img
            # the dataset gives us the bouding box coords vs decimal ration
            # box = [xmin*imgWidth, ymin*imgHeight,
            #        xmax*imgWidth, ymax*imgHeight]
            imgWidth, imgHeight = img.size 

            # Get the proportions
            xmin = label_Data['points'][0]['x']
            ymin = label_Data['points'][0]['y']
            xmax = label_Data['points'][1]['x']
            ymax = label_Data['points'][1]['y']
            # print([xmin, ymin, xmax, ymax])

            # Get the coords 
            # xmin = int(label_Data['points'][0]['x'] * imgWidth)
            # ymin = int(label_Data['points'][0]['y'] * imgHeight)
            # xmax = int(label_Data['points'][1]['x'] * imgWidth)
            # ymax = int(label_Data['points'][1]['y'] * imgHeight)

            boxes.append([xmin, ymin, xmax, ymax])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32) # turn boxes into tensor
        labels = torch.ones((num_faces,), dtype=torch.int64) # there is only one class, faces
        image_id = torch.tensor([idx])  # create tensor with idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) # tensor with area of each bounding box
        iscrowd = torch.zeros((num_faces,), dtype=torch.int64) # suppose all instances are not crowded

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        height, width = img.shape[-2:]
        # print(img.shape[-2:])
        boxProportions = target['boxes']

        for box in target['boxes']:
            box[0] = box[0] * width
            box[1] = box[1] * height
            box[2] = box[2] * width
            box[3] = box[3] * height
            
        # for area in target['area']:
        #     print(area)
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    
        return img, target
    
    def __len__(self):
        return len(self.imgs)
        
def getData(datafile):
    with open(datafile) as json_file:
        data = json.load(json_file)
    json_file.close()
    return data


def get_transform(train):
    mean = [0.3297]
    std = [0.2566]
    transform = []

    transform.append(T.Resize(IMAGE_SIZE))
    if train:
        transform.append(T.RandomHorizontalFlip(0.5))
    transform.append(T.ToTensor())
    return T.Compose(transform)


def test_dataset():
    # Used to test DataturksFaceDataset class
    dataset = DataturksFaceDataset(ROOT_DATA)

    for i in range(408):
        img_data = dataset[i]
        img = img_data[0]
        opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        boxes = img_data[1]['boxes'].numpy()
        print(type(1))
        for j in range(len(boxes)):
            img = cv2.rectangle(img, 
                                (boxes[j][0], boxes[j][1]),
                                (boxes[j][2], boxes[j][3]),
                                (255, 23, 122), 1) # cv2 HAS STUPID FUCKING MISSLEADING ERROR TELLING ME THAT 1 IS NOT A INT BUT A TUPLE, AND NO SOLUTION ONLINE THAT I CAN FINE FIXES IT.
    
    cv2.imshow('image', img)
    k = cv2.waitKey(1000)  # 5000 = 5 seconds | 0 = indefently
    if k == 27:  # 27 == 'q'
        cv2.destroyAllWindows()
    cv2.destroyAllWindows()


def add_predicted_boxs(image, predictions):
    boxes = prediction[0]['boxes'].cpu().numpy()
    image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    for box in boxes:
        image = cv2.rectangle(image, (box[0], box[1]),
                                     (box[2], box[3]), (0, 0, 255), 1)
    return image
    
# load a model pre-trained pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress = True)

# replace the classifier with a new one, that has num_classes which is user-defined
num_classes = 1 
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


# load a pre-trained model for classification and return only the features
backbone = torchvision.models.mobilenet_v2(pretrained=True).features
# FasterRCNN needs to know the number of output channels in a backbone. 
# For mobilenet_v2, it's 1280 so we need to add it here
backbone.out_channels = 1280

# let's make the RPN generate 5 x 3 anchors per spatial location, with 5 different 
# sizes and 3 different aspect ratios. We have a Tuple[Tuple[int]] because each 
# feature map could potentially have different sizes and aspect ratios
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))

# let's define what are the feature maps that we will use to perform the region of 
# interest cropping, as well as the size of the crop after rescaling. If your 
# backbone returns a  Tensor, featmap_names is expected to be [0].
# More generally, the backbone should return an OrderedDict[Tensor], and 
# in featmap_names you can choose which feature maps to use.
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                output_size=7,
                                                sampling_ratio=2)

# put the pieces together inside a FasterRCNN model
model = FasterRCNN(backbone, num_classes=2,rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler, min_size=100, max_size=5000)
                    # FIXME: change size from (100,5000) to (800,1333)
                    # image_mean=[0.3297], image_std=[0.2566]

############################################################################

# use our dataset and defined transformations
dataset = DataturksFaceDataset(ROOT_DATA, get_transform(train=True))
dataset_test = DataturksFaceDataset(ROOT_DATA, get_transform(train=False))

# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=3, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=3, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)


# move model to the right device
model.to(device)
# construct an optimizer
# print([p for p in model.parameters() if p.requires_grad])

# *_, last = model.parameters()
# for param in model.parameters():
#     param.requires_grad = False
# last.requires_grad = True
params = [p for p in model.parameters() if p.requires_grad]
# params = [last]

# no_grad_params = [p for p in model.parameters() if not p.requires_grad]
# print(params)
# time.sleep(10000)
optimizer = torch.optim.SGD(params, lr=0.01,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=5,
                                               gamma=0.1)

# let's train it for 10 epochs
num_epochs = 1

if __name__ == "__main__":
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader,
                        device, epoch, 
                        print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    # pick one image from the test set
    # put the model in evaluation mode
    # model.eval()
    # for i in range(len(data_loader_test)):
    #     img, _ = dataset_test[i]
    #     with torch.no_grad(): 
    #         prediction = model([img.to(device)])

    #         print(prediction)
    #         # print(img.mul(255).permute(1, 2, 0).byte().numpy())
    #         img = img.mul(255).permute(1, 2, 0).byte().numpy()
            
    #         img = add_predicted_boxs(img, prediction)
    #         cv2.imshow("prediction", img)

    #         k = cv2.waitKey(0)  # 5000 = 5 seconds | 0 = indefently
    #         imageName = "predictions" + str(i) + ".png"
    #         # cv2.imwrite(imageName, img)
    #         if k == 27:
    #             cv2.destroyAllWindows()


    asyncio.run(main())
