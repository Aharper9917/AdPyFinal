# AdPyFinal
Advance Python Final
---------------------
To download the images for our data set you need to run the file getPhotos.py

* FasterRCNN_Finetuning_noAsync.py is our first attempt at the project.
* FasterRCNN_Finetuning_Async.py is the same as above, but while running the NN on the webcam feed with CV2 (aka openCV)
* facenet_noAsync.py is the first test with the fasenet model.
* facenet_Acync.py is the same as above but with asynchronous processing. The facenet model doesn't actually need the async; we just used it as a simple way to test our async code.
* facenet_webcam.py is the facenet model running on the camera feed without async.
* The files: coco_eval.py, coco_utils.py, engine.py, utils.py, transforms.py areall dependecies files for our first NN with the fasterRCNN model.

Please, feel free to email apharper@mavs.coloradomesa.edu if any question arise.


Group Members
-------------
Allen Harper, Darren Gleasson, Grant Kingma


Note
-------------
There is only one commit to the repo because my computer is the only one in our group that has a GPU. So, we had to run our model on said computer.



Abstract
-------------
We are planning to make a Machine Learning Algorithm for Facial Recognition, using PyTorch. We will get a large set of data to feed our NN, tweaking the hyperparameters until we get an optimal results for detecting human faces. We will be implementing this on embedded hardware such as a RaspberryPi and Jeston Nano.

UI Prototype
-------------
Epoch 0/24
train Loss: 0.6058 Acc: 0.6926
val Loss: 0.3300 Acc: 0.8562

Epoch 1/24
train Loss: 0.5581 Acc: 0.7951
val Loss: 0.2182 Acc: 0.9216

Epoch 2/24
train Loss: 0.6139 Acc: 0.7746
val Loss: 0.2058 Acc: 0.8954

Epoch 3/24
train Loss: 0.7887 Acc: 0.7295
val Loss: 0.5233 Acc: 0.8235

Epoch 4/24
train Loss: 0.6316 Acc: 0.7459
val Loss: 0.3267 Acc: 0.8758

Epoch 5/24
train Loss: 0.4477 Acc: 0.8320
val Loss: 0.7233 Acc: 0.7059

Epoch 6/24
train Loss: 0.4509 Acc: 0.8074
val Loss: 0.2602 Acc: 0.8954

Epoch 7/24
train Loss: 0.4134 Acc: 0.8320
val Loss: 0.2239 Acc: 0.9020

Epoch 8/24
train Loss: 0.3443 Acc: 0.8770
val Loss: 0.2326 Acc: 0.9216

Epoch 9/24
train Loss: 0.4378 Acc: 0.8033
val Loss: 0.2433 Acc: 0.9085

Epoch 10/24....

Training complete in 1m 7s
Best val Acc: 0.921569
