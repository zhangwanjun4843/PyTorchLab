from src.pytorchlab.datamodules.yoloVID.VIDDataset import *


make_path(r'E:\HANFENGSHUJU\Data\VID\train','example/yolov/train_new.npy')
make_path(r'E:\HANFENGSHUJU\Data\VID\train','example/yolov/val_new.npy')
import numpy as np
a= np.load('example/yolov/val_new.npy',allow_pickle=True)
print(a)