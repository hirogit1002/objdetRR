import numpy as np
from darkflow.net.build import TFNet
import argparse
import cv2

parser = argparse.ArgumentParser(description='Running')
parser.add_argument('--model','-m', default="cfg/tiny-yolo_custom.cfg", type=str)
parser.add_argument('--weight','-w', default="bin/yolo-tiny.weights", type=str)
parser.add_argument('--ckpt','-c', default=-1, type=int)
parser.add_argument('--load','-l', default="c", type=str)
parser.add_argument('--annotation','-a', default="/content/pascal/VOCdevkit/ANN", type=str)
parser.add_argument('--dataset','-d', default="/content/pascal/VOCdevkit/IMG", type=str)
parser.add_argument('--imgdir','-i', default="/content/drive/My\ Drive/objdetrr/test/_0001.png", type=str)
parser.add_argument('--train','-t', default=1, type=int)
parser.add_argument('--batch_size','-b', default=8, type=int)
parser.add_argument('--gpu','-g', default= 1., type=float)
parser.add_argument('--epochs','-e', default= 100, type=int)
parser.add_argument('--learning_late','-r', default= 1e-4, type=float)
parser.add_argument('--threshold','-th', default= 0.01, type=float)
args = parser.parse_args()

options_train_weight = {"model": args.model, 
           "load": args.weight,
           "batch": args.batch_size,
           "epoch": args.epochs,
           "gpu": args.gpu,
           "train": True,
           "annotation": args.annotation,
           "dataset": args.dataset}

options_train_ckpt = {"model": args.model, 
           "load": args.ckpt,
           "batch": args.batch_size,
           "epoch": args.epochs,
           "gpu": args.gpu,
           "train": True,
           "annotation": args.annotation,
           "dataset": args.dataset}

options_train_null = {"model": args.model, 
           "batch": args.batch_size,
           "epoch": args.epochs,
           "gpu": args.gpu,
           "train": True,
           "annotation": args.annotation,
           "dataset": args.dataset}

options_predict_ckpt = {"model": args.model, 
           "load": args.ckpt,
           "batch": args.batch_size,
           "imgdir": args.imgdir,
           "threshold": args.threshold,
           "gpu": args.gpu}

options_predict_weight = {"model": args.model, 
           "load": args.weight,
           "batch": args.batch_size,
           "imgdir": args.imgdir,
           "threshold": args.threshold,
           "gpu": args.gpu}

load_type = {"n":options_train_null,"c":options_train_ckpt,"w":options_train_weight}
load_type_predict = {"c":options_predict_ckpt,"w":options_predict_weight}

if(args.train):
    tfnet = TFNet(load_type[args.load])
    if(args.load=="c"):
        tfnet.load_from_ckpt()
    tfnet.train()
else :
    tfnet = TFNet(load_type_predict[args.load])
    if(args.load=="c"):
        tfnet.load_from_ckpt()
    imgcv = cv2.imread(args.imgdir)
    result = tfnet.return_predict(imgcv)
    print(result)
