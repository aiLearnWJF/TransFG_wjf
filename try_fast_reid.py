import sys
sys.path.append('/vehicle/yckj3860/code/fast-reid-1.3.0/')

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer

from models.modeling import VisionTransformer, CONFIGS
import torch
import cv2
import numpy as np
# ┌────────────────────────────────────────────────────────────────────────┐
# │                           自定义模型 reid  start         
# └────────────────────────────────────────────────────────────────────────┘

def build_model():
    cfg = get_cfg()
    # cfg.merge_from_file("/vehicle/yckj3860/code/fast-reid-1.3.0/logs/nonmotor/resnest_wjfPubExp_all_download_4700_1009_cosl_f1000/config.yaml")
    # cfg.merge_from_file("/vehicle/yckj3860/code/fast-reid-1.3.0/logs/nonmotor/transfg_cq0416_2_cq0413_baseline_0301/config.yaml")
    cfg.merge_from_file("/home/yckj3860/code/fast-reid-1.3.0/logs/market1501/sbs_transfg/config.yaml")
    cfg.freeze()

    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False
    model = DefaultTrainer.build_model(cfg)

    # Checkpointer(model).load("/vehicle/yckj3860/code/fast-reid-1.3.0/logs/nonmotor/resnest_wjfPubExp_all_download_4700_1009_cosl_f1000/model_best.pth")  # load trained model
    # Checkpointer(model).load("/vehicle/yckj3860/code/fast-reid-1.3.0/logs/nonmotor/transfg_cq0416_2_cq0413_baseline_0301/model_best.pth")  # load trained model
    Checkpointer(model).load("/home/yckj3860/code/fast-reid-1.3.0/logs/market1501/sbs_transfg/model_best.pth")  # load trained model
    # repvgg test need this line
    # model.backbone.deploy(True)


    return model

model = build_model().to("cpu")
class ResnetFeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super(ResnetFeatureExtractor, self).__init__()
        self.model = model
        self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.pool = torch.nn.AdaptiveAvgPool2d((1,1))
        # self.pool = torch.nn.MaxPool2d((1,1))
                
    def __call__(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x)
        return x[:,:,0,0]
model_fea = ResnetFeatureExtractor(model)
model_fea.eval()
# x_input = torch.zeros((2,3,224,224))
# y = model_fea(x_input)
# import pdb;pdb.set_trace()

# ┌────────────────────────────────────────────────────────────────────────┐
# │                           自定义模型 reid  end         
# └────────────────────────────────────────────────────────────────────────┘
def normalize(img,to_rgb=True):
    # mean = np.array([[109.545, 105.783, 108.429]])
    # stdinv = np.array([[0.014923,0.0151096,0.0148982]])
    # mean = np.array([[0., 0., 0.]])
    # stdinv = np.array([[255, 255, 255]])
    mean = np.array([[123.675, 116.28, 103.53]])
    stdinv = np.array([[58.395, 57.12, 57.375]])
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB,img)  # inplace
    cv2.subtract(img, mean,img)  # inplace
    img = img/stdinv  # inplace
    img = img.astype(np.float32)
    return img
def resize_crop_to_tensor(img,resize_shape,crop_shape):
    img = cv2.resize(img,(resize_shape,resize_shape))
    dis = resize_shape-crop_shape
    if True:
        img = img[dis:dis+crop_shape,dis:dis+crop_shape,:]
    img = np.expand_dims(img,axis=0)
    img = img.transpose(0,3,1,2)
    img_tensor = torch.from_numpy(img)
    return img_tensor

def get_img_tensor(path=""):
    img = cv2.imread(path).astype(np.float32)
    img = normalize(img)
    img = resize_crop_to_tensor(img,224,224)
    return img

x = get_img_tensor("my_test/person2.png")
y = model(x)

att_patch =[182, 119, 119,  29, 192,  44, 116,  29, 128, 186, 112, 186]
img = cv2.resize(cv2.imread("my_test/person2.png"),(224,224))
for i in range(14):
    for j in range(14):
        if i*14 + j  in att_patch:
            # import pdb;pdb.set_trace()
            img[i*16:(i+1)*(16),j*16:(j+1)*(16)] = 254
cv2.imwrite("viz.jpg",img)