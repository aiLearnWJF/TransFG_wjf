#%% try
# from models.modeling import VisionTransformer, CONFIGS
# import torch

# config = CONFIGS["ViT-B_16"]
# config.split = 'non-overlap'

# config.hidden_size = 768
# config.transformer.mlp_dim = 3072

# # config.slide_step = args.slide_step

# # if args.dataset == "CUB_200_2011":
# #     num_classes = 200
# # elif args.dataset == "car":
# #     num_classes = 196
# # elif args.dataset == "nabirds":
# #     num_classes = 555
# # elif args.dataset == "dog":
# #     num_classes = 120
# # elif args.dataset == "INat2017":
# #     num_classes = 5089

# x = torch.rand(2,3,224,224)
# model = VisionTransformer(config, (224,224), zero_head=True, num_classes=1024,smoothing_value=0.)
# y = model(x)
# # torch.save(model,"1.pth")

#%% plot attenstion patch
from models.modeling import VisionTransformer, CONFIGS
import torch
import cv2
import numpy as np

config = CONFIGS["ViT-B_16"]
config.split = 'non-overlap'

config.hidden_size = 768
config.transformer.mlp_dim = 3072

model = VisionTransformer(config, (224,224), zero_head=True, num_classes=21843,smoothing_value=0.)
model.load_from(np.load("pretrained/imagenet21k+imagenet2012_ViT-B_16.npz"))

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

x = get_img_tensor("my_test/nonvehile3_2.jpg")
y = model(x)

att_patch = [  1,  72,  59, 196, 155,  12,  25,  63, 144,  39,  62,  49]
img = cv2.resize(cv2.imread("my_test/nonvehile3_2.jpg"),(224,224))
for i in range(14):
    for j in range(14):
        if i*14 + j  in att_patch:
            # import pdb;pdb.set_trace()
            img[i*16:(i+1)*(16),j*16:(j+1)*(16)] = 254
cv2.imwrite("viz.jpg",img)