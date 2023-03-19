import torch
from torch.utils import data
from data import TestDataset
from PIL.Image import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import numpy as np
import h5py
#from model import Model
#from HDP_Crowd_basic import *
from HDP_Crowd_basic_vgg_dilation import *
#from HDS import*
#from HDP_Crowd_Two_Attension_Module import *
#from HDP_Crowd_Attension_end_No_dilation import *
#from HDP_Crowd_basic_vgg_dilation import *
from HDP_Crowd_No_dilation_Attension_end import *
#from CASA_Crowd import DenseScaleNet
from Convnet_01 import *
import os

save_path = 'C:/Users/Naveed/Desktop/Datasets/venice_01/'
#save_path = 'C:/Users/Naveed/Desktop/CSRNet/CSRNet/HDP_Crowd_Venice'
test_dataset = TestDataset()
test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device=torch.device('cuda:0')

#model = Model().to(device)
model = Net().to(device)
#model = DenseScaleNet().to(device)
checkpoint = torch.load(os.path.join(save_path, 'checkpoint_best.pth'))
model.load_state_dict(checkpoint['model'])
print('Epoch:{} MAE:{} MSE:{}'.format(checkpoint['epoch'], checkpoint['mae'], checkpoint['mse']))

model.eval()
with torch.no_grad():
    mae, mse, count = 0.0, 0.0, 0
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        predict = model(images)

        print('predict:{:.2f} label:{:.2f}'.format(predict.sum().item(), labels.sum().item()))
        mae += torch.abs(predict.sum() - labels.sum()).item()
        mse += ((predict.sum() - labels.sum()) ** 2).item()
        count += 1

    mae /= count
    mse /= count
    mse = mse ** 0.5
    print('MAE:{} MSE:{}'.format(mae, mse))



from matplotlib import cm as c
#img = transform(Image.open('part_A/test_data/images/IMG_100.jpg').convert('RGB')).cuda()
#img = transform(Image.open('/home/naveedilyas/Dataset_preparation/ShanghaiTech/part_A/test_data/images/IMG_147.jpg').convert('RGB')).cuda()

#output = model(img.unsqueeze(0))
print("Predicted Count : ",int(predict.detach().cpu().sum().numpy()))
temp = np.asarray(predict.detach().cpu().reshape(predict.detach().cpu().shape[2],predict.detach().cpu().shape[3]))
plt.imshow(temp,cmap = c.jet)
plt.show()
temp = h5py.File('/home/naveedilyas/Dataset_preparation/ShanghaiTech/part_A/test_data/ground-truth/IMG_147.h5', 'r')
temp_1 = np.asarray(temp['density'])
plt.imshow(temp_1,cmap = c.jet)
print("Original Count : ",int(np.sum(temp_1)) + 1)
plt.show()
print("Original Image")
plt.imshow(plt.imread('/home/naveedilyas/Dataset_preparation/ShanghaiTech/part_A/test_data/images/IMG_147.jpg'))
plt.show()
