
# coding: utf-8

# In[53]:


import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np
from matplotlib import pyplot as plt
from torch import optim as optm
import cv2
import h5py


# In[2]:


class Network(nn.Module):
    def __init__(self,spat_size,temp_size,layers):
        super(Network,self).__init__()
        self.inp = spat_size*spat_size
        self.op = self.inp*temp_size
        print(self.inp,self.op)
        self.inp_layer = nn.Linear(self.inp,self.op)
        self.hl1 = nn.Linear(self.op,self.op)
        self.hl2 = nn.Linear(self.op,self.op)
        self.hl3 = nn.Linear(self.op,self.op)
        self.hl4 = nn.Linear(self.op,self.op)
    
    def forward(self,inp_frame):
        fn_op = F.relu(self.inp_layer(inp_frame))
        fn_op = F.relu(self.hl1(fn_op))
        fn_op = F.relu(self.hl2(fn_op))
        fn_op = F.relu(self.hl3(fn_op))
        fn_op = F.relu(self.hl4(fn_op))
        return fn_op


# In[3]:


def init_weights(layer):
    if type(layer) == nn.Linear:
        n = layer.in_features
        scale = np.sqrt(1/n)
        layer.weight.data.uniform_(-scale,scale)
        layer.bias.data.zero_()


# In[4]:


device = t.device("cuda" if t.cuda.is_available() else "cpu")
var = np.random.binomial(1,0.5,size=[8,8]).reshape(1,64)
net = Network(8,16,4).to(device)
net.apply(init_weights)


# In[5]:


lr_net = 0.01
loss_fnc = nn.MSELoss()
optimizer = optm.SGD(net.parameters(),lr=lr_net,weight_decay=0.001,momentum=0.9)


# In[6]:


def read_video(video_link,frames_to_read):
        video = cv2.VideoCapture(video_link)
        video_frames=[]
        frames_read=0
        while video.isOpened():
                if frames_read == frames_to_read:
                        break;
                ret,frame = video.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #gray = cv2.normalize(gray.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
                gray = gray.astype('double')
                video_frames.append(gray)
                frames_read+=1
        return video_frames


# In[21]:


vid_frm = read_video('ValVideos/FlatAsAPancake.avi',16)


# In[22]:


vid_frm=np.asarray(vid_frm)


# In[24]:


cod = np.random.randint(0,2,vid_frm.shape)


# In[26]:


vid_cod = np.multiply(cod,vid_frm)


# In[37]:


sum_vid_cod = np.sum(vid_cod,0)


# In[44]:


v = t.tensor(sum_vid_cod[:8,:8].reshape(1,64),device=device).float()


# In[71]:


d=net(v).reshape(1024,1).reshape(8,8,16)


# In[55]:


file = h5py.File('train_data/trainData_1.h5')


# In[61]:


list(file.keys())


# In[65]:


label = file['label']


# In[69]:


plt.imshow(label[0].reshape(8,8,16)[:,:,0])

