import os,torch,warnings,math
import argparse,datetime
import numpy as np
# from mGRNLayerDropout import *
from MGRN import *
from torch.utils.data import TensorDataset, DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings('ignore')
def si():
    TureLabel = torch.from_numpy(np.load('Label1024.npy')).long()
    LabelUpdate = np.load('LabelUpdate)10)3.npy')
    length,stride,epoch = 15,1,100
    Querytrainlabel = np.load('Querytrainlabel.npy')
    QueryData = LabelUpdate[Querytrainlabel]
    QueryLabel = TureLabel[Querytrainlabel].detach().numpy()
    num = (epoch-length)//stride
    Data = np.zeros((len(QueryLabel),num,length,8))
    Label = np.zeros((len(QueryLabel),num,8+1))
    for i in range(len(Label)):
        for n in range(num):
            Data[i,n] = QueryData[i,n*stride:n*stride+length]
            Temp = np.zeros((8+1))
            Temp[QueryLabel[i]] = 1
            Temp[8] = np.equal(QueryLabel[i],np.argmax(np.bincount(np.argmax(Data[i,n],1)))) and np.max(np.bincount(np.argmax(Data[i,n],1)))>=0.6*length
            Label[i,n] = Temp
    Data,Label = Data.reshape(-1,length,8),Label.reshape(-1,9)
    TrainData,TrainLabel=torch.from_numpy(Data).float(),torch.from_numpy(Label).float()
    TrainDataSet = TensorDataset(TrainData,TrainLabel)
    TrainDataSet = DataLoader(dataset=TrainDataSet, batch_size=32, shuffle=True, drop_last=False)
    return TrainDataSet

TrainDataSet = si()
# model_param_dict = {'model_name': 'mGRN',
#                     'n_rnn_units': 6,
#                     'n_layers': 1,
#                     'num_classes': 8,
#                     'batch_first': False,
#                     'device': device,
#                     'input_size_list':  [1,1,1,1,1,1,1,1],
#                     'size_of': 4}
# MGRN = get_model(**model_param_dict)
# Data1D = np.float32(np.random.uniform(1,50,(33,20,8)))
# Torch1D = torch.from_numpy(Data1D).to(device)
# Torch1D1 = Torch1D.permute(1, 0, 2)
# a = MGRN(Torch1D1)
MGRN = mGRNDropout(rnn_hidden_size=6, num_classes=9, device=device, input_size_list=[1,1,1,1,1,1,1,1],
                  size_of=11, dropouti = 0, dropoutw = 0, dropouto = 0,
                  keras_initialization = False, batch_first = False).to(device)
# mGRNDropout(rnn_hidden_size=6, num_classes=8, device=device, input_size_list=[1,1,1,1,1,1,1,1],
#                   size_of=11, dropouti = 0, dropoutw = 0, dropouto = 0,
#                   keras_initialization = False, batch_first = False).to(device)
MGRNOptimizer =  torch.optim.Adam(MGRN.parameters())
criterion = torch.nn.MSELoss()
for _, (MGRNinputs,MGRNlabel) in enumerate(TrainDataSet):
    MGRNinputs,MGRNlabel = MGRNinputs.to(device),MGRNlabel.to(device)
MGRNinputs1 = MGRNinputs.permute(1,0,2)
MGRNlogits = MGRN(MGRNinputs1)
# MGRNLoss = torch.zeros((1))
MGRNLoss = criterion(MGRNlogits,MGRNlabel)
# # MGRNLoss = torch.sqrt(torch.sum((MGRNlogits-MGRNlabel).pow(2)))
MGRNOptimizer.zero_grad()
MGRNLoss.backward()#
MGRNOptimizer.step()

# Blockslogits = Blocks(MGRNinputs)
# MGRNLoss = criterion(Blockslogits,MGRNlabel)
# print(BlocksOptimizer.zero_grad())
# print(MGRNLoss.backward())#retain_graph=True
# print(BlocksOptimizer.step())
