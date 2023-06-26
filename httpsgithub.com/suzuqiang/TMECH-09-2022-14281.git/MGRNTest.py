import os,torch,warnings,sys
import numpy as np
from Schrobine import *
import matplotlib.pyplot as plt
global args
warnings.filterwarnings('ignore')
SetSeed()
args = Parse()
class DefineMyDataCenter():
    def __init__(self,Label,args,**kwargs):
        self.args = args
        self.TureLabel = torch.from_numpy(np.load('Label1024.npy')).long()
        self.ActuralLabel = torch.from_numpy(np.load('Label1024.npy')).long()
        self.LabelUpdate = np.load('LabelUpdate4)25)16)38.npy') #6400*100*8
    def GetMGRNTrainData(self):
        Querytrainlabel = np.load('Querytrainlabel.npy')
        QueryData = self.LabelUpdate[Querytrainlabel]
        QueryLabel = self.TureLabel[Querytrainlabel].detach().numpy()
        num = (self.args.PreTrainProNetEpoch-self.args.length)//self.args.stride
        Data = np.zeros((len(QueryLabel),num,self.args.length,self.args.classnum),dtype='float32')
        Label = np.zeros((len(QueryLabel),num,self.args.classnum+1),dtype='float32')
        for i in range(len(Label)):
            for n in range(num):
                Data[i,n] = QueryData[i,n*self.args.stride:n*self.args.stride+self.args.length]
                Temp = np.zeros((self.args.classnum+1))
                Temp[QueryLabel[i]] = np.equal(QueryLabel[i],np.argmax(np.bincount(np.argmax(Data[i,n],1))))
                Temp[self.args.classnum] = np.argmax(np.bincount(np.argmax(Data[i,n],1)))
                Label[i,n] = Temp
        Data,Label = Data.reshape(-1,self.args.length,8),Label.reshape(-1,9)
        TrainData,TrainLabel=torch.from_numpy(Data).float(),torch.from_numpy(Label).float()
        TrainDataSet = TensorDataset(TrainData,TrainLabel)
        TrainDataSet = DataLoader(dataset=TrainDataSet, batch_size=self.args.batch, shuffle=True, drop_last=False)
        return TrainDataSet
    def GetUnlabeledData1(self):
        Unlabeledtrainlabel = np.load('Unlabeledtrainlabel.npy')
        UnlabeledData = self.LabelUpdate[Unlabeledtrainlabel]
        UnlabeledLabel = self.TureLabel[Unlabeledtrainlabel].detach().numpy()
        UnlabeledData,UnlabeledLabel=torch.from_numpy(UnlabeledData).float(),torch.from_numpy(UnlabeledLabel).float()
        UnlabeledDataSet = TensorDataset(UnlabeledData,UnlabeledLabel)
        UnlabeledDataSet = DataLoader(dataset=UnlabeledDataSet, batch_size=200, shuffle=False, drop_last=False)
        return UnlabeledDataSet
    def GetUnlabeledData2(self):
        Unlabeledtrainlabel = np.load('Unlabeledtrainlabel.npy')
        UnlabeledData = self.LabelUpdate[Unlabeledtrainlabel]
        UnlabeledLabel = self.TureLabel[Unlabeledtrainlabel].detach().numpy()
        Data,Label = UnlabeledData[:,self.args.PreTrainProNetEpoch-self.args.length:self.args.PreTrainProNetEpoch],\
                     np.zeros((len(UnlabeledLabel),self.args.classnum+1),dtype='float32')
        for i in range(len(UnlabeledLabel)):
            Temp = np.zeros((self.args.classnum+1))
            Temp[UnlabeledLabel[i]] = np.equal(UnlabeledLabel[i],np.argmax(np.bincount(np.argmax(Data[i],1))))
            Temp[self.args.classnum] = np.argmax(np.bincount(np.argmax(Data[i],1)))
            Label[i] = Temp
        UnlabeledData,UnlabeledLabel=torch.from_numpy(Data).float(),torch.from_numpy(Label).float()
        UnlabeledDataSet = TensorDataset(UnlabeledData,UnlabeledLabel)
        UnlabeledDataSet = DataLoader(dataset=UnlabeledDataSet, batch_size=200, shuffle=False, drop_last=False)
        return UnlabeledDataSet
    def GetUnlabeledData3(self):
        Unlabeledtrainlabel = np.load('Unlabeledtrainlabel.npy')
        UnlabeledData = self.LabelUpdate[Unlabeledtrainlabel]
        UnlabeledLabel = self.TureLabel[Unlabeledtrainlabel].detach().numpy()
        Data = UnlabeledData[:,self.args.PreTrainProNetEpoch-self.args.length:self.args.PreTrainProNetEpoch]
        num = np.zeros(3)
        for i in range(len(UnlabeledLabel)):
            temp = np.argmax(np.bincount(np.argmax(Data[i],1)))
            if np.max(np.bincount(np.argmax(Data[i],1))) > 0.75*self.args.length:
                num[1] += 1
                if UnlabeledLabel[i] == temp:
                    num[2] += 1
            if UnlabeledLabel[i] == temp:
                num[0] += 1
        print(num[2]/num[1])
        print(num[1]/len(UnlabeledLabel))
        # UnlabeledData,UnlabeledLabel=torch.from_numpy(Data).float(),torch.from_numpy(Label).float()
        # UnlabeledDataSet = TensorDataset(UnlabeledData,UnlabeledLabel)
        # UnlabeledDataSet = DataLoader(dataset=UnlabeledDataSet, batch_size=200, shuffle=False, drop_last=False)
        # return UnlabeledDataSet
    def GetQueryData(self):
        Unlabeledtrainlabel = np.load('Querytrainlabel.npy')
        UnlabeledData = self.LabelUpdate[Unlabeledtrainlabel]
        UnlabeledLabel = self.TureLabel[Unlabeledtrainlabel].detach().numpy()
        Data,Label = UnlabeledData[:,self.args.PreTrainProNetEpoch-self.args.length:self.args.PreTrainProNetEpoch],\
                     np.zeros((len(UnlabeledLabel),self.args.classnum+1),dtype='float32')
        for i in range(len(UnlabeledLabel)):
            Temp = np.zeros((self.args.classnum+1))
            Temp[UnlabeledLabel[i]] = np.equal(UnlabeledLabel[i],np.argmax(np.bincount(np.argmax(Data[i],1))))
            Temp[self.args.classnum] = np.argmax(np.bincount(np.argmax(Data[i],1)))
            Label[i] = Temp
        UnlabeledData,UnlabeledLabel=torch.from_numpy(Data).float(),torch.from_numpy(Label).float()
        UnlabeledDataSet = TensorDataset(UnlabeledData,UnlabeledLabel)
        UnlabeledDataSet = DataLoader(dataset=UnlabeledDataSet, batch_size=200, shuffle=False, drop_last=False)
        return UnlabeledDataSet
MyDataCenter = DefineMyDataCenter(1,args)
# MyDataCenter.GetUnlabeledData3()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MGRN = mGRNDropout(rnn_hidden_size=6,num_classes=9,device=device,input_size_list=[1,1,1,1,1,1,1,1],
                  size_of=11,dropouti=0,dropoutw=0,dropouto=0,keras_initialization=False,batch_first=False)
MGRN.to(device)
def Tain(device,MGRN,MyDataCenter):
    TrainDataSet = MyDataCenter.GetMGRNTrainData()
    Criterion = torch.nn.MSELoss()
    learning_rate = 0.001#学习率
    for epoch in range(args.PreTrainMGRNEpoch):
        Num = np.zeros(2)
        lr = learning_rate * np.exp(-1*epoch/10)
        MGRNOptimizer = torch.optim.Adam(MGRN.parameters(),lr)
        for _, (MGRNinputs,MGRNlabel) in enumerate(TrainDataSet):
            MGRNinputs,MGRNlabel = MGRNinputs.to(device),MGRNlabel.to(device)
            MGRNinputs = MGRNinputs.permute(1,0,2)
            MGRNlogits = MGRN(MGRNinputs)#torch.Size([15, 24, 8])
            MGRNLoss = Criterion(MGRNlogits,MGRNlabel)
            MGRNOptimizer.zero_grad()
            MGRNLoss.backward()
            MGRNOptimizer.step()
            Num[0] += MGRNLoss.cpu().detach().numpy()*len(MGRNlabel)
            Num[1] += len(MGRNlabel)
        Temp = "Epoch:"+str(epoch+1)+" MSE损失:" +str(Num[0]/Num[1])
        print("\r" + Temp, end="")
    print('')
    torch.save(MGRN.state_dict(),'MGRN.pkl')
    return MGRN
# MGRN = Tain(device,MGRN,MyDataCenter)

print('训练')
MGRN.load_state_dict(torch.load('MGRN.pkl'))
UnlabeledData = MyDataCenter.GetQueryData()
Num = np.zeros(6)
for _, (MGRNinputs,MGRNlabel) in enumerate(UnlabeledData):
    MGRNinputs,MGRNlabel = MGRNinputs.to(device),MGRNlabel.to(device)
    MGRNlogits = MGRN(MGRNinputs.permute(1,0,2))
    a = torch.argmax(MGRNlogits[:,:8],1)[torch.where(MGRNlogits[:,8]>0.5)]
    b = MGRNlabel[torch.where(MGRNlogits[:,8]>0.5)][:,-1]
    c = torch.argmax(MGRNlogits[:,:8],1)[torch.where(MGRNlogits[:,8]<0.5)]
    e = torch.argmax(MGRNlogits[:,:8],1)
    f = MGRNlabel[:,-1]
    if len(c) > 0:
        d = MGRNlabel[torch.where(MGRNlogits[:,8]<0.5)][:,-1].long()
        Num[2] += torch.sum(c!=d).long().cpu().detach().numpy()
    Num[0] += torch.sum(a==b).long().cpu().detach().numpy()
    Num[1] += len(a)
    Num[3] += len(c)
    Num[4] += torch.sum(e==f).long().cpu().detach().numpy()
    Num[5] += len(e)
print('选出的准确率: '+str(Num[0]/Num[1]))
print('排除的准确率: '+str(Num[2]/Num[3]))
print('正确挑选的: '+str(Num[4]/Num[5]))
print('\n\n无标签')
UnlabeledData = MyDataCenter.GetUnlabeledData2()
Num = np.zeros(6)
for _, (MGRNinputs,MGRNlabel) in enumerate(UnlabeledData):
    MGRNinputs,MGRNlabel = MGRNinputs.to(device),MGRNlabel.to(device)
    MGRNlogits = MGRN(MGRNinputs.permute(1,0,2))
    a = torch.argmax(MGRNlogits[:,:8],1)[torch.where(MGRNlogits[:,8]>0.5)]
    b = MGRNlabel[torch.where(MGRNlogits[:,8]>0.5)][:,-1]
    c = torch.argmax(MGRNlogits[:,:8],1)[torch.where(MGRNlogits[:,8]<0.5)]
    e = torch.argmax(MGRNlogits[:,:8],1)
    f = MGRNlabel[:,-1]
    if len(c) > 0:
        d = MGRNlabel[torch.where(MGRNlogits[:,8]<0.5)][:,-1].long()
        Num[2] += torch.sum(c!=d).long().cpu().detach().numpy()
    Num[0] += torch.sum(a==b).long().cpu().detach().numpy()
    Num[1] += len(a)
    Num[3] += len(c)
    Num[4] += torch.sum(e==f).long().cpu().detach().numpy()
    Num[5] += len(e)
print('选出的准确率: '+str(Num[0]/Num[1]))
print('排除的准确率: '+str(Num[2]/Num[3]))
print('正确挑选的: '+str(Num[4]/Num[5]))
# # UnlabeledData = MyDataCenter.GetUnlabeledData2()
# # for _, (MGRNinputs,MGRNlabel) in enumerate(UnlabeledData):
# #     MGRNinputs,MGRNlabel = MGRNinputs.to(device),MGRNlabel.to(device)

