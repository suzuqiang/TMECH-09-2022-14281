import os,torch,warnings,sys
import numpy as np
from Schrobine import *
import matplotlib.pyplot as plt
global args
warnings.filterwarnings('ignore')
SetSeed()
class ProNet(torch.nn.Module):
    def __init__(self,args,**kwargs):
        super(ProNet, self).__init__( **kwargs)
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        SupportTrainDataSet,self.QueryTrainDataSet,self.UnlabeledTrainDataSet,self.TestDataSet,self.MyDataCenter = ProNetGetDataSet(self.args)
        self.blocks = torch.nn.Sequential(GetSomeLayer('Blocks1DTorch',1,512),  GetSomeLayer('Blocks1DTorch',512,256),
                                          GetSomeLayer('Blocks1DTorch',256,256),GetSomeLayer('Blocks1DTorch',256,128),
                                          GetSomeLayer('Blocks1DTorch',128,128),GetSomeLayer('Blocks1DTorch',128,32),
                                          torch.nn.Flatten())
        self.SWN = torch.nn.Sequential( torch.nn.Linear(in_features = 1024, out_features = 128),
                                        torch.nn.BatchNorm1d(num_features=128),torch.nn.ReLU(),
                                        torch.nn.Linear(in_features = 128, out_features = 2),
                                        torch.nn.Softmax(1) )
        for _, (supportinputs,supportlabel) in enumerate(SupportTrainDataSet):
            self.supportinputs,self.supportlabel = supportinputs.cuda(),self.MyDataCenter.GetLabel(supportlabel).cuda()
        self.Prototype = np.zeros((self.args.classnum,512))
        self.ProNetOptimizerLearingRate = 0.001
        self.ProNetOptimizer = torch.optim.Adam(self.blocks.parameters(),self.ProNetOptimizerLearingRate)
        self.SWNOptimizer = torch.optim.Adam(self.SWN.parameters(),self.SWNOptimizerLearingRate)
        
        print('随机数：'+self.args.RandomNum[:-4])
        print('shot: {0} support: {1} test: {2}. '.format(self.args.shot,self.args.supportusenum,self.args.testnum)) 
    def GetPro(self):
        supportlogits = self.blocks(self.supportinputs)
        for i in range(self.args.classnum):
            self.Prototype[i] = torch.mean(supportlogits[torch.where(self.supportlabel==i)],0).cpu().detach().numpy()
    def TrainSWN(self,logits,label,):
        Prototype = torch.from_numpy(self.Prototype).cuda()
        num = 0
        for i in range(len(label)):
            logits = SWN(torch.cat((Prototype,logits[i].repeat((8,1))),1))
            label = 
            loss = GetLoss(logits,label,self.device) if num==0 else loss+GetLoss(logits,label,self.device)
            num += 1
        self.SWNOptimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.SWNOptimizer.step()
    def GetSoftMax(self,logits,batch):
        tempdis = torch.zeros((batch,self.args.classnum),requires_grad=True).cuda()
        for i in range(batch):
            temp = torch.exp(-torch.sqrt(torch.sum((torch.from_numpy(self.Prototype).cuda()-logits[i]).pow(2),1)))
            tempdis[i] = temp/torch.sum(temp)
        tempdis = tempdis.cuda()
        return tempdis
    def SaveModel(self,Name):
        torch.save(self.blocks.state_dict(),Name)
    def SaveMyDataCenter(self,Name):
        np.save(Name,self.MyDataCenter.LabelUpdate)
    def LoadPreMyDataCenter(self,Name):
        self.MyDataCenter.LabelUpdate,self.MyDataCenter.LabelUpdateIndex = np.load(Name),\
            self.args.PreTrainProNetEpoch*np.ones(len(self.MyDataCenter.TureLabel),dtype='int32')
    def LoadPreModel(self,Name):
        self.blocks.load_state_dict(torch.load(Name))
        temp = args.PreTrainProNetEpoch
        args.PreTrainProNetEpoch = 1
        _=Model.PreTrainProNet()
        args.PreTrainProNetEpoch = temp
    def LoadPLLMyDataCenter(self,Name):
        self.MyDataCenter.LabelUpdate,self.MyDataCenter.LabelUpdateIndex = np.load(Name),\
            (self.args.PreTrainProNetEpoch+self.args.PLLEpoch)*np.ones(len(self.MyDataCenter.TureLabel),dtype='int32')
    def LoadPLLModel(self,Name):
        self.blocks.load_state_dict(torch.load(Name))
        temp = self.args.PLLEpoch
        self.args.PLLEpoch = 1
        _=self.SSLProNetOld1()
        self.args.PLLEpoch = temp
    def PreTrainProNet(self):
        Begain,Over = ('===PreTrainProNet开始').ljust(50,'='),('===PreTrainProNet结束').ljust(50,'=')
        print(Begain)
        History = np.zeros((self.args.PreTrainProNetEpoch,5))
        for epoch in range(self.args.PreTrainProNetEpoch):
            if self.args.PreTrainProNetEpoch == 1:
                learingratetemp = self.ProNetOptimizerLearingRate * (np.exp(-3))
            else:
                learingratetemp = self.ProNetOptimizerLearingRate * (np.exp(-3*epoch/self.args.PreTrainProNetEpoch))
            self.ProNetOptimizer = torch.optim.Adam(self.blocks.parameters(),learingratetemp)
            self.SWNOptimizer = torch.optim.Adam(self.SWN.parameters(),learingratetemp)
            self.GetPro()
            for step, (queryinputs,querylabel) in enumerate(self.QueryTrainDataSet):
                queryinputs,querylabel = queryinputs.to(self.device),self.MyDataCenter.GetLabel(querylabel).to(self.device)
                queryfeature = self.blocks(queryinputs)
                querylogits = self.GetSoftMax(queryfeature,querylabel.size()[0])
                if step ==0:
                    loss = GetLoss(querylogits,querylabel,self.device)
                else:
                    loss += GetLoss(querylogits,querylabel,self.device)
            self.ProNetOptimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.ProNetOptimizer.step()
            self.GetPro()
            for _, (queryinputs,querylabel) in enumerate(self.QueryTrainDataSet):
                temp = querylabel
                queryinputs,querylabel = queryinputs.to(self.device),self.MyDataCenter.GetLabel(querylabel).to(self.device)
                queryfeature = self.blocks(queryinputs)
                querylogits = self.GetSoftMax(queryfeature,querylabel.size()[0])
                queryacc,queryloss = GetAccuracy(querylogits,querylabel),GetLoss(querylogits,querylabel,self.device)
                self.MyDataCenter.GetUpdate(temp.detach().numpy(),querylogits.cpu().detach().numpy())
                History[epoch,0] += queryacc
                History[epoch,1] += querylabel.size()[0]
                History[epoch,2] += queryloss.cpu().detach().numpy()
            Queryacc,Queryloss = 100*History[epoch,0]/History[epoch,1],History[epoch,2]/History[epoch,1]
            for _, (testinputs,testlabel) in enumerate(self.TestDataSet):
                testinputs,testlabel = testinputs.to(self.device),self.MyDataCenter.GetLabel(testlabel).to(self.device)
                testfeature = self.blocks(testinputs)
                testlogits = self.GetSoftMax(testfeature,testlabel.size()[0])
                testacc = GetAccuracy(testlogits,testlabel)
                History[epoch,3] += testacc
                History[epoch,4] += testlabel.size()[0]
            TestAcc = 100*History[epoch,3]/History[epoch,4]
            Temp = "Epoch:"+str(epoch+1)+" 查询集损失:" +str(round(Queryloss,2))+" 查询集精度:"+str(round(Queryacc,2))+" 测试精度:"+ str(round(TestAcc,2))
            print("\r" + Temp, end="") 
            # print(Temp) 
            for _, (unlabeledinputs,unlabeledlabel) in enumerate(self.UnlabeledTrainDataSet):
                unlabeledinputs = unlabeledinputs.to(self.device)
                unlabeledfeature = self.blocks(unlabeledinputs)
                unlabeledlogits = self.GetSoftMax(unlabeledfeature,unlabeledlabel.size()[0])
                self.MyDataCenter.GetUpdate(unlabeledlabel.detach().numpy(),unlabeledlogits.cpu().detach().numpy())
        print('')
        print(Over)
        return History
    def SSLProNet(self):
        Begain,Over = ('===SSLProNet开始').ljust(50,'='),('===SSLProNet结束').ljust(50,'=')
        print(Begain)
        History = np.zeros(self.args.PLLEpoch)
        ST = (self.args.PreTrainProNetEpoch+self.args.PLLEpoch)
        for epoch in range(self.args.PLLEpoch):
            learingratetemp = self.ProNetOptimizerLearingRate * (np.exp(-3*epoch/self.args.PLLEpoch))
            self.ProNetOptimizer = torch.optim.Adam(self.blocks.parameters(),learingratetemp)
            for _, (queryinputs,querylabel) in enumerate(self.QueryTrainDataSet):
                self.GetPro()
                queryinputs,querylabel = queryinputs.to(self.device),self.MyDataCenter.GetLabel(querylabel).to(self.device)
                queryfeature = self.blocks(queryinputs)
                querylogits = self.GetSoftMax(queryfeature,querylabel.size()[0])
                loss = GetLoss(querylogits,querylabel,self.device)
                self.ProNetOptimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.ProNetOptimizer.step()
            historyloss = []
            for _, (unlabeledinputs,unlabeledlabel) in enumerate(self.UnlabeledTrainDataSet):
                unlabeledinputs,unlabeledlabel = unlabeledinputs.to(self.device),self.MyDataCenter.GetLabel(unlabeledlabel).to(self.device)
                unlabeledfeature = self.blocks(unlabeledinputs)
                unlabeledlogits = self.GetSoftMax(unlabeledfeature,unlabeledlabel.size()[0])
                for i in range(len(unlabeledlabel)):
                    temp = torch.log10(unlabeledlogits[i,unlabeledlabel[i].cpu().detach().numpy()])
                    historyloss.append(temp.cpu().detach().numpy())
            historyloss = np.array(historyloss)
            lossmax,lossmin = np.max(historyloss),np.mim(historyloss)
            Weights = (lossmax - historyloss) / (lossmax-lossmin)
            temp = 0
            for _, (unlabeledinputs,unlabeledlabel) in enumerate(self.UnlabeledTrainDataSet):
                weighttemp = Weights[temp:temp+len(unlabeledlabel)]
                temp += len(unlabeledlabel)
                Idex,LabelIndex,pseudolabel = self.MyDataCenter.GetPseudoLabel(unlabeledlabel)
                weighttemp = weighttemp[Idex]
                if len(Idex)>2:
                    unlabeledinputs,pseudolabel = unlabeledinputs[Idex].to(self.device),pseudolabel.to(self.device)
                    unlabeledfeature = self.blocks(unlabeledinputs)
                    self.UpdateWCProForSSLProNet(weighttemp,unlabeledfeature,pseudolabel)
                    unlabeledlogits = self.GetSoftMax(unlabeledfeature,pseudolabel.size()[0])
                    loss = GetLoss(unlabeledlogits,pseudolabel,self.device)
                    self.ProNetOptimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    self.ProNetOptimizer.step()
            History[epoch] = self.SSLTest(epoch)
        print(Over)
        return History
if __name__ == '__main__':
    args = Parse()
    Model = ProNet(args)
    Model.cuda()
    PreName = 'SaveModel/PreProNet'+')'+\
            args.RandomNum[:-4]+')'+str(args.shot)+')'+\
            str(args.supportusenum)+')'+str(args.PreTrainProNetEpoch)+')'+\
            str(args.PLLEpoch)
    PLLName = 'SaveModel/PLLProNet'+')'+\
            args.RandomNum[:-4]+')'+str(args.shot)+')'+\
            str(args.supportusenum)+')'+str(args.PreTrainProNetEpoch)+')'+\
            str(args.PLLEpoch)
    PreModelName = PreName +'.pkl'
    PreMyDataCenterName = PreName + '.npy'
    PLLModelName = PLLName +'.pkl'
    PLLMyDataCenterName = PLLName + '.npy'
    PreTrainHistoryName = 'SaveModel/ProNet)PreTrainHistory'+')'+\
            args.RandomNum[:-4]+')'+str(args.shot)+')'+\
            str(args.supportusenum)+')'+str(args.PreTrainProNetEpoch)+')'+\
            str(args.PLLEpoch) + '.npy'
    PLLTrainHistoryName = 'SaveModel/ProNet)PLLTrainHistory'+')'+\
            args.RandomNum[:-4]+')'+str(args.shot)+')'+\
            str(args.supportusenum)+')'+str(args.PreTrainProNetEpoch)+')'+\
            str(args.PLLEpoch) + '.npy'
    PreTrainHistoryDrowName = ['Correct Query','All Query','Query Loss',\
                               'Correct Test','All Test']
    PLLTrainHistoryDrowName = ['Correct Query','All Query','Query Loss',\
                               'Correct Unlabeled','All Unlabeled','Unlabeled Loss','ALL PseudoLabel','Correct PseudoLabel',\
                               'Correct Test','All Test']
    #%%预训练
    if os.path.exists(PreModelName) and os.path.exists(PreMyDataCenterName):
        Model.LoadPreModel(PreModelName)
        Model.LoadPreMyDataCenter(PreMyDataCenterName)
        PreTrainHistory = np.load(PreTrainHistoryName)
    else:
        PreTrainHistory = Model.PreTrainProNet()
        Model.SaveModel(PreModelName)
        Model.SaveMyDataCenter(PreMyDataCenterName)
        np.save(PreTrainHistoryName,PreTrainHistory)
    








