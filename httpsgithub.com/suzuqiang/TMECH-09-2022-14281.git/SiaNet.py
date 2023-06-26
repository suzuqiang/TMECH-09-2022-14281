import os,torch,warnings,sys
import numpy as np
from Schrobine import *
import matplotlib.pyplot as plt
global args
warnings.filterwarnings('ignore')
SetSeed()


class SiaNet(torch.nn.Module):
    def __init__(self,args,**kwargs):
        super(SiaNet, self).__init__( **kwargs)
        self.args = args
        SupportTrainDataSet,self.TrainDataSet,self.UnlabeledTrainDataSet,self.TestDataSet,self.MyDataCenter = SiaNetGetDataSet(self.args)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.Blocks = torch.nn.Sequential(GetSomeLayer('Blocks1DTorch',1,512),  GetSomeLayer('Blocks1DTorch',512,256),
                                          GetSomeLayer('Blocks1DTorch',256,256),GetSomeLayer('Blocks1DTorch',256,128),
                                          GetSomeLayer('Blocks1DTorch',128,128),GetSomeLayer('Blocks1DTorch',128,32),
                                          torch.nn.Flatten()).to(self.device)
        for _, (supportinputs,supportlabel) in enumerate(SupportTrainDataSet):
            self.supportinputs,self.supportlabel = supportinputs.to(self.device),self.MyDataCenter.GetLabel(supportlabel).to(self.device)
        self.SiaNetOptimizerLearingRate = 0.001
        self.SiaNetOptimizer = torch.optim.Adam(self.Blocks.parameters(),self.SiaNetOptimizerLearingRate)
    def TrainSiaNet(self):
        Begain,Over = ('===TrainSiaNet').ljust(50,'='),('===TrainSiaNet').ljust(50,'=')
        # print(Begain)
        History = np.zeros(self.args.PreTrainProNetEpoch)
        for epoch in range(self.args.PreTrainProNetEpoch):
            learingratetemp = self.SiaNetOptimizerLearingRate * (np.exp(-3*epoch/self.args.PreTrainProNetEpoch))
            self.SiaNetOptimizer = torch.optim.Adam(self.Blocks.parameters(),learingratetemp)
            for _, (traininputs,trainlabel) in enumerate(self.TrainDataSet):
                traininputs,trainlabel = traininputs.to(self.device),self.MyDataCenter.GetLabel(trainlabel).to(self.device)
                tempresult = torch.zeros((len(trainlabel),self.args.classnum)).to(self.device)
                for n in range(self.args.classnum):
                    test = self.Blocks(traininputs)-self.Blocks(self.supportinputs[torch.where(self.supportlabel==n)])
                    tempresult[:,n] = torch.sqrt(torch.sum(test.pow(2),1))/512
                tempresult = torch.softmax(tempresult,0)
                loss = GetLoss(tempresult,trainlabel,self.device)
                self.SiaNetOptimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.SiaNetOptimizer.step()
            Num = np.zeros(2)
            for _, (traininputs,trainlabel) in enumerate(self.TrainDataSet):
                traininputs,trainlabel = traininputs.to(self.device),self.MyDataCenter.GetLabel(trainlabel).to(self.device)
                tempresult = torch.zeros((len(trainlabel),self.args.classnum)).to(self.device)
                for n in range(self.args.classnum):
                    test = self.Blocks(traininputs)-self.Blocks(self.supportinputs[torch.where(self.supportlabel==n)])
                    tempresult[:,n] = torch.sqrt(torch.sum(test.pow(2),1))/512
                tempresult = torch.softmax(tempresult,0)
                Num[0] += torch.sum(torch.argmax(tempresult,1)==trainlabel)
                Num[1] += trainlabel.size()[0]
            Trainacc = 100*Num[0]/Num[1]
            Num = np.zeros(2)
            for _, (testinputs,testlabel) in enumerate(self.TestDataSet):
                testinputs,testlabel = testinputs.to(self.device),self.MyDataCenter.GetLabel(testlabel).to(self.device)
                tempresult = torch.zeros((len(testlabel),self.args.classnum)).to(self.device)
                for n in range(self.args.classnum):
                    test = self.Blocks(testinputs)-self.Blocks(self.supportinputs[torch.where(self.supportlabel==n)])
                    tempresult[:,n] = torch.sqrt(torch.sum(test.pow(2),1))/512
                tempresult = torch.softmax(tempresult,0)   
                Num[0] += torch.sum(torch.argmax(tempresult,1)==testlabel)
                Num[1] += testlabel.size()[0]
            TestAcc = 100*Num[0]/Num[1]
            Temp = "Epoch:"+str(epoch+1)+" 查询集精度:"+str(round(Trainacc,2))+" 测试精度:"+ str(round(TestAcc,2))
            print("\r" + Temp, end="") 
            # print(Temp) 
            # for _, (unlabeledinputs,unlabeledlabel) in enumerate(self.UnlabeledTrainDataSet):
            #     unlabeledinputs = unlabeledinputs.to(self.device)
            #     unlabeledfeature = self.blocks(unlabeledinputs)
            #     unlabeledlogits = self.GetSoftMax(unlabeledfeature,unlabeledlabel.size()[0])
            #     self.MyDataCenter.GetUpdate(unlabeledlabel.detach().numpy(),unlabeledlogits.cpu().detach().numpy())
            History[epoch] = TestAcc
        print('shot: {0} support: {1} test: {2}. '.format(self.args.shot,self.args.supportusenum,self.args.testnum)) 
        # print(Over)
        return History
if __name__ == '__main__':
    args = Parse()
    for i in range(1,5):
        args.shot = i*5
        Model = SiaNet(args)
        PreTrainHistory = Model.TrainSiaNet()
        temp = str(round(np.max(PreTrainHistory),3))+'-'+str(round(np.mean(PreTrainHistory[-40:]),3))
        print(temp)
    # Model.PreTrainMGRN()
    # Model.SaveModel('PreTrain',PreTrainHistory)
    # LabelUpdate,LabelUpdateIndex = Model.MyDataCenter.LabelUpdate,Model.MyDataCenter.LabelUpdateIndex
    # np.save('LabelUpdate)10)3',LabelUpdate)








