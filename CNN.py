import os,torch,warnings,sys
import numpy as np
from Schrobine import *
import matplotlib.pyplot as plt
global args
warnings.filterwarnings('ignore')
SetSeed()
class CNN(torch.nn.Module):
    def __init__(self,args,**kwargs):
        super(CNN, self).__init__( **kwargs)
        self.args = args
        self.TrainDataSet,self.UnlabeledTrainDataSet,self.TestDataSet,self.MyDataCenter = CNNGetDataSet(self.args)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.Convnet = torch.nn.Sequential(GetSomeLayer('Blocks1DTorch',1,512),  GetSomeLayer('Blocks1DTorch',512,256),
                                          GetSomeLayer('Blocks1DTorch',256,256),GetSomeLayer('Blocks1DTorch',256,128),
                                          GetSomeLayer('Blocks1DTorch',128,128),GetSomeLayer('Blocks1DTorch',128,32),
                                          GetSomeLayer('ClasserTorch',512,8))
        self.Convnet.to(self.device)
        self.CNNOptimizerLearingRate = 0.001
        self.MGRNOptimizerLearingRate = 0.001
        self.CNNNetOptimizer = torch.optim.Adam(self.Convnet.parameters(),self.CNNOptimizerLearingRate)
    def SaveModel(self,Name,History):
        temp = str(round(np.mean(np.sort(History)[-int(0.1*len(History)):]),2))+')'+str(self.args.epoch)+')'+str(self.args.batch)+')'+\
            str(self.args.shot)+')'+str(self.args.supportusenum)+')'+str(self.args.testnum)+')'
        torch.save(self.state_dict(),'SaveModel/'+Name+')'+temp+GetTime()+'.pkl')
    def TrainCNN(self):
        Begain,Over = ('===PTrainCNN开始').ljust(50,'='),('===PTrainCNN结束').ljust(50,'=')
        print(Begain)
        History = np.zeros(self.args.PreTrainProNetEpoch)
        ST = (self.args.PreTrainProNetEpoch+self.args.PLLEpoch)
        for epoch in range(self.args.PreTrainProNetEpoch):
            learingratetemp = self.CNNOptimizerLearingRate*(np.exp(-3*epoch/ST))
            self.CNNNetOptimizer = torch.optim.Adam(self.Convnet.parameters(),learingratetemp)
            for _, (traininputs,trainlabel) in enumerate(self.TrainDataSet):
                traininputs,trainlabel = traininputs.to(self.device),self.MyDataCenter.GetLabel(trainlabel).to(self.device)
                trainlogits = self.Convnet(traininputs)
                loss = GetLoss(trainlogits,trainlabel,self.device)
                self.CNNNetOptimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.CNNNetOptimizer.step()
            Num = np.zeros(3)
            for _, (traininputs,trainlabel) in enumerate(self.TrainDataSet):
                temp = trainlabel
                traininputs,trainlabel = traininputs.to(self.device),self.MyDataCenter.GetLabel(trainlabel).to(self.device)
                trainlogits = self.Convnet(traininputs)
                trainacc,trainloss = GetAccuracy(trainlogits,trainlabel),GetLoss(trainlogits,trainlabel,self.device)
                self.MyDataCenter.GetUpdate(temp.detach().numpy(),trainlogits.cpu().detach().numpy())
                Num[0] += trainacc
                Num[1] += trainlabel.size()[0]
                Num[2] += trainloss.cpu().detach().numpy()*Num[1]
            Trainacc, Trainloss = 100*Num[0]/Num[1],Num[2]/Num[1]
            Num = np.zeros(2)
            for _, (testinputs,testlabel) in enumerate(self.TestDataSet):
                testinputs,testlabel = testinputs.to(self.device),self.MyDataCenter.GetLabel(testlabel).to(self.device)
                testlogits = self.Convnet(testinputs)
                testacc = GetAccuracy(testlogits,testlabel)
                Num[0] += testacc
                Num[1] += testlabel.size()[0]
            TestAcc = 100*Num[0]/Num[1]
            Temp = "Epoch:"+str(epoch+1)+" 训练损失:" +str(round(Trainloss,2))+" 训练精度:"+str(round(Trainacc,2))+" 测试精度:"+ str(round(TestAcc,2))
            print("\r" + Temp, end="") 
            # print(Temp) 
            for _, (unlabeledinputs,unlabeledlabel) in enumerate(self.UnlabeledTrainDataSet):
                temp = unlabeledlabel
                unlabeledinputs = unlabeledinputs.to(self.device)
                unlabeledlogits = self.Convnet(unlabeledinputs)
                self.MyDataCenter.GetUpdate(temp.detach().numpy(),unlabeledlogits.cpu().detach().numpy())
            History[epoch] = TestAcc
        print('')
        print(Over)
        return History
    def PLL(self):
        Begain,Over = ('===PLL测试开始').ljust(50,'='),('===PLL测试结束').ljust(50,'=')
        print(Begain)
        Num = np.zeros(3)
        for _, (unlabeledinputs,unlabeledlabel) in enumerate(self.UnlabeledTrainDataSet):
            Idex,LabelIndex,pseudolabel = self.MyDataCenter.GetPseudoLabel(unlabeledlabel)
            Num[0] += torch.sum((pseudolabel==self.MyDataCenter.GetLabel(LabelIndex)))
            Num[1] += pseudolabel.size()[0] 
            Num[2] += unlabeledlabel.size()[0]
        print(Num[0]/Num[1])
        print(Num[1]/Num[2])
        print(Over)
    def PLLCNN(self):
        Begain,Over = ('===PLLCNN开始').ljust(50,'='),('===PLLCNN结束').ljust(50,'=')
        print(Begain)
        History = np.zeros(self.args.PLLEpoch)
        ST = (self.args.PreTrainProNetEpoch+self.args.PLLEpoch)
        for epoch in range(self.args.PLLEpoch):
            learingratetemp = self.CNNOptimizerLearingRate*(np.exp(-3*(epoch+self.args.PreTrainProNetEpoch)/ST))
            self.CNNNetOptimizer = torch.optim.Adam(self.Convnet.parameters(),learingratetemp)
            for _, (traininputs,trainlabel) in enumerate(self.TrainDataSet):
                traininputs,trainlabel = traininputs.to(self.device),self.MyDataCenter.GetLabel(trainlabel).to(self.device)
                trainlogits = self.Convnet(traininputs)
                loss = GetLoss(trainlogits,trainlabel,self.device)
                self.CNNNetOptimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.CNNNetOptimizer.step()
            for _, (unlabeledinputs,unlabeledlabel) in enumerate(self.UnlabeledTrainDataSet):
                Idex,LabelIndex,pseudolabel = self.MyDataCenter.GetPseudoLabel(unlabeledlabel)
                if len(Idex)>1:
                    unlabeledinputs,pseudolabel = unlabeledinputs[Idex].to(self.device),pseudolabel.to(self.device)
                    unlabeledlogits = self.Convnet(unlabeledinputs)
                    loss = GetLoss(unlabeledlogits,pseudolabel,self.device)
                    self.CNNNetOptimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    self.CNNNetOptimizer.step()
            Num = np.zeros(3)
            for _, (traininputs,trainlabel) in enumerate(self.TrainDataSet):
                temp = trainlabel
                traininputs,trainlabel = traininputs.to(self.device),self.MyDataCenter.GetLabel(trainlabel).to(self.device)
                trainlogits = self.Convnet(traininputs)
                trainacc,trainloss = GetAccuracy(trainlogits,trainlabel),GetLoss(trainlogits,trainlabel,self.device)
                self.MyDataCenter.GetUpdate(temp.detach().numpy(),trainlogits.cpu().detach().numpy())
                Num[0] += trainacc
                Num[1] += trainlabel.size()[0]
                Num[2] += trainloss.cpu().detach().numpy()
            TrainAcc, TrainLoss = 100*Num[0]/Num[1],Num[2]/Num[1]
            Num = np.zeros(5)
            for _, (unlabeledinputs,unlabeledlabel) in enumerate(self.UnlabeledTrainDataSet):
                temp = unlabeledlabel
                Idex,LabelIndex,pseudolabel = self.MyDataCenter.GetPseudoLabel(unlabeledlabel)
                unlabeledinputs,unlabeledlabel = unlabeledinputs.to(self.device),self.MyDataCenter.GetLabel(unlabeledlabel).to(self.device)
                unlabeledlogits = self.Convnet(unlabeledinputs)
                unlabeledacc,unlabeledloss = GetAccuracy(unlabeledlogits,unlabeledlabel),GetLoss(unlabeledlogits,unlabeledlabel,self.device)
                self.MyDataCenter.GetUpdate(temp.detach().numpy(),unlabeledlogits.cpu().detach().numpy())
                Num[0] += unlabeledacc
                Num[1] += unlabeledlabel.size()[0]
                Num[2] += unlabeledloss.cpu().detach().numpy()
                Num[3] += pseudolabel.size()[0]
                if len(Idex) > 0:
                    Num[4] += torch.sum((pseudolabel==self.MyDataCenter.GetLabel(LabelIndex)))
            UnlabeledAcc, UnlabeledLoss = 100*Num[0]/Num[1],Num[2]/Num[1]
            UnlabeledUseRate, UnlabeledUsACC = 100*Num[3]/Num[1],100*Num[4]/Num[3]
            Num = np.zeros(2)
            for _, (testinputs,testlabel) in enumerate(self.TestDataSet):
                testinputs,testlabel = testinputs.to(self.device),self.MyDataCenter.GetLabel(testlabel).to(self.device)
                testlogits = self.Convnet(testinputs)
                testacc = GetAccuracy(testlogits,testlabel)
                Num[0] += testacc
                Num[1] += testlabel.size()[0]
            TestAcc = 100*Num[0]/Num[1]
            Temp = "Epoch:"+str(epoch+1)+\
                "\n 标记样本损失:" +str(round(TrainLoss,2))+" 标记样本精度:"+str(round(TrainAcc,2))+\
                "\n 无标签样本损失:" +str(round(UnlabeledLoss,2))+" 无标签样本精度:"+str(round(UnlabeledAcc,2))+\
                "\n 伪标签使用率:" +str(round(UnlabeledUseRate,2))+" 伪标签样本精度:"+str(round(UnlabeledUsACC,2))+\
                "\n 测试精度:"+ str(round(TestAcc,2))
            # print("\r" + Temp, end="") 
            print(Temp) 
            History[epoch] = TestAcc
        print(Over)
        return History
if __name__ == '__main__':
    args = Parse()
    Model = CNN(args)
    PreTrainHistory = Model.TrainCNN() 
    # Model.PLL()
    PLLCNNHistory = Model.PLLCNN()
    print(np.max(PLLCNNHistory),np.mean(PLLCNNHistory[-40:]))
#     args = Parse()
#     args.shot = 0
#     for i in range(4):
#         args.shot += 5
#         Model = CNN(args)
#         Model.cuda()
#         PreTrainHistory = Model.TrainCNN()
#         # print(np.max(PreTrainHistory),np.mean(PreTrainHistory[-40:]))
#     # Model.PreTrainMGRN()
#     # Model.SaveModel('PreTrain',PreTrainHistory)
#     # LabelUpdate,LabelUpdateIndex = Model.MyDataCenter.LabelUpdate,Model.MyDataCenter.LabelUpdateIndex
#     # np.save('LabelUpdate)10)3',LabelUpdate)
