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
        for _, (supportinputs,supportlabel) in enumerate(SupportTrainDataSet):
            self.supportinputs,self.supportlabel = supportinputs.cuda(),self.MyDataCenter.GetLabel(supportlabel).cuda()
        
        self.MGRN =  mGRNDropout(rnn_hidden_size=6,num_classes=2,device=self.device,input_size_list=[1,1,1,1,1,1,1,1],
                  size_of=11,dropouti=0,dropoutw=0,dropouto=0,keras_initialization=False,batch_first=False)
        # self.MyDataCenter = test.DefineMyDataCenter(1,self.args)
        self.Prototype = np.zeros((self.args.classnum,512))
        self.ProNetOptimizerLearingRate = 0.001
        self.MGRNOptimizerLearingRate = 0.001
        self.ProNetOptimizer = torch.optim.Adam(self.blocks.parameters(),self.ProNetOptimizerLearingRate)
        self.MGRNOptimizer = torch.optim.Adam(self.MGRN.parameters(),self.MGRNOptimizerLearingRate)
        print('随机数：'+self.args.RandomNum[:-4])
        print('shot: {0} support: {1} test: {2}. '.format(self.args.shot,self.args.supportusenum,self.args.testnum)) 
    def GetPro(self):
        supportlogits = self.blocks(self.supportinputs)
        for i in range(self.args.classnum):
            self.Prototype[i] = torch.mean(supportlogits[torch.where(self.supportlabel==i)],0).cpu().detach().numpy()
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
    def PreTrainMGRN(self):
        Begain,Over = ('===PreTrainMGRN开始').ljust(50,'='),('===PreTrainMGRN结束').ljust(50,'=')
        print(Begain)
        TrainDataSet = self.MyDataCenter.GetMGRNTrainData()
        Criterion = torch.nn.MSELoss()
        for epoch in range(self.args.PreTrainMGRNEpoch):
            Num = np.zeros(2)
            for _, (MGRNinputs,MGRNlabel) in enumerate(TrainDataSet):
                MGRNinputs,MGRNlabel = MGRNinputs.to(self.device),MGRNlabel.to(self.device)
                MGRNlogits = self.MGRN(MGRNinputs.permute(1,0,2))
                MGRNLoss = Criterion(MGRNlogits,MGRNlabel)
                self.MGRNOptimizer.zero_grad()
                MGRNLoss.backward()
                self.MGRNOptimizer.step()
                Num[0] += MGRNLoss.cpu().detach().numpy()*len(MGRNlabel)
                Num[1] += len(MGRNlabel)
            Temp = "Epoch:"+str(epoch+1)+"  MGRN-MSE损失:" +str(round(Num[0]/Num[1],5))
            print("\r" + Temp, end="")
        print('')
        print(Over)
    def SSLTest(self,epoch):
        Num = np.zeros(10)
        for _, (queryinputs,querylabel) in enumerate(self.QueryTrainDataSet):
            temp = querylabel
            queryinputs,querylabel = queryinputs.to(self.device),self.MyDataCenter.GetLabel(querylabel).to(self.device)
            queryfeature = self.blocks(queryinputs)
            querylogits = self.GetSoftMax(queryfeature,querylabel.size()[0])
            queryacc,queryloss = GetAccuracy(querylogits,querylabel),GetLoss(querylogits,querylabel,self.device)
            self.MyDataCenter.GetUpdate(temp.detach().numpy(),querylogits.cpu().detach().numpy())
            Num[0] += queryacc
            Num[1] += querylabel.size()[0]
            Num[2] += queryloss.cpu().detach().numpy()
        QueryAcc,QueryLoss = 100*Num[0]/Num[1],Num[2]/Num[1]
        for _, (unlabeledinputs,unlabeledlabel) in enumerate(self.UnlabeledTrainDataSet):
            temp = unlabeledlabel
            Idex,LabelIndex,pseudolabel = self.MyDataCenter.GetPseudoLabel(unlabeledlabel)
            unlabeledinputs,unlabeledlabel = unlabeledinputs.to(self.device),self.MyDataCenter.GetLabel(unlabeledlabel).to(self.device)
            unlabeledfeature = self.blocks(unlabeledinputs)
            unlabeledlogits = self.GetSoftMax(unlabeledfeature,unlabeledlabel.size()[0])
            unlabeledacc,unlabeledloss = GetAccuracy(unlabeledlogits,unlabeledlabel),GetLoss(unlabeledlogits,unlabeledlabel,self.device)
            self.MyDataCenter.GetUpdate(temp.detach().numpy(),unlabeledlogits.cpu().detach().numpy())
            Num[3] += unlabeledacc
            Num[4] += unlabeledlabel.size()[0]
            Num[5] += unlabeledloss.cpu().detach().numpy()
            Num[6] += pseudolabel.size()[0]
            if len(Idex) > 0:
                Num[7] += torch.sum((pseudolabel==self.MyDataCenter.GetLabel(LabelIndex)))
        UnlabeledAcc, UnlabeledLoss = 100*Num[3]/Num[4],Num[5]/Num[4]
        UnlabeledUseRate, UnlabeledUsACC = 100*Num[6]/Num[4],100*Num[7]/Num[6]
        for _, (testinputs,testlabel) in enumerate(self.TestDataSet):
            testinputs,testlabel = testinputs.to(self.device),self.MyDataCenter.GetLabel(testlabel).to(self.device)
            testfeature = self.blocks(testinputs)
            testlogits = self.GetSoftMax(testfeature,testlabel.size()[0])
            testacc = GetAccuracy(testlogits,testlabel)
            Num[8] += testacc
            Num[9] += testlabel.size()[0]
        TestAcc = 100*Num[8]/Num[9]
        Temp = "Epoch:"+str(epoch+1)+\
            "\n 标记样本损失:" +str(round(QueryLoss,3))+" 标记样本精度:"+str(round(QueryAcc,3))+\
            "\n 无标签样本损失:" +str(round(UnlabeledLoss,3))+" 无标签样本精度:"+str(round(UnlabeledAcc,3))+\
            "\n 伪标签使用率:" +str(round(UnlabeledUseRate,3))+" 伪标签样本精度:"+str(round(UnlabeledUsACC,3))+\
            "\n 测试精度:"+ str(round(TestAcc,3)
        # print("\r" + Temp, end="") 
        print(Temp) 
        return Num
    def UpdateWCProForSSLProNetNew(self,weighttemp,unlabeledfeature,pseudolabel,queryfeature,querylabel):
        supportlogits = self.blocks(self.supportinputs)
        for i in range(self.args.classnum):
            if torch.where(pseudolabel==i)!=[]:
                classnum = len(torch.where(self.supportlabel==i))+len(np.where(querylabel==i))+np.sum(weighttemp[torch.where(pseudolabel==i)])
                PrototypeProceser = torch.sum(supportlogits[torch.where(self.supportlabel==i)],0).cpu().detach().numpy() +\
                                    np.sum(queryfeature[np.where(querylabel==i)],0)+\
                                    np.sum(weighttemp[torch.where(pseudolabel==i)]*np.transpose((unlabeledfeature[torch.where(pseudolabel==i)].cpu().detach().numpy()),(1,0)),1)
            else:
                classnum = len(torch.where(self.supportlabel==i))+np.sum(weighttemp[torch.where(pseudolabel==i)])
                PrototypeProceser = torch.sum(supportlogits[torch.where(self.supportlabel==i)],0).cpu().detach().numpy() +\
                                    np.sum(queryfeature[np.where(querylabel==i)],0)
            self.Prototype[i] = PrototypeProceser/classnum
    def UpdateWCProForSSLProNetNew1(self,weighttemp,unlabeledfeature,pseudolabel,queryfeature,querylabel):
        supportlogits = self.blocks(self.supportinputs)
        for i in range(self.args.classnum):
            classnum = len(torch.where(self.supportlabel==i))+np.sum(weighttemp[torch.where(pseudolabel==i)])
            PrototypeProceser = torch.sum(supportlogits[torch.where(self.supportlabel==i)],0).cpu().detach().numpy() +\
                np.sum(queryfeature[np.where(querylabel==i)],0)
            self.Prototype[i] = PrototypeProceser/classnum
    def SSLProNetNew(self):
        Begain,Over = ('===SSLProNetNew开始').ljust(50,'='),('===SSLProNetNew结束').ljust(50,'=')
        print(Begain)
        History = np.zeros((self.args.PLLEpoch,10))
        for epoch in range(self.args.PLLEpoch):
            learingratetemp = self.ProNetOptimizerLearingRate * (np.exp(-3*epoch/self.args.PLLEpoch))
            self.ProNetOptimizer = torch.optim.Adam(self.blocks.parameters(),learingratetemp)
            historyloss,labelindex = [],[]
            for _, (unlabeledinputs,unlabeledlabel) in enumerate(self.UnlabeledTrainDataSet):
                Idex,LabelIndex,pseudolabel = self.MyDataCenter.GetPseudoLabel(unlabeledlabel)
                unlabeledinputs,unlabeledlabel = unlabeledinputs[Idex].to(self.device),self.MyDataCenter.GetLabel(unlabeledlabel[Idex]).to(self.device)
                unlabeledfeature = self.blocks(unlabeledinputs)
                unlabeledlogits = self.GetSoftMax(unlabeledfeature,unlabeledlabel.size()[0])
                for i in range(len(unlabeledlabel)):
                    temp = torch.log10(unlabeledlogits[i,unlabeledlabel[i].cpu().detach().numpy()])
                    historyloss.append(temp.cpu().detach().numpy())
                    labelindex.append(unlabeledlabel)
            historyloss = np.array(historyloss)
            for i 
            labelindex
            lossmax,lossmin = np.max(historyloss),np.min(historyloss)
            Weights = (lossmax - historyloss) / (lossmax-lossmin)
            temp = 0
            for _, (unlabeledinputs,unlabeledlabel) in enumerate(self.UnlabeledTrainDataSet):
                for step, (queryinputs,querylabel) in enumerate(self.QueryTrainDataSet):
                    queryinputs,querylabel = queryinputs.to(self.device),self.MyDataCenter.GetLabel(querylabel).to(self.device)
                    queryfeature = self.blocks(queryinputs)
                    if step == 0:
                        QueryFeature,QueryLabel = queryfeature.cpu().detach().numpy(),querylabel.cpu().detach().numpy()
                    else:
                        QueryFeature,QueryLabel = np.concatenate((QueryFeature,queryfeature.cpu().detach().numpy()),0),\
                                                  np.concatenate((QueryLabel,querylabel.cpu().detach().numpy()),0)
                Idex,LabelIndex,pseudolabel = self.MyDataCenter.GetPseudoLabel(unlabeledlabel)
                weighttemp = Weights[temp:temp+len(Idex)]
                temp += len(Idex)
                if len(Idex)>2:
                    unlabeledinputs,pseudolabel = unlabeledinputs[Idex].to(self.device),pseudolabel.to(self.device)
                    unlabeledfeature = self.blocks(unlabeledinputs)
                    # self.GetPro()
                    self.UpdateWCProForSSLProNetNew1(weighttemp,unlabeledfeature,pseudolabel,QueryFeature,QueryLabel)
                    unlabeledlogits = self.GetSoftMax(unlabeledfeature,pseudolabel.size()[0])
                    loss = GetLoss(unlabeledlogits,pseudolabel,self.device)
                    loss.requires_grad_(True)
                    self.ProNetOptimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    self.ProNetOptimizer.step()
            History[epoch] = self.SSLTest(epoch)
        print(Over)
        return History         
    def UpdateWCProForSSLProNet(self,weighttemp,unlabeledfeature,pseudolabel):
        supportlogits = self.blocks(self.supportinputs)
        for i in range(self.args.classnum):
            classnum = len(torch.where(self.supportlabel==i)) + np.sum(weighttemp)
            PrototypeProceser = torch.sum(supportlogits[torch.where(self.supportlabel==i)],0).cpu().detach().numpy()+\
                                weighttemp*torch.sum(pseudolabel[torch.where(self.pseudolabel==i)],0).cpu().detach().numpy()
            self.Prototype[i] = PrototypeProceser/classnum
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
    def UpdateProForSSLProNetOld(self,unlabeledfeature,unlabeledlabel):
        supportlogits = self.blocks(self.supportinputs)
        for i in range(self.args.classnum):
            classnum = len(torch.where(self.supportlabel==i)) + len(torch.where(unlabeledlabel==i))
            PrototypeProceser = torch.sum(supportlogits[torch.where(self.supportlabel==i)],0).cpu().detach().numpy()+\
                                torch.sum(unlabeledfeature[torch.where(self.unlabeledlabel==i)],0).cpu().detach().numpy()
            self.Prototype[i] = PrototypeProceser/classnum
    def SSLProNetOld(self):
        Begain,Over = ('===SSLProNet开始').ljust(50,'='),('===SSLProNet结束').ljust(50,'=')
        print(Begain)
        History = np.zeros(self.args.PLLEpoch)
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
            for _, (unlabeledinputs,unlabeledlabel) in enumerate(self.UnlabeledTrainDataSet):
                Idex,LabelIndex,pseudolabel = self.MyDataCenter.GetPseudoLabel(unlabeledlabel)
                if len(Idex)>2:
                    unlabeledinputs,pseudolabel = unlabeledinputs[Idex].to(self.device),pseudolabel.to(self.device)
                    unlabeledfeature = self.blocks(unlabeledinputs)
                    self.UpdateWCProForSSLProNet(weighttemp,unlabeledfeature,pseudolabel)
                    unlabeledlogits = self.GetSoftMax(unlabeledfeature,pseudolabel.size()[0])
                    loss = GetLoss(unlabeledlogits,pseudolabel,self.device)
                    self.ProNetOptimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    self.ProNetOptimizer.step()
            History[epoch] = self.SSLTest()
        print(Over)
        return History
    def SSLProNetOld1(self):
        Begain,Over = ('===SSLProNet开始').ljust(50,'='),('===SSLProNet结束').ljust(50,'=')
        print(Begain)
        self.GetPro()
        _ = self.SSLTest(-1)
        History = np.zeros((self.args.PLLEpoch,10))
        for epoch in range(self.args.PLLEpoch):
            # learingratetemp = self.ProNetOptimizerLearingRate * (np.exp(-3*epoch/(self.args.PLLEpoch)))
            learingratetemp = self.ProNetOptimizerLearingRate * (np.exp(-3))
            self.ProNetOptimizer = torch.optim.Adam(self.blocks.parameters(),learingratetemp)
            for step, (queryinputs,querylabel) in enumerate(self.QueryTrainDataSet):
                queryinputs,querylabel = queryinputs.to(self.device),self.MyDataCenter.GetLabel(querylabel).to(self.device)
                queryfeature = self.blocks(queryinputs)
                querylogits = self.GetSoftMax(queryfeature,querylabel.size()[0])
                if step==0:
                    loss = 8*GetLoss(querylogits,querylabel,self.device)
                else:
                    loss += 8*GetLoss(querylogits,querylabel,self.device)
            self.ProNetOptimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.ProNetOptimizer.step()
            self.GetPro()
            for step, (unlabeledinputs,unlabeledlabel) in enumerate(self.UnlabeledTrainDataSet):
                Idex,LabelIndex,pseudolabel = self.MyDataCenter.GetPseudoLabel(unlabeledlabel)
                if len(Idex)>2:
                    unlabeledinputs,pseudolabel = unlabeledinputs[Idex].to(self.device),pseudolabel.to(self.device)
                    # unlabeledinputs,pseudolabel = unlabeledinputs[Idex].to(self.device),self.MyDataCenter.GetLabel(LabelIndex).to(self.device)
                    unlabeledfeature = self.blocks(unlabeledinputs)
                    unlabeledlogits = self.GetSoftMax(unlabeledfeature,pseudolabel.size()[0])
                    loss = GetLoss(unlabeledlogits,pseudolabel,self.device)
                    loss.requires_grad_(True)
                    self.ProNetOptimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    self.ProNetOptimizer.step()
                    self.GetPro()
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
    #%%半监督
    if os.path.exists(PLLModelName) and os.path.exists(PLLMyDataCenterName):
        Model.LoadPLLMyDataCenter(PLLMyDataCenterName)
        Model.LoadPLLModel(PLLModelName)
        Model.LoadPLLMyDataCenter(PLLMyDataCenterName)
        PLLTrainHistory = np.load(PLLTrainHistoryName)
    else:
        PLLTrainHistory = Model.SSLProNetOld1()
        Model.SaveModel(PLLModelName)
        Model.SaveMyDataCenter(PLLMyDataCenterName)
        np.save(PLLTrainHistoryName,PLLTrainHistory)
    print(100*np.mean((PreTrainHistory[:,3]/PreTrainHistory[:,4])[-20:]))
    print(100*np.mean((PLLTrainHistory[:,8]/PLLTrainHistory[:,9])[-20:]))
    a = (PLLTrainHistory[:,8]/PLLTrainHistory[:,9])
    b = (PreTrainHistory[:,3]/PreTrainHistory[:,4])
    # plt.plot((PLLTrainHistory[:,8]/PLLTrainHistory[:,9]))
    # plt.plot((PreTrainHistory[:,3]/PreTrainHistory[:,4]))
    # np.delete(PreTrainHistory,2,1)
    # del PreTrainHistoryDrowName[2]








