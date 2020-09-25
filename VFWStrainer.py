import torch

class VFWStrainer():
    def __init__(self,model,trainloader,testloader,criterion,optimizer,num_epoch,device=0):
        self.device = torch.device('cuda:{}'.format(device))
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.crterion = criterion
        self.optimizer = optimizer
        self.num_epoch = num_epoch
        self.epochs = 0
        self.model= self.model.to(self.device)

    def step(self):
        return

    def train(self):
        return

    def inference(self):
        return



    def load(self,path):
        self.model.load_state_dict(torch.load(path))
