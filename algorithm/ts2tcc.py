from torch.utils.data import TensorDataset, DataLoader
from models.ts2tcc.model import *
from models.ts2tcc.TC import *
from models.ts2tcc.loss import *
from utils import *
from tqdm import tqdm
import torch.nn as nn
from dataloader import Load_Dataset


class TS_TCC(nn.Module):
    '''The Proposed TS_CoT model'''

    def __init__(
            self,
            device='cuda',
            lr=0.001,
            args=None
    ):

        super().__init__()
        self.device = device
        self.lr = lr
        self.args = args
        self.model = base_Model(self.args).to(device)
        self.temporal_contr_model = TC(self.args, device).to(device)
        
    
    def fit_ts_cot(self, train_loader,n_epochs=None,logger=None):
        total_loss = []

        self.model.train()
        self.temporal_contr_model.train()

        model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, betas=(self.args.beta1, self.args.beta2),
                                    weight_decay=3e-4)

        temporal_contr_optimizer = torch.optim.Adam(self.temporal_contr_model.parameters(), lr=self.args.lr,
                                                betas=(self.args.beta1, self.args.beta2), weight_decay=3e-4)
        train_loader = Load_Dataset(train_loader, self.args)
        train_loader = DataLoader(dataset=train_loader, batch_size=self.args.batch_size,
                                           shuffle=True, drop_last=True, num_workers=0)
        device = self.device
        for epoch in range(1, self.args.epochs + 1):
            for batch_idx, (data, aug1, aug2) in enumerate(train_loader):
                # send to device
                data = data.float().to(device),
                aug1, aug2 = aug1.float().to(device), aug2.float().to(device)

                # optimizer
                model_optimizer.zero_grad()
                temporal_contr_optimizer.zero_grad()
                features1 = self.model(aug1)
                features2 = self.model(aug2)
                # normalize projection feature vectors
                features1 = F.normalize(features1, dim=1)
                features2 = F.normalize(features2, dim=1)

                temp_cont_loss1, temp_cont_feat1 = self.temporal_contr_model(features1, features2)
                temp_cont_loss2, temp_cont_feat2 = self.temporal_contr_model(features2, features1)


                lambda1 = 1
                lambda2 = 0.7

                nt_xent_criterion = NTXentLoss(device, self.args.batch_size, self.args.temperature,
                                                self.args.cc_use_cosine_similarity)
                loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + \
                        nt_xent_criterion(temp_cont_feat1, temp_cont_feat2) * lambda2


                total_loss.append(loss.item())
                loss.backward()
                model_optimizer.step()
                temporal_contr_optimizer.step()
            logger.info(f"Epoch : {epoch} | Loss : {loss.item()}")
        total_loss = torch.tensor(total_loss).mean()

        return total_loss

    def encode(self, data, batch_size=None):
        '''
        离线用于训练分类器
        '''
        self.model.eval()
        self.temporal_contr_model.eval()

        # train_dataset1 = TwoViewloader(data)
        # loader = DataLoader(train_dataset1, batch_size=min(self.batch_size, len(train_dataset1)), shuffle=False,
        #                           drop_last=False)
        
        train_loader = Load_Dataset(data, self.args)
        train_loader = DataLoader(dataset=train_loader, batch_size=self.args.batch_size,
                                           shuffle=True, drop_last=False, num_workers=0)

        with torch.no_grad():
            output = []
            for batch_idx, (data, aug1, aug2) in enumerate(train_loader):
                # send to device
                #print(aug1.shape,aug2.shape,data.shape)
                data = data.float().to(self.device)
                # aug1, aug2 = aug1.float().to(self.device), aug2.float().to(self.device)
                feat = self.model(data)
                print(feat.shape)
                feat = feat.reshape(feat.shape[0], -1)
                output.append(feat)
            
            output = torch.cat(output, dim=0)
            print(output.shape)

        return output.cpu().numpy()
    
    def encode_online(self,data):
        '''
        在线推理
        '''

        self.model.eval()
        self.temporal_contr_model.eval()

        with torch.no_grad():
            data = torch.from_numpy(data).to(torch.float).to(self.device)
            feat = self.model(data)
            return feat.cpu().numpy()


    def save(self, fn):
        torch.save({'model': self.model.state_dict(), 'temporal_contr_model': self.temporal_contr_model.state_dict()}, fn)
    
    def load(self,fn):
        # state_dict = torch.load(fn, map_location=self.device)
        self.model.load_state_dict(torch.load(fn,map_location=self.device)['model'])
        self.temporal_contr_model.load_state_dict(torch.load(fn, map_location=self.device)['temporal_contr_model'])
