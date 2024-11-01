from torch.utils.data import TensorDataset, DataLoader
from models import *
from utils import *
from tqdm import tqdm
import torch.nn as nn
from models.info_nce import InfoNCE

def generate_binomial_mask(target_matrix, p=0.5, seed_=42):
    # np.random.seed(seed_)
    return torch.from_numpy(np.random.binomial(1, p, size=(target_matrix.shape)))

class TS_SEA(nn.Module):
    '''The Proposed TS_SEA model'''

    def __init__(
            self,
            input_dims,
            output_dims=32,
            hidden_dims=64,
            device='cuda',
            lr=0.001,
            args=None
    ):

        super().__init__()
        self.device = device
        self.lr = lr
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims

        if args.dataloader in  ['ECG','SleepEEG','SleepEDF','EMG']:
            self.tem_encoder = Conv_Pyram_model_EDF(input_dims=input_dims, output_dims=output_dims).to(self.device)
            self.fre_encoder = Conv_Pyram_model_EDF(input_dims=input_dims, output_dims=output_dims).to(self.device)
            self.sea_encoder = Conv_Pyram_model_EDF(input_dims=input_dims, output_dims=output_dims).to(self.device)

        elif args.dataloader  in ['FD-A','FD-B','Gesture']:
            self.tem_encoder = Conv_Pyram_model_EDF(input_dims=input_dims, output_dims=output_dims).to(self.device)
            self.fre_encoder = Conv_Pyram_model_EDF(input_dims=input_dims, output_dims=output_dims).to(self.device)
            self.sea_encoder = Conv_Pyram_model_EDF(input_dims=input_dims, output_dims=output_dims).to(self.device)

        elif args.dataloader  in ['HAR','UEA','UCR','general_cls']:
            self.tem_encoder = Conv_Pyram_model_HAR(input_dims=input_dims, output_dims=output_dims).to(self.device)
            self.fre_encoder = Conv_Pyram_model_HAR(input_dims=input_dims, output_dims=output_dims).to(self.device)
            self.sea_encoder = Conv_Pyram_model_HAR(input_dims=input_dims, output_dims=output_dims).to(self.device)
        elif args.dataloader == 'Epilepsy' or args.dataloader == 'Epi':
            self.tem_encoder = Conv_Pyram_model_Epi(input_dims=input_dims, output_dims=output_dims).to(self.device)
            self.fre_encoder = Conv_Pyram_model_Epi(input_dims=input_dims, output_dims=output_dims).to(self.device)
            self.sea_encoder = Conv_Pyram_model_Epi(input_dims=input_dims, output_dims=output_dims).to(self.device)
        elif args.dataloader == 'Waveform':
            self.tem_encoder = Conv_Pyram_model_Epi(input_dims=input_dims, output_dims=output_dims).to(self.device)
            self.fre_encoder = Conv_Pyram_model_Epi(input_dims=input_dims, output_dims=output_dims).to(self.device)
            self.sea_encoder = Conv_Pyram_model_Epi(input_dims=input_dims, output_dims=output_dims).to(self.device)
            self.sea_encoder = Conv_Pyram_model_Epi(input_dims=input_dims, output_dims=output_dims).to(self.device)
        elif args.dataloader == 'ISRUC':
            self.tem_encoder = Conv_Pyram_model_ISRUC(input_dims=input_dims, output_dims=output_dims).to(self.device)
            self.fre_encoder = Conv_Pyram_model_ISRUC(input_dims=input_dims, output_dims=output_dims).to(self.device)
            self.sea_encoder = Conv_Pyram_model_ISRUC(input_dims=input_dims, output_dims=output_dims).to(self.device)
        elif args.dataloader == 'RoadBank' or args.dataloader == "Bridge":
            self.tem_encoder = Conv_Pyram_model_ISRUC(input_dims=input_dims, output_dims=output_dims).to(self.device)
            self.fre_encoder = Conv_Pyram_model_ISRUC(input_dims=input_dims, output_dims=output_dims).to(self.device)
            self.sea_encoder = Conv_Pyram_model_ISRUC(input_dims=input_dims, output_dims=output_dims).to(self.device)

        else:
            print('Unknown Dataset')

        self.n_epochs = 0
        self.args = args
        self.cluster_result = None
        self.batch_size = args.batch_size

    def clustring_calc(self,train_data,train_loader,last_clusters):
        
        if self.n_epochs == max(int(self.args.warmup), 0):
            
            features = self.encode(train_data )

            feature_split = int(features.shape[1]/3)
            if np.any(np.isnan(features)).item():
                return
            features_tem = features[:, 0:feature_split]
            features_fre = features[:, feature_split:2*feature_split]
            features_sea = features[:, 2*feature_split:]

            cluster_result_tem =  {'im2cluster': [], 'centroids': [], 'density': [], 'ma_centroids':[]}
            cluster_result_fre = {'im2cluster': [], 'centroids': [], 'density': [],  'ma_centroids':[]}
            cluster_result_sea = {'im2cluster': [], 'centroids': [], 'density': [],  'ma_centroids':[]}

            for num_cluster in self.args.num_cluster:
                cluster_result_tem['im2cluster'].append(torch.zeros(len(train_loader), dtype=torch.long).cuda())
                cluster_result_fre['im2cluster'].append(torch.zeros(len(train_loader), dtype=torch.long).cuda())
                cluster_result_sea['im2cluster'].append(torch.zeros(len(train_loader), dtype=torch.long).cuda())
                cluster_result_tem['centroids'].append(
                    torch.zeros(int(num_cluster), train_data[0].shape[1] * self.args.repr_dims).cuda())
                cluster_result_fre['centroids'].append(
                    torch.zeros(int(num_cluster), train_data[0].shape[1] * self.args.repr_dims).cuda())
                cluster_result_sea['centroids'].append(
                    torch.zeros(int(num_cluster), train_data[0].shape[1] * self.args.repr_dims).cuda())
                cluster_result_tem['density'].append(torch.zeros(int(num_cluster)).cuda())
                cluster_result_fre['density'].append(torch.zeros(int(num_cluster)).cuda())
                cluster_result_sea['density'].append(torch.zeros(int(num_cluster)).cuda())
                cluster_result_tem = run_kmeans(features_tem, self.args, last_clusters)
                cluster_result_fre = run_kmeans(features_fre, self.args, last_clusters)
                cluster_result_sea = run_kmeans(features_sea, self.args, last_clusters)

            for tmp in range(len(self.args.num_cluster)):
                tem_im2cluster = cluster_result_tem['im2cluster'][tmp]
                fre_im2cluster = cluster_result_fre['im2cluster'][tmp]
                sea_im2cluster = cluster_result_sea['im2cluster'][tmp]
                dist_tem = cluster_result_tem['distance_2_center'][tmp]
                dist_fre = cluster_result_fre['distance_2_center'][tmp]
                dist_sea = cluster_result_sea['distance_2_center'][tmp]
                print("=="*50)
                print(tem_im2cluster)
                print(fre_im2cluster)
                print(sea_im2cluster)
                tmp_cluster = self.args.num_cluster[tmp]
                for tmpp in range(int(tmp_cluster)):
                    sort_tem_index = np.array(np.argsort(dist_tem[:,tmpp])) #
                    sort_fre_index = np.array(np.argsort(dist_fre[:,tmpp])) #
                    sort_sea_index = np.array(np.argsort(dist_sea[:,tmpp])) #
                    sort_tem_index = sort_tem_index[:int(0.8*len(np.array(torch.where(tem_im2cluster == tmpp)[0].cpu())))]
                    sort_fre_index = sort_fre_index[:int(0.8*len(np.array(torch.where(fre_im2cluster == tmpp)[0].cpu())))]
                    sort_sea_index = sort_sea_index[:int(0.8*len(np.array(torch.where(sea_im2cluster == tmpp)[0].cpu())))]

                    set_tem = np.intersect1d(np.array(torch.where(tem_im2cluster == tmpp)[0].cpu()), sort_tem_index)
                    set_fre = np.intersect1d(np.array(torch.where(fre_im2cluster == tmpp)[0].cpu()), sort_fre_index)
                    set_sea = np.intersect1d(np.array(torch.where(sea_im2cluster == tmpp)[0].cpu()), sort_sea_index)
                    
                    neg_tem = np.setdiff1d(np.array(torch.where(tem_im2cluster == tmpp)[0].cpu()), sort_tem_index)
                    neg_fre = np.setdiff1d(np.array(torch.where(fre_im2cluster == tmpp)[0].cpu()), sort_fre_index)
                    neg_sea = np.setdiff1d(np.array(torch.where(sea_im2cluster == tmpp)[0].cpu()), sort_sea_index)
                    if len(neg_tem) > 0 and len(set_tem) > 0:
                        tem_im2cluster[neg_tem] = -1
                    if len(neg_fre) > 0 and len(set_fre) > 0:
                        fre_im2cluster[neg_fre] = -1
                    if len(neg_sea) > 0 and len(set_fre) > 0:
                        sea_im2cluster[neg_sea] = -1
                    
                    fre_tem_mean = torch.mean(torch.tensor(features_fre[np.array(torch.where(tem_im2cluster == tmpp)[0].cpu()), :]),0).cuda()
                    fre_sea_mean = torch.mean(torch.tensor(features_fre[np.array(torch.where(sea_im2cluster == tmpp)[0].cpu()), :]),0).cuda()
                    cluster_result_fre['centroids'][tmp][tmpp, :] =  torch.mean(fre_tem_mean/ torch.norm(fre_tem_mean) + fre_sea_mean/ torch.norm(fre_sea_mean))

                    tem_fre_mean = torch.mean(torch.tensor(features_tem[np.array(torch.where(fre_im2cluster == tmpp)[0].cpu()), :]), 0).cuda()
                    tem_sea_mean = torch.mean(torch.tensor(features_tem[np.array(torch.where(sea_im2cluster == tmpp)[0].cpu()), :]), 0).cuda()
                    cluster_result_tem['centroids'][tmp][tmpp, :] = torch.mean(tem_fre_mean/ torch.norm(tem_fre_mean) + tem_sea_mean/ torch.norm(tem_sea_mean))

                    sea_tmp_mean = torch.mean(torch.tensor(features_sea[np.array(torch.where(tem_im2cluster == tmpp)[0].cpu()), :]), 0).cuda()
                    sea_fre_mean = torch.mean(torch.tensor(features_sea[np.array(torch.where(fre_im2cluster == tmpp)[0].cpu()), :]), 0).cuda()
                    cluster_result_sea['centroids'][tmp][tmpp, :] = torch.mean(sea_tmp_mean/ torch.norm(sea_tmp_mean) + tem_sea_mean/ torch.norm(sea_fre_mean))

                    # cluster_result_sea['centroids'][tmp][tmpp, :] = torch.mean(
                    #     torch.tensor(features_sea[np.array(torch.where(sea_im2cluster == tmpp)[0].cpu()), :]),
                    #     0).cuda() / torch.norm(torch.mean(
                    #     torch.tensor(features_sea[np.array(torch.where(sea_im2cluster == tmpp)[0].cpu()), :]),
                    #     0).cuda())
            cluster_result_tem['ma_centroids'] = cluster_result_tem['centroids']
            cluster_result_fre['ma_centroids'] = cluster_result_fre['centroids']
            cluster_result_sea['ma_centroids'] = cluster_result_sea['centroids']

        if self.n_epochs > max(int(self.args.warmup), 0):
            features = self.encode(train_data)
            feature_split = int(features.shape[1]/3)
            if np.any(np.isnan(features)).item():
                return 
            features_tem = features[:, :feature_split]
            features_fre = features[:, feature_split:2*feature_split]
            features_sea = features[:, 2*feature_split:]
            for jj in range(len(self.args.num_cluster)):
                ma_centroids_tem = cluster_result_tem['ma_centroids'][jj]/torch.norm(cluster_result_tem['ma_centroids'][jj], dim=1, keepdim=True)
                cp_tem = torch.matmul(torch.tensor(features_tem).cuda(), ma_centroids_tem.transpose(1, 0))
                cluster_result_tem['im2cluster'][jj] = torch.argmax(cp_tem, dim=1)
                cluster_result_tem['distance_2_center'][jj] = 1-cp_tem.cpu().numpy()

                ma_centroids_fre = cluster_result_fre['ma_centroids'][jj]/torch.norm(cluster_result_fre['ma_centroids'][jj], dim=1, keepdim=True)
                cp_fre = torch.matmul(torch.tensor(features_fre).cuda(), ma_centroids_fre.transpose(1, 0))
                cluster_result_fre['im2cluster'][jj] = torch.argmax(cp_fre, dim=1)
                cluster_result_fre['distance_2_center'][jj] = 1-cp_fre.cpu().numpy()
                
                ma_centroids_sea = cluster_result_sea['ma_centroids'][jj]/torch.norm(cluster_result_sea['ma_centroids'][jj], dim=1, keepdim=True)
                cp_sea = torch.matmul(torch.tensor(features_sea).cuda(), ma_centroids_sea.transpose(1, 0))
                cluster_result_sea['im2cluster'][jj] = torch.argmax(cp_sea, dim=1)
                cluster_result_sea['distance_2_center'][jj] = 1-cp_sea.cpu().numpy()

                cluster_result_tem['density'][jj] = torch.ones(cluster_result_tem['density'][jj].shape).cuda()
                cluster_result_fre['density'][jj] = torch.ones(cluster_result_fre['density'][jj].shape).cuda()
                cluster_result_sea['density'][jj] = torch.ones(cluster_result_sea['density'][jj].shape).cuda()

            cluster_result_tem = run_kmeans(features_tem, self.args, last_clusters)
            cluster_result_fre = run_kmeans(features_fre, self.args, last_clusters)
            cluster_result_sea = run_kmeans(features_sea, self.args, last_clusters)

            for tmp in range(len(self.args.num_cluster)):
                tem_im2cluster = cluster_result_tem['im2cluster'][tmp]
                fre_im2cluster = cluster_result_fre['im2cluster'][tmp]
                sea_im2cluster = cluster_result_sea['im2cluster'][tmp]

                dist_tem = cluster_result_tem['distance_2_center'][tmp]
                dist_fre = cluster_result_fre['distance_2_center'][tmp]
                dist_sea = cluster_result_sea['distance_2_center'][tmp]

                tmp_cluster = self.args.num_cluster[tmp]
                for tmpp in range(int(tmp_cluster)):
                    keep_ratio = 1
                    sort_tem_index = np.array(np.argsort(dist_tem[:,tmpp])) # 前面的距离小
                    sort_fre_index = np.array(np.argsort(dist_fre[:,tmpp])) #
                    sort_sea_index = np.array(np.argsort(dist_sea[:,tmpp])) #

                    sort_tem_index = sort_tem_index[:int(keep_ratio*len(np.array(torch.where(tem_im2cluster == tmpp)[0].cpu())))]
                    sort_fre_index = sort_fre_index[:int(keep_ratio*len(np.array(torch.where(fre_im2cluster == tmpp)[0].cpu())))]
                    sort_sea_index = sort_sea_index[:int(keep_ratio*len(np.array(torch.where(sea_im2cluster == tmpp)[0].cpu())))]

                    set_tem = np.intersect1d(np.array(torch.where(tem_im2cluster == tmpp)[0].cpu()), sort_tem_index)
                    set_fre = np.intersect1d(np.array(torch.where(fre_im2cluster == tmpp)[0].cpu()), sort_fre_index)
                    set_sea = np.intersect1d(np.array(torch.where(sea_im2cluster == tmpp)[0].cpu()), sort_sea_index)

                    neg_tem = np.setdiff1d(np.array(torch.where(tem_im2cluster == tmpp)[0].cpu()), sort_tem_index)
                    neg_fre = np.setdiff1d(np.array(torch.where(fre_im2cluster == tmpp)[0].cpu()), sort_fre_index)
                    neg_sea = np.setdiff1d(np.array(torch.where(sea_im2cluster == tmpp)[0].cpu()), sort_sea_index)

                    if len(neg_tem) > 0 and len(set_tem) > 0:
                        tem_im2cluster[neg_tem] = -1
                    if len(neg_fre) > 0 and len(set_fre) > 0:
                        fre_im2cluster[neg_fre] = -1
                    if len(neg_sea) > 0 and len(set_sea) > 0:
                        sea_im2cluster[neg_fre] = -1

                    fre_tem_mean = torch.mean(torch.tensor(features_fre[np.array(torch.where(tem_im2cluster == tmpp)[0].cpu()), :]),0).cuda()
                    fre_sea_mean = torch.mean(torch.tensor(features_fre[np.array(torch.where(sea_im2cluster == tmpp)[0].cpu()), :]),0).cuda()
                    cluster_result_fre['centroids'][tmp][tmpp, :] =  torch.mean(fre_tem_mean/ torch.norm(fre_tem_mean) + fre_sea_mean/ torch.norm(fre_sea_mean))

                    tem_fre_mean = torch.mean(torch.tensor(features_tem[np.array(torch.where(fre_im2cluster == tmpp)[0].cpu()), :]), 0).cuda()
                    tem_sea_mean = torch.mean(torch.tensor(features_tem[np.array(torch.where(sea_im2cluster == tmpp)[0].cpu()), :]), 0).cuda()
                    cluster_result_tem['centroids'][tmp][tmpp, :] = torch.mean(tem_fre_mean/ torch.norm(tem_fre_mean) + tem_sea_mean/ torch.norm(tem_sea_mean))

                    sea_tmp_mean = torch.mean(torch.tensor(features_sea[np.array(torch.where(tem_im2cluster == tmpp)[0].cpu()), :]), 0).cuda()
                    sea_fre_mean = torch.mean(torch.tensor(features_sea[np.array(torch.where(fre_im2cluster == tmpp)[0].cpu()), :]), 0).cuda()
                    cluster_result_sea['centroids'][tmp][tmpp, :] = torch.mean(sea_tmp_mean/ torch.norm(sea_tmp_mean) + sea_fre_mean/ torch.norm(sea_fre_mean))

            cluster_result_tem['ma_centroids'] = cluster_result_tem['centroids']
            cluster_result_fre['ma_centroids'] = cluster_result_fre['centroids']
            cluster_result_sea['ma_centroids'] = cluster_result_sea['centroids']

    def fit_ts_cot(self, train_data, n_epochs=None,logger=None):

        train_dataset1 = ThreeViewloader(train_data)
        train_loader = DataLoader(train_dataset1, batch_size=min(self.batch_size, len(train_dataset1)), shuffle=True,
                                  drop_last=False)


        params = list(list(self.tem_encoder.parameters()) + list(self.fre_encoder.parameters())+list(self.sea_encoder.parameters()))
        optimizer = torch.optim.AdamW(params, lr=self.lr)

        cluster_result = None
        last_clusters = None

        self.args.warmup = int(self.args.warmup * self.args.epochs)
        while True:
            # if self.n_epochs>0:
            #     print('Training Epoch '+str(self.n_epochs))
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break
            batch_loss_tem_nce = [0.0]
            batch_loss_fre_nce = [0.0]
            batch_loss_sea_nce = [0.0]
            batch_loss_proto_tem = [0.0]
            batch_loss_proto_fre = [0.0]
            batch_loss_proto_sea = [0.0]

            if self.n_epochs == max(int(self.args.warmup), 0):
                
                features = self.encode(train_data )

                feature_split = int(features.shape[1]/3)
                if np.any(np.isnan(features)).item():
                    continue
                features_tem = features[:, 0:feature_split]
                features_fre = features[:, feature_split:2*feature_split]
                features_sea = features[:, 2*feature_split:]

                cluster_result_tem =  {'im2cluster': [], 'centroids': [], 'density': [], 'ma_centroids':[]}
                cluster_result_fre = {'im2cluster': [], 'centroids': [], 'density': [],  'ma_centroids':[]}
                cluster_result_sea = {'im2cluster': [], 'centroids': [], 'density': [],  'ma_centroids':[]}

                for num_cluster in self.args.num_cluster:
                    cluster_result_tem['im2cluster'].append(torch.zeros(len(train_loader), dtype=torch.long).cuda())
                    cluster_result_fre['im2cluster'].append(torch.zeros(len(train_loader), dtype=torch.long).cuda())
                    cluster_result_sea['im2cluster'].append(torch.zeros(len(train_loader), dtype=torch.long).cuda())
                    cluster_result_tem['centroids'].append(
                        torch.zeros(int(num_cluster), train_data[0].shape[1] * self.args.repr_dims).cuda())
                    cluster_result_fre['centroids'].append(
                        torch.zeros(int(num_cluster), train_data[0].shape[1] * self.args.repr_dims).cuda())
                    cluster_result_sea['centroids'].append(
                        torch.zeros(int(num_cluster), train_data[0].shape[1] * self.args.repr_dims).cuda())
                    cluster_result_tem['density'].append(torch.zeros(int(num_cluster)).cuda())
                    cluster_result_fre['density'].append(torch.zeros(int(num_cluster)).cuda())
                    cluster_result_sea['density'].append(torch.zeros(int(num_cluster)).cuda())
                    cluster_result_tem = run_kmeans(features_tem, self.args, last_clusters)
                    cluster_result_fre = run_kmeans(features_fre, self.args, last_clusters)
                    cluster_result_sea = run_kmeans(features_sea, self.args, last_clusters)

                for tmp in range(len(self.args.num_cluster)):
                    tem_im2cluster = cluster_result_tem['im2cluster'][tmp]
                    fre_im2cluster = cluster_result_fre['im2cluster'][tmp]
                    sea_im2cluster = cluster_result_sea['im2cluster'][tmp]
                    dist_tem = cluster_result_tem['distance_2_center'][tmp]
                    dist_fre = cluster_result_fre['distance_2_center'][tmp]
                    dist_sea = cluster_result_sea['distance_2_center'][tmp]
                    print("=="*50)
                    print(tem_im2cluster)
                    print(fre_im2cluster)
                    print(sea_im2cluster)
                    tmp_cluster = self.args.num_cluster[tmp]
                    for tmpp in range(int(tmp_cluster)):
                        sort_tem_index = np.array(np.argsort(dist_tem[:,tmpp])) #
                        sort_fre_index = np.array(np.argsort(dist_fre[:,tmpp])) #
                        sort_sea_index = np.array(np.argsort(dist_sea[:,tmpp])) #
                        sort_tem_index = sort_tem_index[:int(0.8*len(np.array(torch.where(tem_im2cluster == tmpp)[0].cpu())))]
                        sort_fre_index = sort_fre_index[:int(0.8*len(np.array(torch.where(fre_im2cluster == tmpp)[0].cpu())))]
                        sort_sea_index = sort_sea_index[:int(0.8*len(np.array(torch.where(sea_im2cluster == tmpp)[0].cpu())))]

                        set_tem = np.intersect1d(np.array(torch.where(tem_im2cluster == tmpp)[0].cpu()), sort_tem_index)
                        set_fre = np.intersect1d(np.array(torch.where(fre_im2cluster == tmpp)[0].cpu()), sort_fre_index)
                        set_sea = np.intersect1d(np.array(torch.where(sea_im2cluster == tmpp)[0].cpu()), sort_sea_index)
                        
                        neg_tem = np.setdiff1d(np.array(torch.where(tem_im2cluster == tmpp)[0].cpu()), sort_tem_index)
                        neg_fre = np.setdiff1d(np.array(torch.where(fre_im2cluster == tmpp)[0].cpu()), sort_fre_index)
                        neg_sea = np.setdiff1d(np.array(torch.where(sea_im2cluster == tmpp)[0].cpu()), sort_sea_index)
                        if len(neg_tem) > 0 and len(set_tem) > 0:
                            tem_im2cluster[neg_tem] = -1
                        if len(neg_fre) > 0 and len(set_fre) > 0:
                            fre_im2cluster[neg_fre] = -1
                        if len(neg_sea) > 0 and len(set_fre) > 0:
                            sea_im2cluster[neg_sea] = -1
                        
                        fre_tem_mean = torch.mean(torch.tensor(features_fre[np.array(torch.where(tem_im2cluster == tmpp)[0].cpu()), :]),0).cuda()
                        fre_sea_mean = torch.mean(torch.tensor(features_fre[np.array(torch.where(sea_im2cluster == tmpp)[0].cpu()), :]),0).cuda()
                        cluster_result_fre['centroids'][tmp][tmpp, :] =  torch.mean(fre_tem_mean/ torch.norm(fre_tem_mean) + fre_sea_mean/ torch.norm(fre_sea_mean))

                        tem_fre_mean = torch.mean(torch.tensor(features_tem[np.array(torch.where(fre_im2cluster == tmpp)[0].cpu()), :]), 0).cuda()
                        tem_sea_mean = torch.mean(torch.tensor(features_tem[np.array(torch.where(sea_im2cluster == tmpp)[0].cpu()), :]), 0).cuda()
                        cluster_result_tem['centroids'][tmp][tmpp, :] = torch.mean(tem_fre_mean/ torch.norm(tem_fre_mean) + tem_sea_mean/ torch.norm(tem_sea_mean))

                        sea_tmp_mean = torch.mean(torch.tensor(features_sea[np.array(torch.where(tem_im2cluster == tmpp)[0].cpu()), :]), 0).cuda()
                        sea_fre_mean = torch.mean(torch.tensor(features_sea[np.array(torch.where(fre_im2cluster == tmpp)[0].cpu()), :]), 0).cuda()
                        cluster_result_sea['centroids'][tmp][tmpp, :] = torch.mean(sea_tmp_mean/ torch.norm(sea_tmp_mean) + tem_sea_mean/ torch.norm(sea_fre_mean))

                        # cluster_result_sea['centroids'][tmp][tmpp, :] = torch.mean(
                        #     torch.tensor(features_sea[np.array(torch.where(sea_im2cluster == tmpp)[0].cpu()), :]),
                        #     0).cuda() / torch.norm(torch.mean(
                        #     torch.tensor(features_sea[np.array(torch.where(sea_im2cluster == tmpp)[0].cpu()), :]),
                        #     0).cuda())
                cluster_result_tem['ma_centroids'] = cluster_result_tem['centroids']
                cluster_result_fre['ma_centroids'] = cluster_result_fre['centroids']
                cluster_result_sea['ma_centroids'] = cluster_result_sea['centroids']

            if self.n_epochs > max(int(self.args.warmup), 0):
                features = self.encode(train_data)
                feature_split = int(features.shape[1]/3)
                if np.any(np.isnan(features)).item():
                    continue
                features_tem = features[:, :feature_split]
                features_fre = features[:, feature_split:2*feature_split]
                features_sea = features[:, 2*feature_split:]
                for jj in range(len(self.args.num_cluster)):
                    ma_centroids_tem = cluster_result_tem['ma_centroids'][jj]/torch.norm(cluster_result_tem['ma_centroids'][jj], dim=1, keepdim=True)
                    cp_tem = torch.matmul(torch.tensor(features_tem).cuda(), ma_centroids_tem.transpose(1, 0))
                    cluster_result_tem['im2cluster'][jj] = torch.argmax(cp_tem, dim=1)
                    cluster_result_tem['distance_2_center'][jj] = 1-cp_tem.cpu().numpy()

                    ma_centroids_fre = cluster_result_fre['ma_centroids'][jj]/torch.norm(cluster_result_fre['ma_centroids'][jj], dim=1, keepdim=True)
                    cp_fre = torch.matmul(torch.tensor(features_fre).cuda(), ma_centroids_fre.transpose(1, 0))
                    cluster_result_fre['im2cluster'][jj] = torch.argmax(cp_fre, dim=1)
                    cluster_result_fre['distance_2_center'][jj] = 1-cp_fre.cpu().numpy()
                    
                    ma_centroids_sea = cluster_result_sea['ma_centroids'][jj]/torch.norm(cluster_result_sea['ma_centroids'][jj], dim=1, keepdim=True)
                    cp_sea = torch.matmul(torch.tensor(features_sea).cuda(), ma_centroids_sea.transpose(1, 0))
                    cluster_result_sea['im2cluster'][jj] = torch.argmax(cp_sea, dim=1)
                    cluster_result_sea['distance_2_center'][jj] = 1-cp_sea.cpu().numpy()

                    cluster_result_tem['density'][jj] = torch.ones(cluster_result_tem['density'][jj].shape).cuda()
                    cluster_result_fre['density'][jj] = torch.ones(cluster_result_fre['density'][jj].shape).cuda()
                    cluster_result_sea['density'][jj] = torch.ones(cluster_result_sea['density'][jj].shape).cuda()

                cluster_result_tem = run_kmeans(features_tem, self.args, last_clusters)
                cluster_result_fre = run_kmeans(features_fre, self.args, last_clusters)
                cluster_result_sea = run_kmeans(features_sea, self.args, last_clusters)

                for tmp in range(len(self.args.num_cluster)):
                    tem_im2cluster = cluster_result_tem['im2cluster'][tmp]
                    fre_im2cluster = cluster_result_fre['im2cluster'][tmp]
                    sea_im2cluster = cluster_result_sea['im2cluster'][tmp]

                    dist_tem = cluster_result_tem['distance_2_center'][tmp]
                    dist_fre = cluster_result_fre['distance_2_center'][tmp]
                    dist_sea = cluster_result_sea['distance_2_center'][tmp]

                    tmp_cluster = self.args.num_cluster[tmp]
                    for tmpp in range(int(tmp_cluster)):
                        keep_ratio = 1
                        sort_tem_index = np.array(np.argsort(dist_tem[:,tmpp])) # 前面的距离小
                        sort_fre_index = np.array(np.argsort(dist_fre[:,tmpp])) #
                        sort_sea_index = np.array(np.argsort(dist_sea[:,tmpp])) #

                        sort_tem_index = sort_tem_index[:int(keep_ratio*len(np.array(torch.where(tem_im2cluster == tmpp)[0].cpu())))]
                        sort_fre_index = sort_fre_index[:int(keep_ratio*len(np.array(torch.where(fre_im2cluster == tmpp)[0].cpu())))]
                        sort_sea_index = sort_sea_index[:int(keep_ratio*len(np.array(torch.where(sea_im2cluster == tmpp)[0].cpu())))]

                        set_tem = np.intersect1d(np.array(torch.where(tem_im2cluster == tmpp)[0].cpu()), sort_tem_index)
                        set_fre = np.intersect1d(np.array(torch.where(fre_im2cluster == tmpp)[0].cpu()), sort_fre_index)
                        set_sea = np.intersect1d(np.array(torch.where(sea_im2cluster == tmpp)[0].cpu()), sort_sea_index)

                        neg_tem = np.setdiff1d(np.array(torch.where(tem_im2cluster == tmpp)[0].cpu()), sort_tem_index)
                        neg_fre = np.setdiff1d(np.array(torch.where(fre_im2cluster == tmpp)[0].cpu()), sort_fre_index)
                        neg_sea = np.setdiff1d(np.array(torch.where(sea_im2cluster == tmpp)[0].cpu()), sort_sea_index)

                        if len(neg_tem) > 0 and len(set_tem) > 0:
                            tem_im2cluster[neg_tem] = -1
                        if len(neg_fre) > 0 and len(set_fre) > 0:
                            fre_im2cluster[neg_fre] = -1
                        if len(neg_sea) > 0 and len(set_sea) > 0:
                            sea_im2cluster[neg_fre] = -1

                        fre_tem_mean = torch.mean(torch.tensor(features_fre[np.array(torch.where(tem_im2cluster == tmpp)[0].cpu()), :]),0).cuda()
                        fre_sea_mean = torch.mean(torch.tensor(features_fre[np.array(torch.where(sea_im2cluster == tmpp)[0].cpu()), :]),0).cuda()
                        cluster_result_fre['centroids'][tmp][tmpp, :] =  torch.mean(fre_tem_mean/ torch.norm(fre_tem_mean) + fre_sea_mean/ torch.norm(fre_sea_mean))

                        tem_fre_mean = torch.mean(torch.tensor(features_tem[np.array(torch.where(fre_im2cluster == tmpp)[0].cpu()), :]), 0).cuda()
                        tem_sea_mean = torch.mean(torch.tensor(features_tem[np.array(torch.where(sea_im2cluster == tmpp)[0].cpu()), :]), 0).cuda()
                        cluster_result_tem['centroids'][tmp][tmpp, :] = torch.mean(tem_fre_mean/ torch.norm(tem_fre_mean) + tem_sea_mean/ torch.norm(tem_sea_mean))

                        sea_tmp_mean = torch.mean(torch.tensor(features_sea[np.array(torch.where(tem_im2cluster == tmpp)[0].cpu()), :]), 0).cuda()
                        sea_fre_mean = torch.mean(torch.tensor(features_sea[np.array(torch.where(fre_im2cluster == tmpp)[0].cpu()), :]), 0).cuda()
                        cluster_result_sea['centroids'][tmp][tmpp, :] = torch.mean(sea_tmp_mean/ torch.norm(sea_tmp_mean) + sea_fre_mean/ torch.norm(sea_fre_mean))

                cluster_result_tem['ma_centroids'] = cluster_result_tem['centroids']
                cluster_result_fre['ma_centroids'] = cluster_result_fre['centroids']
                cluster_result_sea['ma_centroids'] = cluster_result_sea['centroids']

            tq_bar = tqdm(train_loader) 
            for indexs, sample_tem, sample_fre,sample_sea in tq_bar:

                optimizer.zero_grad()
                sample_tem = sample_tem.to(self.device)
                
                sample_fre = sample_fre.to(self.device)
                sample_sea = sample_sea.to(self.device)
                tem_feat_1, tem_feat_2, tem_feat_3, tem_z = self.tem_encoder(sample_tem.float(), 1)
                # print("tem_z shape",tem_z.shape)
                # print("tem_feat_1 shape",tem_feat_1.shape)
                # print("tem_feat_2 shape",tem_feat_2.shape)
                # print("tem_feat_3 shape",tem_feat_3.shape)
                tem_feat_1_m, tem_feat_2_m, tem_feat_3_m, tem_z_m = self.tem_encoder(sample_tem.float(), self.args.dropmask, self.args.seed)
                fre_feat_1, fre_feat_2, fre_feat_3, fre_z = self.fre_encoder(sample_fre.float(), 1)
                fre_feat_1_m, fre_feat_2_m, fre_feat_3_m, fre_z_m = self.fre_encoder(sample_fre.float(), self.args.dropmask, self.args.seed)
                fre_feat_1, fre_feat_2, fre_feat_3, sea_z = self.sea_encoder(sample_sea.float(), 1)
                fre_feat_1_m, fre_feat_2_m, fre_feat_3_m, sea_z_m = self.sea_encoder(sample_sea.float(), self.args.dropmask, self.args.seed)

                criterion = InfoNCE(self.args.temperature)
                loss_tem_nce = criterion(tem_z.squeeze(-1), tem_z_m.squeeze(-1))
                batch_loss_tem_nce.append(loss_tem_nce.item())
                loss_fre_nce = criterion(fre_z.squeeze(-1), fre_z_m.squeeze(-1))
                batch_loss_fre_nce.append(loss_fre_nce.item())
                #print("loss_sea_nce : ",loss_fre_nce.item(),fre_z,fre_z_m)
                loss_sea_nce = criterion(sea_z.squeeze(-1), sea_z_m.squeeze(-1))
                #print("loss_sea_nce : ",loss_sea_nce.item(),sea_z,sea_z_m)
                batch_loss_sea_nce.append(loss_sea_nce.item())
                # loss = loss_fre_nce
                loss = loss_fre_nce + loss_tem_nce + loss_sea_nce

                if self.n_epochs > (self.args.warmup):
                    loss_prototype_tem, cluster_result_tem['ma_centroids']= prototype_loss_cotrain(tem_z , indexs, cluster_result_tem, self.args)
                    loss_prototype_fre, cluster_result_fre['ma_centroids'] = prototype_loss_cotrain(fre_z, indexs, cluster_result_fre, self.args)
                    loss_prototype_sea, cluster_result_sea['ma_centroids'] = prototype_loss_cotrain(sea_z, indexs, cluster_result_sea, self.args)

                    batch_loss_proto_tem.append(loss_prototype_tem.item())
                    batch_loss_proto_fre.append(loss_prototype_fre.item())
                    batch_loss_proto_sea.append(loss_prototype_sea.item())
                    
                    loss += loss_prototype_tem * self.args.prototype_lambda
                    loss += loss_prototype_fre * self.args.prototype_lambda
                    loss += loss_prototype_sea * self.args.prototype_lambda

                tq_bar.set_description(f"epoch: {self.n_epochs}/{self.args.epochs} | loss : {loss.item()}")
                
                loss.backward()
                optimizer.step()
            if logger :
                    logger.info(f"epoch: {self.n_epochs}/{self.args.epochs} | loss : {loss.item()}")
            self.n_epochs += 1
        self.cluster_result = cluster_result
        return 0


    def encode(self, data, batch_size=None):
        self.tem_encoder.eval()
        self.fre_encoder.eval()
        self.sea_encoder.eval()

        train_dataset1 = ThreeViewloader(data)
        loader = DataLoader(train_dataset1, batch_size=min(self.batch_size
                                                           , len(train_dataset1)), shuffle=False,
                                  drop_last=False)

        with torch.no_grad():
            output = []
            for indexs, sample_tem, sample_fre,sample_sea in loader:
                sample_tem = sample_tem.to(self.device)
                sample_fre = sample_fre.to(self.device)
                sample_sea = sample_sea.to(self.device)
                _, _, _, tem_z = self.tem_encoder(sample_tem.float())
                _, _, _, fre_z = self.fre_encoder(sample_fre.float())
                _, _, _, sea_z = self.sea_encoder(sample_sea.float())
                out = torch.cat((tem_z.squeeze(-1), fre_z.squeeze(-1),sea_z.squeeze(-1)), dim=1)
                output.append(out)

            output = torch.cat(output, dim=0)

        return output.cpu().numpy()

    def encode_online(self,data):

        sample_tem = data[0].to(self.device)
        sample_fre = data[0].to(self.device)
        sample_sea = data[0].to(self.device)

        _, _, _, tem_z = self.tem_encoder(sample_tem.float())
        _, _, _, fre_z = self.fre_encoder(sample_fre.float())
        _, _, _, sea_z = self.sea_encoder(sample_sea.float())
        out = torch.cat((tem_z.squeeze(-1), fre_z.squeeze(-1),sea_z.squeeze(-1)), dim=1)
        return out.cpu().numpy()

    def save(self, fn):
        print(fn)
        torch.save({'TemEncoder': self.tem_encoder.state_dict()
                    , 'FreEncoder': self.fre_encoder.state_dict()
                    ,'SeaEncoder': self.sea_encoder.state_dict()}
                    , fn)

    def load(self,fn):
        # state_dict = torch.load(fn, map_location=self.device)
        self.tem_encoder.load_state_dict(torch.load(fn,map_location=self.device)['TemEncoder'])
        self.fre_encoder.load_state_dict(torch.load(fn, map_location=self.device)['FreEncoder'])
        self.sea_encoder.load_state_dict(torch.load(fn, map_location=self.device)['SeaEncoder'])