import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from models import TSEncoder
from models.ts2vec.losses import hierarchical_contrastive_loss
from utils import take_per_row, split_with_nan, centerize_vary_length_series, torch_pad_nan,run_kmeans,prototype_loss_cotrain
import math
from dataloader  import TwoViewloader

class TS2Dual:
    '''The TS2Dual model'''
    
    def __init__(
        self,
        input_dims,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        device='cuda',
        lr=0.001,
        batch_size=16,
        max_train_length=None,
        temporal_unit=0,
        after_iter_callback=None,
        after_epoch_callback=None,
        args=None
    ):
        ''' Initialize a TS2Dual model.
        
        Args:
            input_dims (int): The input dimension. For a univariate time series, this should be set to 1.
            output_dims (int): The representation dimension.
            hidden_dims (int): The hidden dimension of the encoder.
            depth (int): The number of hidden residual blocks in the encoder.
            device (int): The gpu used for training and inference.
            lr (int): The learning rate.
            batch_size (int): The batch size.
            max_train_length (Union[int, NoneType]): The maximum allowed sequence length for training. For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length>.
            temporal_unit (int): The minimum unit to perform temporal contrast. When training on a very long sequence, this param helps to reduce the cost of time and memory.
            after_iter_callback (Union[Callable, NoneType]): A callback function that would be called after each iteration.
            after_epoch_callback (Union[Callable, NoneType]): A callback function that would be called after each epoch.
        '''
        
        super().__init__()
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_length = max_train_length
        self.temporal_unit = temporal_unit
        
        self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth).to(self.device)
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)

        self._net_freq = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth).to(self.device)
        self.net_freq = torch.optim.swa_utils.AveragedModel(self._net_freq)
        self.net_freq.update_parameters(self._net_freq)

        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback
        
        self.n_epochs = 0
        self.n_iters = 0
        self.args = args
    
    def fit_ts_cot(self, train_data, n_epochs=None, n_iters=None, verbose=False,logger=None):
        ''' Training the TS2Vec model.
        
        Args:
            train_data (numpy.ndarray): The training data. It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
            n_iters (Union[int, NoneType]): The number of iterations. When this reaches, the training stops. If both n_epochs and n_iters are not specified, a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
            verbose (bool): Whether to print the training loss after each epoch.
            
        Returns:
            loss_log: a list containing the training losses on each epoch.
        '''
        assert train_data[0].ndim == 3
        
        if n_iters is None and n_epochs is None:
            n_iters = 200 #if train_data[0].shape[0] <= 100000 else 600  # default param for n_iters
        
        if self.max_train_length is not None:
            sections = train_data[0].shape[1] // self.max_train_length
            if sections >= 2:
                train_data[0] = np.concatenate(split_with_nan(train_data[0], sections, axis=1), axis=0)
                train_data[1] = np.concatenate(split_with_nan(train_data[1], sections, axis=1), axis=0)

        temporal_missing = np.isnan(train_data[0]).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            train_data[0] = centerize_vary_length_series(train_data[0])

        temporal_missing = np.isnan(train_data[1]).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            train_data[1] = centerize_vary_length_series(train_data[1])
        print("train_data[0]",train_data[0].shape)      
        train_data[0] = train_data[0][~np.isnan(train_data[0]).all(axis=2).all(axis=1)]
        train_data[1] = train_data[1][~np.isnan(train_data[1]).all(axis=2).all(axis=1)]
        
        train_dataset = TwoViewloader(train_data)
        #train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float))
        train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True, drop_last=True)
        
        optimizer = torch.optim.AdamW(list(self._net.parameters())+list(self._net_freq.parameters()), lr=self.lr)
        
        loss_log = []
        cluster_result = None
        last_clusters = None
        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break
            
            cum_loss = 0
            n_epoch_iters = 0

            if self.n_epochs == max(int(self.args.warmup), 0):
                features = self.encode(train_data )
                feature_split = int(features.shape[1]/2)
                features_tem = features[:, :feature_split]
                features_fre = features[:, feature_split:]
                #print("features_fre shape",features_fre.shape)
                cluster_result_tem =  {'im2cluster': [], 'centroids': [], 'density': [], 'ma_centroids':[]}
                cluster_result_fre = {'im2cluster': [], 'centroids': [], 'density': [],  'ma_centroids':[]}

                for num_cluster in self.args.num_cluster:
                    cluster_result_tem['im2cluster'].append(torch.zeros(len(train_loader), dtype=torch.long).cuda())
                    cluster_result_fre['im2cluster'].append(torch.zeros(len(train_loader), dtype=torch.long).cuda())
                    cluster_result_tem['centroids'].append(
                        torch.zeros(int(num_cluster),  self.args.repr_dims).cuda())
                    cluster_result_fre['centroids'].append(
                        torch.zeros(int(num_cluster),  self.args.repr_dims).cuda())
                    cluster_result_tem['density'].append(torch.zeros(int(num_cluster)).cuda())
                    cluster_result_fre['density'].append(torch.zeros(int(num_cluster)).cuda())
                    cluster_result_tem = run_kmeans(features_tem, self.args, last_clusters)
                    cluster_result_fre = run_kmeans(features_fre, self.args, last_clusters)

                for tmp in range(len(self.args.num_cluster)):
                    tem_im2cluster = cluster_result_tem['im2cluster'][tmp]
                    fre_im2cluster = cluster_result_fre['im2cluster'][tmp]
                    dist_tem = cluster_result_tem['distance_2_center'][tmp]
                    dist_fre = cluster_result_fre['distance_2_center'][tmp]
                    # print("=="*50)
                    # print(tem_im2cluster)
                    # print(fre_im2cluster)
                    tmp_cluster = self.args.num_cluster[tmp]
                    for tmpp in range(int(tmp_cluster)):
                        sort_tem_index = np.array(np.argsort(dist_tem[:,tmpp])) #
                        sort_fre_index = np.array(np.argsort(dist_fre[:,tmpp])) #
                        sort_tem_index = sort_tem_index[:int(0.8*len(np.array(torch.where(tem_im2cluster == tmpp)[0].cpu())))]
                        sort_fre_index = sort_fre_index[:int(0.8*len(np.array(torch.where(fre_im2cluster == tmpp)[0].cpu())))]

                        set_tem = np.intersect1d(np.array(torch.where(tem_im2cluster == tmpp)[0].cpu()), sort_tem_index)
                        set_fre = np.intersect1d(np.array(torch.where(fre_im2cluster == tmpp)[0].cpu()), sort_fre_index)
                        neg_tem = np.setdiff1d(np.array(torch.where(tem_im2cluster == tmpp)[0].cpu()), sort_tem_index)
                        neg_fre = np.setdiff1d(np.array(torch.where(fre_im2cluster == tmpp)[0].cpu()), sort_fre_index)
                        if len(neg_tem) > 0 and len(set_tem) > 0:
                            tem_im2cluster[neg_tem] = -1
                        if len(neg_fre) > 0 and len(set_fre) > 0:
                            fre_im2cluster[neg_fre] = -1
                        cluster_result_fre['centroids'][tmp][tmpp, :] = torch.mean(
                            torch.tensor(features_fre[np.array(torch.where(tem_im2cluster == tmpp)[0].cpu()), :]),
                            0).cuda() / torch.norm(torch.mean(
                            torch.tensor(features_fre[np.array(torch.where(tem_im2cluster == tmpp)[0].cpu()), :]),
                            0).cuda())
                        cluster_result_tem['centroids'][tmp][tmpp, :] = torch.mean(
                            torch.tensor(features_tem[np.array(torch.where(fre_im2cluster == tmpp)[0].cpu()), :]),
                            0).cuda() / torch.norm(torch.mean(
                            torch.tensor(features_tem[np.array(torch.where(fre_im2cluster == tmpp)[0].cpu()), :]),
                            0).cuda())
                cluster_result_tem['ma_centroids'] = cluster_result_tem['centroids']
                cluster_result_fre['ma_centroids'] = cluster_result_fre['centroids']

            if self.n_epochs > max(int(self.args.warmup), 0):
                features = self.encode(train_data )
                feature_split = int(features.shape[1]/2)
                features_tem = features[:, :feature_split]
                features_fre = features[:, feature_split:]
                for jj in range(len(self.args.num_cluster)):
                    ma_centroids_tem = cluster_result_tem['ma_centroids'][jj]/torch.norm(cluster_result_tem['ma_centroids'][jj], dim=1, keepdim=True)
                    cp_tem = torch.matmul(torch.tensor(features_tem).cuda(), ma_centroids_tem.transpose(1, 0))
                    cluster_result_tem['im2cluster'][jj] = torch.argmax(cp_tem, dim=1)
                    cluster_result_tem['distance_2_center'][jj] = 1-cp_tem.cpu().numpy()
                    ma_centroids_fre = cluster_result_fre['ma_centroids'][jj]/torch.norm(cluster_result_fre['ma_centroids'][jj], dim=1, keepdim=True)
                    cp_fre = torch.matmul(torch.tensor(features_fre).cuda(), ma_centroids_fre.transpose(1, 0))
                    cluster_result_fre['im2cluster'][jj] = torch.argmax(cp_fre, dim=1)
                    cluster_result_fre['distance_2_center'][jj] = 1-cp_fre.cpu().numpy()
                    cluster_result_tem['density'][jj] = torch.ones(cluster_result_tem['density'][jj].shape).cuda()
                    cluster_result_fre['density'][jj] = torch.ones(cluster_result_fre['density'][jj].shape).cuda()

                cluster_result_tem = run_kmeans(features_tem, self.args, last_clusters)
                cluster_result_fre = run_kmeans(features_fre, self.args, last_clusters)

                for tmp in range(len(self.args.num_cluster)):
                    tem_im2cluster = cluster_result_tem['im2cluster'][tmp]
                    fre_im2cluster = cluster_result_fre['im2cluster'][tmp]

                    dist_tem = cluster_result_tem['distance_2_center'][tmp]
                    dist_fre = cluster_result_fre['distance_2_center'][tmp]
                    tmp_cluster = self.args.num_cluster[tmp]
                    for tmpp in range(int(tmp_cluster)):
                        keep_ratio = 1
                        sort_tem_index = np.array(np.argsort(dist_tem[:,tmpp])) # 前面的距离小
                        sort_fre_index = np.array(np.argsort(dist_fre[:,tmpp])) #
                        sort_tem_index = sort_tem_index[:int(keep_ratio*len(np.array(torch.where(tem_im2cluster == tmpp)[0].cpu())))]
                        sort_fre_index = sort_fre_index[:int(keep_ratio*len(np.array(torch.where(fre_im2cluster == tmpp)[0].cpu())))]
                        set_tem = np.intersect1d(np.array(torch.where(tem_im2cluster == tmpp)[0].cpu()), sort_tem_index)
                        set_fre = np.intersect1d(np.array(torch.where(fre_im2cluster == tmpp)[0].cpu()), sort_fre_index)
                        neg_tem = np.setdiff1d(np.array(torch.where(tem_im2cluster == tmpp)[0].cpu()), sort_tem_index)
                        neg_fre = np.setdiff1d(np.array(torch.where(fre_im2cluster == tmpp)[0].cpu()), sort_fre_index)

                        if len(neg_tem) > 0 and len(set_tem) > 0:
                            tem_im2cluster[neg_tem] = -1
                        if len(neg_fre) > 0 and len(set_fre) > 0:
                            fre_im2cluster[neg_fre] = -1

                        cluster_result_fre['centroids'][tmp][tmpp, :] = torch.mean(
                            torch.tensor(features_fre[np.array(torch.where(tem_im2cluster == tmpp)[0].cpu()), :]),
                            0).cuda() / torch.norm(torch.mean(
                            torch.tensor(features_fre[np.array(torch.where(tem_im2cluster == tmpp)[0].cpu()), :]),
                            0).cuda())
                        cluster_result_tem['centroids'][tmp][tmpp, :] = torch.mean(
                            torch.tensor(features_tem[np.array(torch.where(fre_im2cluster == tmpp)[0].cpu()), :]),
                            0).cuda() / torch.norm(torch.mean(
                            torch.tensor(features_tem[np.array(torch.where(fre_im2cluster == tmpp)[0].cpu()), :]),
                            0).cuda())
                cluster_result_tem['ma_centroids'] = cluster_result_tem['centroids']
                cluster_result_fre['ma_centroids'] = cluster_result_fre['centroids']


            interrupted = False
            for batch,x,fft_x in train_loader:
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break
                # print(x.shape)
                # x = batch[0]
                if self.max_train_length is not None and x.size(1) > self.max_train_length:
                    window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                    x = x[:, window_offset : window_offset + self.max_train_length]
                x =x.float()
                x = x.to(self.device)
                fft_x =fft_x.float()
                fft_x = fft_x.to(self.device)
                # x,fft_x = self.decomp(x)
                ts_l = x.size(1)
                crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l+1)
                crop_left = np.random.randint(ts_l - crop_l + 1)
                crop_right = crop_left + crop_l
                crop_eleft = np.random.randint(crop_left + 1)
                crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))
                
                optimizer.zero_grad()
                
                out1 = self._net(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft))
                out1 = out1[:, -crop_l:]
                
                out2 = self._net(take_per_row(x, crop_offset + crop_left, crop_eright - crop_left))
                out2 = out2[:, :crop_l]
                
                fft_ts_l = fft_x.size(1)
                fft_crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=fft_ts_l+1)
                fft_crop_left = np.random.randint(fft_ts_l - fft_crop_l + 1)
                fft_crop_right = fft_crop_left + fft_crop_l
                fft_crop_eleft = np.random.randint(fft_crop_left + 1)
                fft_crop_eright = np.random.randint(low=fft_crop_right, high=fft_ts_l + 1)
                fft_crop_offset = np.random.randint(low=-fft_crop_eleft, high=fft_ts_l - fft_crop_eright + 1, size=fft_x.size(0))
                
                optimizer.zero_grad()
                
                out3 = self._net_freq(take_per_row(fft_x, fft_crop_offset + fft_crop_eleft, fft_crop_right - fft_crop_eleft))
                out3 = out3[:, -fft_crop_l:]
                
                out4 = self._net_freq(take_per_row(fft_x, fft_crop_offset + fft_crop_left, fft_crop_eright - fft_crop_left))
                out4 = out4[:, :fft_crop_l]


                loss = hierarchical_contrastive_loss(
                    out1,
                    out2,
                    temporal_unit=self.temporal_unit
                )

                loss2 = hierarchical_contrastive_loss(
                    out3,
                    out4,
                    temporal_unit=self.temporal_unit
                )
                loss = loss + loss2 

                #criterion = InfoNCE(self.args.temperature)
                #loss_tem_nce = criterion(tem_z.squeeze(-1), tem_z_m.squeeze(-1))

                if self.n_epochs > (self.args.warmup):
                    #print(batch)
                    #print("out1 shape",out1.shape)
                    #print("out3 shape",out3.shape)
                    temp = torch.mean(out1,dim=1) # out1.reshape(out1.shape[0],-1)
                    freq = torch.mean(out3,dim=1) # out1.reshape(out1.shape[0],-1)
                    #freq = out3.reshape(out3.shape[0],-1)
                    #print("temp shape",temp.shape)
                    #print("freq shape",freq.shape)
                    loss_prototype_tem, cluster_result_tem['ma_centroids']= prototype_loss_cotrain(temp , batch, cluster_result_tem, self.args)
                    loss_prototype_fre, cluster_result_fre['ma_centroids'] = prototype_loss_cotrain(freq, batch, cluster_result_fre, self.args)
                    
                    # if torch.isnan(loss_prototype_tem).any() or torch.isnan(loss_prototype_fre).any():
                    #     logger.debug(f"Epoch #{self.n_epochs}: loss_prototype_tem loss={loss_prototype_tem} | loss_prototype_fre loss={loss_prototype_fre} ")
                    # if not torch.isnan(loss_prototype_tem).any():
                    #     loss += loss_prototype_tem * 0.8
                        
                    # if not torch.isnan(loss_prototype_fre).any():
                    loss += loss_prototype_fre * self.args.ma_alpha
                    loss += loss_prototype_tem * self.args.ma_alpha

                loss.backward()
                optimizer.step()
                self.net.update_parameters(self._net)
                self.net_freq.update_parameters(self._net_freq)
                    
                cum_loss += loss.item()
                n_epoch_iters += 1
                
                self.n_iters += 1
                
                if self.after_iter_callback is not None:
                    self.after_iter_callback(self, loss.item())
            
            if interrupted:
                break
            
            cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)
            if logger:
                logger.debug(f"Epoch #{self.n_epochs}: loss={cum_loss}")
            if verbose:
                print(f"Epoch #{self.n_epochs}: loss={cum_loss}")
            self.n_epochs += 1
            
            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self, cum_loss)
            
        return loss_log
    
    def _eval_with_pooling(self, x, mask=None, slicing=None, encoding_window=None):
        out = self.net(x.to(self.device, non_blocking=True), mask)
        if encoding_window == 'full_series':
            if slicing is not None:
                out = out[:, slicing]
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size = out.size(1),
            ).transpose(1, 2)
            
        elif isinstance(encoding_window, int):
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size = encoding_window,
                stride = 1,
                padding = encoding_window // 2
            ).transpose(1, 2)
            if encoding_window % 2 == 0:
                out = out[:, :-1]
            if slicing is not None:
                out = out[:, slicing]
            
        elif encoding_window == 'multiscale':
            p = 0
            reprs = []
            while (1 << p) + 1 < out.size(1):
                t_out = F.max_pool1d(
                    out.transpose(1, 2),
                    kernel_size = (1 << (p + 1)) + 1,
                    stride = 1,
                    padding = 1 << p
                ).transpose(1, 2)
                if slicing is not None:
                    t_out = t_out[:, slicing]
                reprs.append(t_out)
                p += 1
            out = torch.cat(reprs, dim=-1)
            
        else:
            if slicing is not None:
                out = out[:, slicing]
            
        return out.cpu()
    
    def _eval_with_pooling_freq(self, x, mask=None, slicing=None, encoding_window=None):
        out = self.net_freq(x.to(self.device, non_blocking=True), mask)
        if encoding_window == 'full_series':
            if slicing is not None:
                out = out[:, slicing]
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size = out.size(1),
            ).transpose(1, 2)
            
        elif isinstance(encoding_window, int):
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size = encoding_window,
                stride = 1,
                padding = encoding_window // 2
            ).transpose(1, 2)
            if encoding_window % 2 == 0:
                out = out[:, :-1]
            if slicing is not None:
                out = out[:, slicing]
            
        elif encoding_window == 'multiscale':
            p = 0
            reprs = []
            while (1 << p) + 1 < out.size(1):
                t_out = F.max_pool1d(
                    out.transpose(1, 2),
                    kernel_size = (1 << (p + 1)) + 1,
                    stride = 1,
                    padding = 1 << p
                ).transpose(1, 2)
                if slicing is not None:
                    t_out = t_out[:, slicing]
                reprs.append(t_out)
                p += 1
            out = torch.cat(reprs, dim=-1)
            
        else:
            if slicing is not None:
                out = out[:, slicing]
            
        return out.cpu()
    
    def encode(self, data, mask=None, encoding_window=None, causal=False, sliding_length=None, sliding_padding=0, batch_size=None):
        ''' Compute representations using the model.
        
        Args:
            data (numpy.ndarray): This should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            mask (str): The mask used by encoder can be specified with this parameter. This can be set to 'binomial', 'continuous', 'all_true', 'all_false' or 'mask_last'.
            encoding_window (Union[str, int]): When this param is specified, the computed representation would the max pooling over this window. This can be set to 'full_series', 'multiscale' or an integer specifying the pooling kernel size.
            causal (bool): When this param is set to True, the future informations would not be encoded into representation of each timestamp.
            sliding_length (Union[int, NoneType]): The length of sliding window. When this param is specified, a sliding inference would be applied on the time series.
            sliding_padding (int): This param specifies the contextual data length used for inference every sliding windows.
            batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the same batch size as training.
            
        Returns:
            repr: The representations for data.
        '''
        assert self.net is not None, 'please train or load a net first'
  
        assert data[0].ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data[0].shape

        org_training = self.net.training
        org_training_freq = self.net_freq.training
        self.net.eval()
        train_dataset = TwoViewloader(data)
        #dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(train_dataset, batch_size=batch_size)
        
        with torch.no_grad():
            output = []
            for batch,x,fft_x in loader:
                #x = batch[0]

                x =x.float()
                fft_x =fft_x.float()
                
                out = self._eval_with_pooling(x, mask, encoding_window=encoding_window)
                #print("out1 shape:",out.shape)
                # if encoding_window == 'full_series':
                #     out = out.squeeze(1)

                out = torch.mean(out,dim=1) # out.reshape(out.shape[0],-1)
                #print("out1 shape:",out.shape)

                out_fft = self._eval_with_pooling_freq(fft_x, mask, encoding_window=encoding_window)
                #print("out shape:",out_fft.shape)
                # if encoding_window == 'full_series':
                #     out_fft = out_fft.squeeze(1)
                #print(out_fft.shape)
                out_fft= torch.mean(out_fft,dim=1)  #out_fft.reshape(out_fft.shape[0],-1)
            
                out = torch.cat((out, out_fft), dim=1)  

                #print("evental shape ",out.shape)   
                output.append(out)
                
            output = torch.cat(output, dim=0)
        #print(output.shape)    
        self.net.train(org_training)
        self.net_freq.train(org_training_freq)
        return output.numpy()
    
    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        torch.save(self.net.state_dict(), fn)
    
    def load(self, fn):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        state_dict = torch.load(fn, map_location=self.device)
        self.net.load_state_dict(state_dict)
    
