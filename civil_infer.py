
import torch
import numpy as np
import argparse
import os
from datetime import datetime
from algorithm.ts_sea import TS_SEA
from algorithm.ts_cot import TS_CoT
from algorithm.ts2vec import TS2Vec
from tasks import eval_protocols

import infer_datautils
from utils import init_cuda, get_logger
import random
import joblib
from configs.LoadConfig import load_json_config

TRANS_DATA_DICT = {
    "TS_CoT": infer_datautils.format_ts2cot,
    "TS_SEA": infer_datautils.format_ts2sea,
    "TS2Vec": infer_datautils.format_ts2vec,
}

EVAL_PROTO = {
    'linear' : eval_protocols.pipe_lr,
    'svm' : eval_protocols.pipe_knn,
    'if' : eval_protocols.pipe_if,
    'knn' : eval_protocols.pipe_knn,
    'mlp' : eval_protocols.Linear_probe
}


MODEL_CACHE = {

}

############ 模仿数据库 #################

MODEL_INFO_TMP = {
    '20241101_045329':'exp_logs/civil_bi/HAR/TS2Vec/20241101_045329/config.json',
    '20241101_051903':'exp_logs/civil_bi/HAR/TS2Vec/20241101_051903/config.json',
    '20241101_070207':'exp_logs/civil/HAR/TS_CoT/20241101_070207/config.json'
}

def build_model(device,args,):
    
    ##############################################################################
    ###### 加载模型骨架
    ##############################################################################

    if args.backbone_type =="TS_CoT":
        
        model = TS_CoT(
                    input_dims=args.in_dims,
                    output_dims=args.repr_dims,
                    device=device,
                    args=args
                )
    elif args.backbone_type =="TS_SEA":



        model = TS_SEA(
                    input_dims=args.in_dims,
                    output_dims=args.repr_dims,
                    device=device,
                    args=args
                )
    elif args.backbone_type =="TS2Vec":

        config = dict(
            batch_size=args.batch_size,
            lr=args.lr,
            output_dims=args.repr_dims,
            max_train_length=args.max_train_length
        )
        model = TS2Vec(
            input_dims=args.in_dims,
            device=device,
            **config
        )
    else :
        raise Exception("Unknown Backbone")
    
    return model



def inference(model_id,data):
    

    if model_id not in  MODEL_CACHE : 
        print("No hit cache and loading model and task protocol .... ")
        # 假设我的配置文件通过modelid查查询得到器路径
        config_path = MODEL_INFO_TMP[model_id]
        model,args,task_proto = infer_model(config_path)

        MODEL_CACHE[model_id] = {
            'model':model,
            'args':args,
            'task_proto':task_proto
        }

    # 获取缓存中的模型
    model = MODEL_CACHE[model_id]['model']
    args = MODEL_CACHE[model_id]['args']
    task_proto = MODEL_CACHE[model_id]['task_proto']
    
    # 数据预处理
    data = TRANS_DATA_DICT[args.backbone_type](data)
    data_repr = model.encode_online(data)

    
    if args.eval_protocol == 'mlp':
        data_repr = torch.from_numpy(data_repr)
        test_pred = task_proto(data_repr)
        test_pred = torch.functional.F.softmax(test_pred, dim=1)
        y_score = torch.argmax(test_pred, dim=1)
        return test_pred , y_score
    elif args.eval_protocol in ['linear','knn'] :
        y_score = task_proto.predict_proba(data_repr)
    else:
        y_score = task_proto.score(data_repr)
 
    return y_score,y_score.argmax(axis=1)


def infer_model(config_path="configs/ts_cot.json"):
    
    args = load_json_config(config_path)
    
    device = init_cuda(args.gpu, seed=args.seed, max_threads=args.max_threads)

    model= build_model(device=device,args=args)
    model.load(args.model_path)

    
    if args.eval_protocol == 'mlp':
        args.num_cluster = args.num_cluster[0]
        task_proto = eval_protocols.Linear_probe(args.feat_dim,args.num_cluster)
        # task_proto.load_state_dict(torch.load(f'{args.model_path}/{args.eval_protocol}.pkl',map_location=args.device))
        task_proto.load_state_dict(torch.load(f'{args.run_dir}/model_{args.eval_protocol}.pkl')['mlp_model'])
    else:
        #fit_clf = EVAL_PROTO[eval_protocol] #  eval_protocols.fit_knn
        task_proto = joblib.load(f'{args.run_dir}/{args.eval_protocol}.joblib')
        
    return model,args,task_proto

    # data = torch.randn(10,9,128)

    

    # data = TRANS_DATA_DICT[args.backbone_type](data)
    # data_repr = model.encode_online(data)

    # y_score,y_score1 = tasks.eval_infer(data_repr,args)
    # print(y_score1)

    # return y_score,y_score1



if __name__ == '__main__':
    # train_model("/workspace/Civil/configs/ts_cot/ts_cot_4_epi.json")
    # train_model("configs/ts_sea/ts_sea.json")
    # infer_model("/workspace/Civil/exp_logs/civil_bi/HAR/TS2Vec/20241101_045329/config.json")
    # data = torch.randn(10,9,128)
    dataset_root = "/workspace/CA-TCC/data"
    data_path = f"{dataset_root}/HAR/"
    train_ = torch.load(data_path + "test.pt")
    for idx in range(0,len(train_['samples']),100):
        print(idx)
        data = train_['samples'][idx:idx+100]
        y_score,y_label = inference("20241101_070207",data)
        print(y_label)
