
import torch
import numpy as np
import argparse
import os
from datetime import datetime
from algorithm.ts_sea import TS_SEA
from algorithm.ts_cot import TS_CoT
from algorithm.ts2vec import TS2Vec
from algorithm.ts2tcc import TS_TCC
import tasks
import datautils
from utils import init_cuda, get_logger
import random

from configs.LoadConfig import load_json_config,config_to_json

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def eval_mlp(device,logger,args):

    model,train_data,train_labels, test_data, test_labels = build_model(device=device,args=args)
    
    model.load(args.model_path)

    out, eval_res = tasks.eval_classification(model, train_data, train_labels, test_data, test_labels,args)
    for evals,val in eval_res.items() :
        logger.info(f"{evals} : {val}")
    # logger.info(f"Evaluation result: ACC: {eval_res['acc']}   AUROC: {eval_res['auroc']}")


def init_agrs(args):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.run_dir = os.path.join("exp_logs", args.run_desc, args.dataset,args.backbone_type,now) #'exp_logs/'+ args.run_desc + '/' + args.dataset + '/' + args.backbone_type+ '/'+ now
    
    os.makedirs(args.run_dir, exist_ok=True)

    # experiment_log_dir = os.path.join("exp_logs", args.run_desc, args.dataset,args.backbone_type)
    # os.makedirs(args.run_dir, exist_ok=True)
    # Logging
    log_file_name = os.path.join(args.run_dir, f"training.log")
    args.log_file_name = log_file_name

    if args.dataloader is None :
        args.dataloader = args.dataset
    
    
    if args.eval:
        args.epochs = 0
    
    return args 


def build_model(device,args):
    
    ##############################################################################
    ###### 加载模型骨架
    ##############################################################################

    if args.backbone_type =="TS_CoT":


        train_data, train_labels, test_data, test_labels = datautils.load_itwo_view(args)
        args.in_dims = train_data[0].shape[-1]
        model = TS_CoT(
                    input_dims=train_data[0].shape[-1],
                    output_dims=args.repr_dims,
                    device=device,
                    args=args
                )
    elif args.backbone_type =="TS_SEA":

        # print(type(args.data_perc),2)
        train_data, train_labels, test_data, test_labels = datautils.load_itri_view(args)

        args.in_dims = train_data[0].shape[-1]
        model = TS_SEA(
                    input_dims=train_data[0].shape[-1],
                    output_dims=args.repr_dims,
                    device=device,
                    args=args
                )
    elif args.backbone_type =="TS2Vec":

        train_data, train_labels, test_data, test_labels = datautils.get_ts2vec_loader(args)

        config = dict(
            batch_size=args.batch_size,
            lr=args.lr,
            output_dims=args.repr_dims,
            max_train_length=args.max_train_length
        )
        args.in_dims = train_data.shape[-1]
        model = TS2Vec(
            input_dims=train_data.shape[-1],
            device=device,
            **config
        )
    elif args.backbone_type =="TS_TCC":

        train_data, train_labels, test_data, test_labels = datautils.get_ts2vec_loader(args)


        args.in_dims = args.input_channels
        model = TS_TCC(
            device=device,
            args=args
        )
    else :
        raise Exception("Unknown Backbone")
    
    return model,train_data,train_labels, test_data, test_labels
    

def train_model(config_path="configs/ts_cot.json"):
    
    args = load_json_config(config_path)
    
    args = init_agrs(args)
    
    logger = get_logger(args.log_file_name)

    device = init_cuda(args.gpu, seed=args.seed, max_threads=args.max_threads)
    logger.info("=====================================================================")
    for key,val in args.items():
        logger.info(f"===== {str(key),str(val)}")
    logger.info("=====================================================================")
    
    logger.info('Loading data... ')

    logger.info(f"Backbone is {args.backbone_type}")
    
    ##############################################################################
    ###### 模型训练
    ##############################################################################

    if not args.eval:
        model,train_data,_,_,_ = build_model(device=device,args=args)
        
        model.fit_ts_cot(
            train_data,
            n_epochs=args.epochs
            ,logger=logger
        )

        args.model_path = f'{args.run_dir}/model.pkl'
        model.save(args.model_path)
        
        args.eval = True 
        
        logger.info(f"saving model to  :{args.model_path}")
    else:
        logger.info('Unknown Backbone')

    ##############################################################################
    ###### 模型评估
    ######
    ##############################################################################
    logger.info(args.eval)
    
    if args.eval:
        eval_mlp(device=device,logger=logger,args=args)
    config_to_json(args,f'{args.run_dir}/config.json')
    
    logger.info(os.path.basename(__file__))
    logger.info("Finished.")


if __name__ == '__main__':
    # train_model("/workspace/Civil/configs/ts_cot/ts_cot_4_epi.json")
    # train_model("configs/ts_sea/ts_sea_4_har.json")
    # train_model("configs/ts2vec/ts2vec_4_har.json")
    # train_model("configs/ts_cot/ts_cot_4_har.json")
    train_model("configs/ts2tcc/ts2tcc_4_har.json")
