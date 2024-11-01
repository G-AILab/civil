import numpy as np
from . import _eval_protocols as eval_protocols
from utils import *
import joblib


EVAL_PROTO = {
    'linear' : eval_protocols.pipe_lr,
    'svm' : eval_protocols.pipe_knn,
    'knn' : eval_protocols.pipe_knn,
    'mlp' : eval_protocols.Linear_probe
}


def eval_infer(data_repr,args=None):
    
    eval_protocol = args.eval_protocol if args.eval_protocol else  "mlp"
    
    if eval_protocol == 'mlp':
        mlp_model = eval_protocols.Linear_probe(args.in_dims,args.num_cluster)
        mlp_model.load_state_dict(torch.load(f'{args.model_path}/{eval_protocol}.pkl',map_location=args.device))
        test_pred = mlp_model(data_repr)
        test_pred = torch.functional.F.softmax(test_pred, dim=1)
        y_score = torch.argmax(test_pred, dim=1)
        # y_score, metricss = eval_protocols.fit_mlp(data_repr)
        return y_score
    
    else:
        #fit_clf = EVAL_PROTO[eval_protocol] #  eval_protocols.fit_knn
        fit_clf = joblib.load(f'{args.run_dir}/{eval_protocol}.joblib')

    if eval_protocol in ['linear','knn'] :
        y_score = fit_clf.predict_proba(data_repr)
    else:
        y_score = fit_clf.score(data_repr)
 

    return y_score,y_score.argmax(axis=1)


    

