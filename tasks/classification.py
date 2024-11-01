import numpy as np
from . import _eval_protocols as eval_protocols
from utils import *
import datautils
from sklearn.metrics import average_precision_score,f1_score,classification_report,roc_auc_score
import joblib
# from . import _visualization as t_sne_visual
def merge_dim01(array):
    return array.reshape(array.shape[0]*array.shape[1], *array.shape[2:])

def eval_classification(model, train_data, train_labels, test_data
                        , test_labels, args=None):
    eval_protocol = args.eval_protocol if args.eval_protocol else  "mlp"
    
    train_repr = model.encode(train_data)
    test_repr = model.encode(test_data)
    # t_sne_visual.t_sne_visual(test_repr,test_labels,args.dataset)
    # val_data,val_y  = load_val_data(args)
    # print(train_data.shape)
    # val_repr = model.encode(val_data)

    print(test_repr.shape)
    if eval_protocol == 'mlp':
        repr_results = {}
        repr_results['train_repr'] = train_repr
        repr_results['test_repr'] = test_repr
        repr_results['train_labels'] = train_labels
        repr_results['test_labels'] = test_labels
        args.feat_dim = train_repr.shape[1]
        mlp_model,y_score, metricss = eval_protocols.fit_mlp(train_repr, train_labels, test_repr, test_labels)
        torch.save({'mlp_model': mlp_model.state_dict()}, f'{args.run_dir}/model_{eval_protocol}.pkl')
        return y_score, metricss
    
    elif eval_protocol == 'linear':
        fit_clf = eval_protocols.fit_lr
    elif eval_protocol == 'svm':
        fit_clf = eval_protocols.fit_svm
    elif eval_protocol == 'knn':
        fit_clf = eval_protocols.fit_knn
    else:
        assert False, 'unknown evaluation protocol'
    # 非mlp的处理方法 
    if train_labels.ndim == 2:
        train_repr = merge_dim01(train_repr)
        train_labels = merge_dim01(train_labels)
        test_repr = merge_dim01(test_repr)
        test_labels = merge_dim01(test_labels)
    # print(train_repr.shape)
    clf = fit_clf(train_repr, train_labels)
    if args :

        joblib.dump(clf, f'{args.run_dir}/{eval_protocol}.joblib')

    acc = clf.score(test_repr, test_labels)



    if eval_protocol in ['linear','knn'] :
        y_score = clf.predict_proba(test_repr)
    else:
        y_score = clf.score(test_repr)
 

    if int(test_labels.max()+1) >2:
        roc = roc_auc_score(test_labels, y_score, multi_class='ovr')
    else:
        roc = roc_auc_score(test_labels, y_score[:, 1], multi_class='ovr')
    mf1 = f1_score(test_labels,y_score.argmax(axis=1),average="macro")
    return y_score, { 'acc': acc,'mf1':mf1,'roc':roc}


    

