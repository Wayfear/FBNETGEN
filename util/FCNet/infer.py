import torch
import argparse
import yaml
from model import SeqenceModel, FCNet
from dataloader import infer_dataloader
from pathlib import Path
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

def main(args):
    with open(args.config_filename) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    
    if config['model']['type'] == 'FCNet':
        dataset, labels, node_size, timeseries_size = \
            infer_dataloader(config['data'])

    xs, ys = torch.tril_indices(node_size, node_size, offset=-1)

    config['train']["seq_len"] = timeseries_size
    config['train']["node_size"] = node_size

    
    if config['model']['type'] == 'seq':
        model = SeqenceModel(config['model'], node_size, timeseries_size)

    elif config['model']['type'] == 'FCNet':
        model = FCNet(node_size, timeseries_size)

    model.load_state_dict(torch.load(Path(args.model_path)/'model.pt'))

    model.cuda()
    model.eval()

    features = []

    interval = 1000

    for d in dataset:
        
        outputs = []
        for index in range(0, xs.shape[0], interval):
            data = []
            for x, y in zip(xs[index: index+interval], ys[index: index+interval]):
                data.append(d[[x,y],:])
        
            data = torch.stack(data, dim=0).cuda()
            output = model(data)
            outputs.append(output[:, 1].detach().cpu().numpy())
        outputs = np.concatenate(outputs)
        features.append(outputs)
    
    features = np.array(features)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)

    linearmodel = ElasticNet(alpha=1.0, l1_ratio=0.2, fit_intercept=False).fit(X_train, y_train)

    select_feature = linearmodel.coef_!=0
    print('Used feature number: ', np.sum(select_feature))
    X_train = X_train[:, select_feature]
    X_test = X_test[:, select_feature]

    svm = SVC(gamma='auto', probability=True).fit(X_train, y_train)

    print("acc", svm.score(X_test, y_test))

    prob_result = svm.predict_proba(X_test)

    auc = roc_auc_score(y_test, prob_result[:, 1])

    print("auc", auc)


    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='result/02-07-15-29-00_PNC_FCNet_normal_none_loss_none_none', type=str,
                        help='The path of the folder containing the model.')
    parser.add_argument('--config_filename', default='setting/pnc.yaml', type=str,
                        help='Configuration filename for training the model.')
    args = parser.parse_args()
    main(args)




























