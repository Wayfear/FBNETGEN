from pathlib import Path
import argparse
import yaml
import torch

from model import FBNETGEN, GNNPredictor, SeqenceModel, BrainNetCNN
from train import BasicTrain, BiLevelTrain, SeqTrain, GNNTrain, BrainCNNTrain
from datetime import datetime
from dataloader import init_dataloader



def main(args):
    with open(args.config_filename) as f:
        config = yaml.load(f, Loader=yaml.Loader)

        dataloaders, node_size, node_feature_size, timeseries_size = \
            init_dataloader(config['data'])

        config['train']["seq_len"] = timeseries_size
        config['train']["node_size"] = node_size

        if config['model']['type'] == 'seq':
            model = SeqenceModel(config['model'], node_size, timeseries_size)
            use_train = SeqTrain

        elif config['model']['type'] == 'gnn':
            model = GNNPredictor(node_feature_size, node_size)
            use_train = GNNTrain

        elif config['model']['type'] == 'fbnetgen':
            model = FBNETGEN(config['model'], node_size,
                             node_feature_size, timeseries_size)
            use_train = BasicTrain

        elif config['model']['type'] == 'brainnetcnn':

            model = BrainNetCNN(node_size)

            use_train = BrainCNNTrain


        if config['train']['method'] == 'bilevel' and \
                config['model']['type'] == 'fbnetgen':
            parameters = {
                'lr': config['train']['lr'],
                'weight_decay': config['train']['weight_decay'],
                'params': [
                    {'params': model.extract.parameters()},
                    {'params': model.emb2graph.parameters()}
                ]
            }

            optimizer1 = torch.optim.Adam(**parameters)

            optimizer2 = torch.optim.Adam(model.predictor.parameters(),
                                          lr=config['train']['lr'],
                                          weight_decay=config['train']['weight_decay'])
            opts = (optimizer1, optimizer2)
            use_train = BiLevelTrain

        else:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=config['train']['lr'],
                weight_decay=config['train']['weight_decay'])
            opts = (optimizer,)

        loss_name = 'loss'
        if config['train']["group_loss"]:
            loss_name = f"{loss_name}_group_loss"
        if config['train']["sparsity_loss"]:
            loss_name = f"{loss_name}_sparsity_loss"

        now = datetime.now()

        date_time = now.strftime("%m-%d-%H-%M-%S")

        extractor_type = config['model']['extractor_type'] if 'extractor_type' in config['model'] else "none"
        embedding_size = config['model']['embedding_size'] if 'embedding_size' in config['model'] else "none"
        window_size = config['model']['window_size'] if 'window_size' in config['model'] else "none"

        save_folder_name = Path(config['train']['log_folder'])/Path(
            date_time +
            f"_{config['data']['dataset']}_{config['model']['type']}_{config['train']['method']}" 
            + f"_{extractor_type}_{loss_name}_{embedding_size}_{window_size}")

        train_process = use_train(
            config['train'], model, opts, dataloaders, save_folder_name)

        train_process.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='setting/pnc.yaml', type=str,
                        help='Configuration filename for training the model.')
    parser.add_argument('--repeat_time', default=5, type=int)
    args = parser.parse_args()
    for i in range(args.repeat_time):
        main(args)
