
import numpy as np
import torch
import torch.utils.data as utils
from sklearn import preprocessing
import pandas as pd
from scipy.io import loadmat
import pathlib

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def infer_dataloader(dataset_config):

    label_df = pd.read_csv(dataset_config["label"])


    if dataset_config["dataset"] == "PNC":
        fc_data = np.load(dataset_config["time_seires"], allow_pickle=True).item()
        fc_timeseires = fc_data['data'].transpose((0, 2, 1))

        fc_id = fc_data['id']
    

        id2gender = dict(zip(label_df['SUBJID'], label_df['sex']))

        final_fc, final_label = [], []

        for fc, l in zip(fc_timeseires, fc_id):
            if l in id2gender:
                final_fc.append(fc)
                final_label.append(id2gender[l])
        final_fc = np.array(final_fc)


    elif dataset_config["dataset"] == 'ABCD':

        fc_data = np.load(dataset_config["time_seires"], allow_pickle=True)

    _, node_size, timeseries = final_fc.shape

    encoder = preprocessing.LabelEncoder()

    encoder.fit(label_df["sex"])

    labels = encoder.transform(final_label)

    final_fc = torch.from_numpy(final_fc).float()

    return final_fc, labels, node_size, timeseries


        
def init_dataloader(dataset_config):

    if dataset_config["dataset"] == 'ABIDE':

        data = np.load(dataset_config["time_seires"], allow_pickle=True).item()
        final_fc = data["timeseires"]
        final_pearson = data["corr"]
        labels = data["label"]

    elif dataset_config["dataset"] == "HIV" or dataset_config["dataset"] == "BP":
        data = loadmat(dataset_config["node_feature"])

        labels = data['label']
        labels = labels.reshape(labels.shape[0])

        labels[labels==-1] = 0

        view = dataset_config["view"]

        final_pearson = data[view]

        final_pearson = np.array(final_pearson).transpose(2, 0, 1)

        final_fc = np.ones((final_pearson.shape[0],1,1))

    elif dataset_config["dataset"] == 'PPMI' or dataset_config["dataset"] == 'PPMI_balanced':
        m = loadmat(dataset_config["node_feature"])
        labels = m['label'] if dataset_config["dataset"] != 'PPMI_balanced' else m['label_new']
        labels = labels.reshape(labels.shape[0])
        data = m['X'] if dataset_config["dataset"] == 'PPMI' else m['X_new']
        final_pearson = np.zeros((data.shape[0], 84, 84))
        modal_index = 0
        for (index, sample) in enumerate(data):
            # Assign the first view in the three views of PPMI to a1
            final_pearson[index, :, :] = sample[0][:, :, modal_index]

        final_fc = np.ones((final_pearson.shape[0],1,1))

    else:

        fc_data = np.load(dataset_config["time_seires"], allow_pickle=True)
        pearson_data = np.load(dataset_config["node_feature"], allow_pickle=True)
        label_df = pd.read_csv(dataset_config["label"])

        if dataset_config["dataset"] == 'ABCD':

            with open(dataset_config["node_id"], 'r') as f:
                lines = f.readlines()
                pearson_id = [line[:-1] for line in lines]

            with open(dataset_config["seires_id"], 'r') as f:
                lines = f.readlines()
                fc_id = [line[:-1] for line in lines]

            id2pearson = dict(zip(pearson_id, pearson_data))

            id2gender = dict(zip(label_df['id'], label_df['sex']))

            final_fc, final_label, final_pearson = [], [], []

            for fc, l in zip(fc_data, fc_id):
                if l in id2gender and l in id2pearson:
                    if np.any(np.isnan(id2pearson[l])) == False:
                        final_fc.append(fc)
                        final_label.append(id2gender[l])
                        final_pearson.append(id2pearson[l])

            final_pearson = np.array(final_pearson)

            final_fc = np.array(final_fc)

        elif dataset_config["dataset"] == "PNC":
            pearson_data, fc_data = pearson_data.item(), fc_data.item()

            pearson_id = pearson_data['id']
            pearson_data = pearson_data['data']
            id2pearson = dict(zip(pearson_id, pearson_data))

            fc_id = fc_data['id']
            fc_data = fc_data['data']

            id2gender = dict(zip(label_df['SUBJID'], label_df['sex']))

            final_fc, final_label, final_pearson = [], [], []

            for fc, l in zip(fc_data, fc_id):
                if l in id2gender and l in id2pearson:
                    final_fc.append(fc)
                    final_label.append(id2gender[l])
                    final_pearson.append(id2pearson[l])

            final_pearson = np.array(final_pearson)

            final_fc = np.array(final_fc).transpose(0, 2, 1)

    _, _, timeseries = final_fc.shape

    _, node_size, node_feature_size = final_pearson.shape

    scaler = StandardScaler(mean=np.mean(
        final_fc), std=np.std(final_fc))
    
    final_fc = scaler.transform(final_fc)

    if dataset_config["dataset"] == 'PNC' or dataset_config["dataset"] == 'ABCD':

        encoder = preprocessing.LabelEncoder()

        encoder.fit(label_df["sex"])

        labels = encoder.transform(final_label)

    final_fc, final_pearson, labels = [torch.from_numpy(
        data).float() for data in (final_fc, final_pearson, labels)]

    length = final_fc.shape[0]
    train_length = int(length*dataset_config["train_set"])
    val_length = int(length*dataset_config["val_set"])

    dataset = utils.TensorDataset(
        final_fc,
        final_pearson,
        labels
    )

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_length, val_length, length-train_length-val_length])

    train_dataloader = utils.DataLoader(
        train_dataset, batch_size=dataset_config["batch_size"], shuffle=True, drop_last=False)

    val_dataloader = utils.DataLoader(
        val_dataset, batch_size=dataset_config["batch_size"], shuffle=True, drop_last=False)

    test_dataloader = utils.DataLoader(
        test_dataset, batch_size=dataset_config["batch_size"], shuffle=True, drop_last=False)

    return (train_dataloader, val_dataloader, test_dataloader), node_size, node_feature_size, timeseries
