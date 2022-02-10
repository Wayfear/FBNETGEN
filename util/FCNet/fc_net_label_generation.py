from sklearn.cluster import AffinityPropagation
import numpy as np
import argparse
import random
import pathlib


def main(args):
    final_fc = np.load(args.data_path, allow_pickle=True)

    if args.dataset == 'PNC':
        final_fc = final_fc.item()
        final_fc = final_fc['data']

    column_idxs = []

    labels = []

    for p, fc in enumerate(final_fc):
        print(p)
        cluster_result = AffinityPropagation().fit(fc).labels_
        node_size = fc.shape[0]


        for i in range(node_size):
            count = 0
            for j in range(i, node_size):
                if i == j:
                    continue
                if cluster_result[i] == cluster_result[j]:
                    column_idxs.append((np.array((p, i, j))))
                    labels.append(1)
                    count += 1

            while count:
                t = random.randint(0, node_size-1)
                if t != i and cluster_result[i] == cluster_result[t]:
                    column_idxs.append((np.array((p, i,t))))
                    labels.append(0)
                    count -= 1

    column_idxs = np.array(column_idxs)
    labels = np.array(labels)

    print(f"Sample size: {labels.shape[0]}, positive: {np.sum(labels)}, negative: {labels.shape[0]-np.sum(labels)}")

    parent_path = pathlib.Path(args.data_path).parent

    np.save(parent_path/'fcnet_training_data.npy', {'index': column_idxs, 'label': labels})





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/home/root/dataset/ABCD/abcd_rest-timeseires-HCP2016.npy', type=str,
                        help='Configuration filename for training the model.')
    parser.add_argument('--dataset', default='ABCD', type=str,
                        help='Configuration filename for training the model.')
    args = parser.parse_args()
    main(args)