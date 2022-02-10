import deepdish as dd
import os.path as osp
import os
import numpy as np
import argparse
from pathlib import Path


def main(args):
    data_dir =  os.path.join(args.root_path, 'ABIDE_pcp/cpac/filt_noglobal/raw')
    timeseires = os.path.join(args.root_path, 'ABIDE_pcp/cpac/filt_noglobal/')

    times = []

    labels = []
    pcorrs = []

    corrs = []

    for f in os.listdir(data_dir):
        if osp.isfile(osp.join(data_dir, f)):
            fname = f.split('.')[0]

            files = os.listdir(osp.join(timeseires, fname))

            file = list(filter(lambda x: x.endswith("1D"), files))[0]

            time = np.loadtxt(osp.join(timeseires, fname, file), skiprows=0).T

            if time.shape[1] < 100:
                continue

            temp = dd.io.load(osp.join(data_dir,  f))
            pcorr = temp['pcorr'][()]

            pcorr[pcorr == float('inf')] = 0

            att = temp['corr'][()]

            att[att == float('inf')] = 0

            label = temp['label']

            times.append(time[:,:100])
            labels.append(label[0])
            corrs.append(att)
            pcorrs.append(pcorr)

    np.save(Path(args.root_path)/'ABIDE_pcp/abide.npy', {'timeseires': np.array(times), "label": np.array(labels),"corr": np.array(corrs),"pcorr": np.array(pcorrs)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate the final dataset')
    parser.add_argument('--root_path', default="", type=str, help='The path of the folder containing the dataset folder.')
    args = parser.parse_args()
    main(args)
