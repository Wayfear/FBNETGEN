# FCNet

FCNet (FCNet: A Convolutional Neural Network for Calculating Functional Connectivity from Functional MRI) is the first paper using neural networks to generate functional connectivity matrices. Since they don't provide the source code, we implemented it and also open-sourced it here. FCNet is a pipeline containing several steps, so running it needs several steps. 

Besides, for ABCD, the dataset generation method in FCNet would generate too many samples (The sample size is 78,208,924) for training. Therefore, we only use PNC to test the performance of this method.

## Usage

```bash

python util/FCNet/fc_net_label_generation.py --data_path path/to/your/dataset --dataset datasetname

# Generate correlation matrices.
python main.py --config_filename pnc_fcnet.yaml

# Generate the final dataset.
python util/FCNet/infer.py 
```


