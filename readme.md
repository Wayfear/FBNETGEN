# FBNETGEN

## Dataset

### PNC and ABCD

PNC can be accessed from [NIH](https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs000607.v3.p2), and ABCD can be accessed from [NIMH Data Archive](https://nda.nih.gov/).


### ABIDE

For those who can not access ABCD and PNC dataset, we also provide an open-source dataset, ABIDE. Please follow the [instruction](util/abide/readme.md) to download and process this dataset.

## Usage

### PNC

```bash
python main.py --config_filename setting/pnc_fbnetgen.yaml
```

### ABCD 

```bash
python main.py --config_filename setting/abcd_fbnetgen.yaml
```

### ABIDE 

```bash
python main.py --config_filename setting/abide_fbnetgen.yaml
```

## Hyper parameters

All hyper parameters can be tuned in setting files.

```yaml
model:
  # For the model type, there are 3 choices: "seq", "gnn" or "fbnetgen". 
  type: fbnetgen

  # For the feature extractor, there are 2 choices: "gru" or "cnn".
  extractor_type: gru

  # For the feature extractor, there are 2 choices: "product" or "linear". 
  # We suggest to use "product", since it is faster.
  graph_generation: product

  # Two hyperparameters are tuned in our paper.
  embedding_size: 8
  window_size: 8



train:
  # For training method, there are 2 choices: "normal" or "bilevel".
  # "bilevel" will be in effect only if the model.type is set as "fbnetgen"
  # We suggest to use "normal".
  method: normal
  
  # If the model.type is set as "gnn", this hyper parameter will be in effect.
  # There are 2 choices: "uniform" or "pearson".
  pure_gnn_graph: pearson
```

## Performance

