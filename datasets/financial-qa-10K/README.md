---
dataset_info:
  features:
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: context
    dtype: string
  - name: ticker
    dtype: string
  - name: filing
    dtype: string
  splits:
  - name: train
    num_bytes: 3282240
    num_examples: 7000
  download_size: 1588233
  dataset_size: 3282240
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
