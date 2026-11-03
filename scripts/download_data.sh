#!/bin/bash
# Downloads CIFAR-100 (official), STL-10 and Tiny-ImageNet (Hugging Face parquet mirrors) into ./data
# and converts STL-10 / Tiny-ImageNet to the uint8 tensors that train.py expects.
set -e
cd "$(dirname "$0")/.."
mkdir -p data/raw/stl10 data/raw/tiny
if [ ! -d data/cifar-100-python ]; then
  curl -L -o data/cifar-100-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
  tar -xzf data/cifar-100-python.tar.gz -C data
fi
HF=https://huggingface.co/datasets
curl -L -o data/raw/stl10/train.parquet $HF/tanganke/stl10/resolve/main/data/train-00000-of-00001.parquet
curl -L -o data/raw/stl10/test.parquet  $HF/tanganke/stl10/resolve/main/data/test-00000-of-00001.parquet
curl -L -o data/raw/tiny/train.parquet  $HF/zh-plus/tiny-imagenet/resolve/main/data/train-00000-of-00001-1359597a978bc4fa.parquet
curl -L -o data/raw/tiny/valid.parquet  $HF/zh-plus/tiny-imagenet/resolve/main/data/valid-00000-of-00001-70d52db3c749a935.parquet
python prepare_data.py data/raw data stl10 tiny
