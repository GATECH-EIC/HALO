# HALO: Hardware-Aware Learning to Optimize

Accepted at ECCV 2020 [[Paper](https://dl.acm.org/doi/abs/10.1007/978-3-030-58545-7_29) 

## Setup
```
conda env create -f environment.yml
```

## Step 1: Prepare pre-trained model

```
cd l2o-scale-regularize-test
python train_pretrain_lstm.py --use_second_derivatives=False --include_conv_lstm_problems --test_optimizer=Adam --custom_flag conv_lstm_3_layer_64 --num_testing_itrs 100 --lr 1e-3
```

## Step 2: Train optimizer

### Train H-optimizer
```
cd l2o-scale-regularize-train
python metarun_adapt.py --use_second_derivatives=False --train_dir=adapt_mnist_conv/log-scale-baseline --adapt_mnist_conv_problems --num_partial_unroll_itr_scale=1 --num_meta_iterations=50 --pretrained_model_path ../l2o-scale-regularize-test/records/pretrain/Adam_mnist-B/seed12_model_params.pickle

```

### Train HALO
```
cd l2o-scale-regularize-train
python metarun_adapt.py --use_second_derivatives=False --train_dir=adapt_mnist_conv/log-optimzier-jacob-a-5e-4 --adapt_mnist_conv_problems --num_partial_unroll_itr_scale=1 --num_meta_iterations=50 --regularize_time=none --alpha=5e-4 --reg_optimizer=True --reg_option=jacob --pretrained_model_path ../l2o-scale-regularize-test/records/pretrain/Adam_mnist-B/seed12_model_params.pickle
```

## Step 3: Test the adaption performance
### Adam optimizer
```
cd l2o-scale-regularize-test
python metatest_adapt_lstm.py \
--use_second_derivatives=False \
--test_optimizer=Adam \
--adapt_conv_lstm \
--pretrained_model_path ../l2o-scale-regularize-test/records/pretrain/Adam_lr_0.001_conv_lstm_3_layer_64/seed0_model_params_acc-0.7528409090909091.pickle \
--random_sparse_method layer_wise \
--random_sparse_prob "1.0" \
--batch_size 64 \
--save_dir records_lstm \
--lr 1e-3
```
### Adam optimizer (train from scratch)
```
cd l2o-scale-regularize-test
python metatest_adapt_lstm.py \
--train_dir=adapt_S/ \
--use_second_derivatives=False \
--adapt_conv_lstm \
--test_optimizer=Adam \
--random_sparse_method layer_wise \
--random_sparse_prob "1.0" \
--batch_size 64 \
--save_dir records_lstm \
--lr 1e-3
```

### Adagrad optimizer
```
cd l2o-scale-regularize-test
python metatest_adapt_lstm.py \
--use_second_derivatives=False \
--test_optimizer=Adagrad \
--adapt_conv_lstm \
--pretrained_model_path ../l2o-scale-regularize-test/records/pretrain/Adam_lr_0.001_conv_lstm_3_layer_64/seed0_model_params_acc-0.7528409090909091.pickle \
--random_sparse_method layer_wise \
--random_sparse_prob "1.0" \
--batch_size 64 \
--save_dir records_lstm \
--lr 1e-3
```
### SGD optimizer
```
cd l2o-scale-regularize-test
python metatest_adapt_lstm.py \
--use_second_derivatives=False \
--test_optimizer=SGD \
--adapt_conv_lstm \
--pretrained_model_path ../L2O-Adaptation/l2o-scale-regularize-test/records/pretrain/Adam_lr_0.001_conv_lstm_3_layer_64/seed0_model_params_acc-0.7528409090909091.pickle \
--random_sparse_method layer_wise \
--random_sparse_prob "1.0" \
--batch_size 64 \
--save_dir records_lstm \
--lr 1e-3
```

### H-optimizer
```
cd l2o-scale-regularize-test
python metatest_adapt_lstm.py \
--use_second_derivatives=False \
--train_dir=../l2o-scale-regularize-train/adapt_mnist_conv/log-scale-baseline \
--adapt_conv_lstm \
--pretrained_model_path ../l2o-scale-regularize-test/records/pretrain/Adam_lr_0.001_conv_lstm_3_layer_64/seed0_model_params_acc-0.7528409090909091.pickle \
--random_sparse_method layer_wise \
--random_sparse_prob "1.0" \
--batch_size 64
```

### HALO (random update)
```
cd l2o-scale-regularize-test
python metatest_adapt_lstm.py \
--use_second_derivatives=False \
--train_dir=../l2o-scale-regularize-train/adapt_mnist_conv/log-optimzier-jacob-a-5e-4 \
--adapt_conv_lstm \
--pretrained_model_path ../l2o-scale-regularize-test/records/pretrain/Adam_lr_0.001_conv_lstm_3_layer_64/seed0_model_params_acc-0.7528409090909091.pickle \
--random_sparse_method layer_wise \
--random_sparse_prob "0.1 0.3 0.5" \
--batch_size 64
```

### HALO
```
cd l2o-scale-regularize-test
python metatest_adapt_lstm.py \
--use_second_derivatives=False \
--train_dir=../l2o-scale-regularize-train/adapt_mnist_conv/log-optimzier-jacob-a-5e-4 \
--adapt_conv_lstm \
--pretrained_model_path ../l2o-scale-regularize-test/records/pretrain/Adam_lr_0.001_conv_lstm_3_layer_64/seed0_model_params_acc-0.7528409090909091.pickle \
--random_sparse_method layer_wise \
--random_sparse_prob "1.0" \
--batch_size 64
```

## Citation

```
@inproceedings{li2020halo,
  title={HALO: Hardware-aware learning to optimize},
  author={Li, Chaojian and Chen, Tianlong and You, Haoran and Wang, Zhangyang and Lin, Yingyan},
  booktitle={Computer Vision--ECCV 2020: 16th European Conference, Glasgow, UK, August 23--28, 2020, Proceedings, Part IX},
  pages={500--518},
  year={2020},
  organization={Springer}
}
```

## License

Copyright (c) 2022 GaTech-EIC. All rights reserved.
Licensed under the [MIT](https://github.com/GATECH-EIC/HALO/blob/master/LICENSE) license.