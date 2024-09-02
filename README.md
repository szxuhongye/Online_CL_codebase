# Online Continual learning Code base
Modified based on the: "[PCR: Proxy-based Contrastive Replay for Online Class-Incremental Continual Learning](https://arxiv.org/abs/2304.04408)"

The framework is based on [online-continual-learning](https://github.com/RaptorMai/online-continual-learning).
- CIFAR10 & CIFAR100 will be downloaded during the first run. (datasets/cifar10;/datasets/cifar100)
- Mini-ImageNet: Download from https://www.kaggle.com/whitemoon/miniimagenet/download, and place it in datasets/mini_imagenet/.

### CIFAR-100
```shell
python general_main.py --num_runs 10 --data cifar100 --cl_type nc --agent PCR  --retrieve random --update random --mem_size 10000 --epoch 100 --batch 10 --eps_mem_batch 10 --exp PCR_cifar100_epoch100_random_mem10000 
```

 ### Mini-Imagenet
```shell
python general_main.py --num_runs 10 --data mini_imagenet --cl_type nc --agent PCR  --retrieve random --update random --mem_size 10000 --epoch 100 --batch 10 --eps_mem_batch 10 --exp PCR_mini_epoch100_random_mem10000 
```

