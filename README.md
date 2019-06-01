# OICSR Compressed Models
This repo contains some of the compressed models from paper [**OICSR: Out-In-Channel Sparsity Regularization for Compact Deep Neural Networks (CVPR 2019)**](https://arxiv.org/abs/1905.11664).
## Reference
If you find the models useful, please kindly cite our paper:  
```
@article{li2019oicsr,
  title={OICSR: Out-In-Channel Sparsity Regularization for Compact Deep Neural Networks},
  author={Li, Jiashi and Qi, Qi and Wang, Jingyu and Ge, Ce and Li, Yujian and Yue, Zhangzhang and Sun, Haifeng},
  journal={arXiv preprint arXiv:1905.11664},
  year={2019}
}
```
## Models  
### Pruned ResNet-50  
We provide ResNet-50 model with various pruned FLOPs pruned percents. The channel pruning results are showed as follows:  
Models|Top1 Acc (%)|Drop Top1 Acc (%)|Top5 Acc (%)|Drop Top5 Acc (%)|FLOPs (M)  
-|-|-|-|-|-  
resnet50|76.32|0.00|93.00|0.00|4089  
resnet50-37.3%FLOPs|76.53|-0.21|93.16|-0.16|2563  
resnet50-44.4%FLOPs|76.30|0.02|92.92|0.08|2274  
resnet50-50.0%FLOPs|75.95|0.37|92.66|0.34|2046  


To test the model, run:  
`python eval_prune_model.py --fpp 50.0`

## Contact
To contact the author:  
Jiashi Li, lijiashi@bupt.edu.cn
