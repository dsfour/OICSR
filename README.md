# OICSR Pruned Models
This repo contains some of the pruned models from paper [**OICSR: Out-In-Channel Sparsity Regularization for Compact Deep Neural Networks (CVPR 2019)**](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_OICSR_Out-In-Channel_Sparsity_Regularization_for_Compact_Deep_Neural_Networks_CVPR_2019_paper.pdf).
## Reference
If you find the models useful, please kindly cite our paper:  
```
@inproceedings{li2019oicsr,
  title={OICSR: Out-In-Channel Sparsity Regularization for Compact Deep Neural Networks},
  author={Li, Jiashi and Qi, Qi and Wang, Jingyu and Ge, Ce and Li, Yujian and Yue, Zhangzhang and Sun, Haifeng},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={7046--7055},
  year={2019}
}
```
## Download the pruned models
Download the pretrained models from [here](https://drive.google.com/drive/u/0/folders/10s98eW_25-xEHnpGsC3PsSU7PH5z681B) and put it in ./checkpoints.  
## Models  
### Pruned ResNet-50  
We provide ResNet-50 model with various FLOPs pruned percents. The channel pruning results are showed as follows:  

|Models|Top1 Acc (%)|Drop Top1 Acc (%)|Top5 Acc (%)|Drop Top5 Acc (%)|FLOPs (M)|  
|:-:|:-:|:-:|:-:|:-:|:-:|  
|resnet50|76.32|0.00|93.00|0.00|4089|  
|resnet50-37.3%FLOPs|76.53|-0.21|93.16|-0.16|2563|  
|resnet50-44.4%FLOPs|76.30|0.02|92.92|0.08|2274|  
|resnet50-50.0%FLOPs|75.95|0.37|92.66|0.34|2046|  

To test the model, run:  
`python eval_prune_model.py --test_data /mnt/cephfs_wj/cv/common/datasets/ImageNet/ILSVRC2012_img_val --fpp 50.0`

## Contact
To contact the author:  
Jiashi Li, lijiashi@bupt.edu.cn



