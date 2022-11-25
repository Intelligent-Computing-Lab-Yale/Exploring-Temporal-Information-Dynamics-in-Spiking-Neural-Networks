# Understanding Temporal Information Dynamics in Spiking Neural Networks

Pytorch code for [Understanding Temporal Information Dynamics in Spiking Neural Networks] 

## Dependencies
* Python 3.9    
* PyTorch 1.10.0   
* Spikingjelly
```
git clone https://github.com/fangwei123456/spikingjelly.git
cd spikingjelly
python setup.py install
```       

## Training and Computing fisher information

In this anonymous code, we provide a code for 

  (a) train_snn.py:  train SNN from scratch 
    
```
python train_snn.py  --dataset 'cifar10' --arch 'resnet19' --optimizer 'sgd' --batch_size 128 --learning_rate 3e-1 --timestep 10
```
  
  (b) train_snn_fisherinfo.py: computing fisher information from pretrained model
    
```
python train_snn_fisherinfo.py --dataset 'cifar10' --arch 'resnet19'  --batch_size 16 --timestep 10
```

Also, for skipping (a) Train SNN from scratch, we provide pretrained parameters ([link][e]) for ResNet19_CIFAR10 from epoch 20, 120, 300.

[e]: https://drive.google.com/drive/folders/1X3nhax10zSZXVVLAYxZtOtEB8paTEDsi?usp=sharing

Download three check point under ``snapshots/`` folder


