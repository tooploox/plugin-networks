# Plugin-Networks
Plugin-Network framework is a code that enables adding partial evidence 
(conditional information) to already trained neural networks. More details
are available in the [paper](https://arxiv.org/abs/1901.00326).

## Setting up the environment
Currently, we provide a working example on split1 of 
[SUN397](https://groups.csail.mit.edu/vision/SUN/) dataset.
Please run `download_environment.sh`. This script will download trained
base model and dataset. In the result you should get `workspace` directory
with the following structure:
```
sun397/
├── base_models
│   └── sun397_base_model_split01.pth.tar
├── hierarchy.csv
├── images -> /home/user/Datasets/SUN397/
│   ├── a
│   ├── b
│   ├── c
│   ├── ClassName.txt
│   ├── d
│   ├── e
│   ├── f
│   ├── g
│   ├── h
│   ├── i
│   ├── j
│   ├── k
│   ├── l
│   ├── m
│   ├── n
│   ├── o
│   ├── p
│   ├── r
│   ├── README.txt
│   ├── s
│   ├── t
│   ├── u
│   ├── v
│   ├── w
│   └── y
└── partitions
    ├── ClassName.txt
    ├── Testing_01.txt
    ├── Testing_02.txt
    ├── Testing_03.txt
    ├── Testing_04.txt
    ├── Testing_05.txt
    ├── Testing_06.txt
    ├── Testing_07.txt
    ├── Testing_08.txt
    ├── Testing_09.txt
    ├── Testing_10.txt
    ├── Training_01.txt
    ├── Training_02.txt
    ├── Training_03.txt
    ├── Training_04.txt
    ├── Training_05.txt
    ├── Training_06.txt
    ├── Training_07.txt
    ├── Training_08.txt
    ├── Training_09.txt
    └── Training_10.txt
``` 

## Running the experiment
The best idea is to add the code directory to `PYTHONPATH` 
eg. `export PYTHONPATH=plugin-networks:$PYTHONPATH`, as well as update `PATH`:
`export PATH=plugin-networks/pluginnet:$PATH`.

Now in `workspace/sun397` type `train.py ../../code_release/confs/SUN397/conf_pe.json output`
This runs the training script. The results will be saved in the `output` dir.
Also `runs` directory will be created with TensorBoard output.

The experiment uses `conf_pe.json` as a configuration file. The structure of the file
is explained in the next section.

## Config JSON file
Each experiment is described bu JSON file. The example file is located in 
`plugin-networks/confs/SUN397/conf_pe.json`. 
The file has the following structure:

```
{
   "task": "sun397_pe",
   "dataset_root": "images",
   "split_file_train": "partitions/Training_01.txt",
   "split_file_test": "partitions/Testing_01.txt",
   "hierarchy_file": "hierarchy.csv",
   "base_model_file": "base_models/sun397_base_model_split01.pth.tar",
   "criterion": "cross_entropy",
   "epochs": 15,
   "lr": 0.001,
   "lr_decay": 0.1,
   "lr_step": 5,
   "momentum": 0.9,
   "clip_gradient": -1,
   "weight_decay": 5e-4,
   "plugins": {
       "conv1": [3, 256, 256, 256, 64],
       "conv2": [3, 256, 256, 256, 192],
       "conv3": [3, 256, 256, 256, 384],
       "conv4": [3, 256, 256, 256, 256],
       "conv5": [3, 256, 256, 256, 256],
      "linear1": [3, 256, 256, 256, 4096],
      "linear2": [3, 256, 256, 256, 4096],
      "linear3": [3, 256, 256, 256, 397]
   },
   "tag": "linear3_p10_split01",
   "seed": 0
}
```

Below the JSON file keys are explained:
* `dataset_root` - path to the directory where dataset images are stored
* `split_file_train` - path to file with trainset file list
* `split_file_test` - path to file with testset file list
* `hierarchy_file` - path to file with labels hierarchy (SUN397)
* `base_model_file` - path to base model file 
* `plugins` - contains key-value pair of plugins which will be fused to the
base model. The key is the layer name and value is a list that 
defines plugin network architecture. Each entry in the list defines a number of
neurons in the plugin network layer. Note that first entry should be equal to
plugin network input, while last entry should be equal to the output size of 
base network layer or double of output size if an affine fusion operator is used.

## How to use Plugin Networks framework with custom architecture
If you have your pretrained model with your architecture that is not yet
supported by the current version of code. You can easily extend it.

1. Use function `build_plugins` from `pluginnet/common/model.py`. It takes
a plugin definition dictionary as an input as described in the above section.
The output is a list of `nn.Modules` which are plugin networks.

2. Use function `operator_factory` from `pluginnet/common/model.py` to create
fusion operator.

3. Modify your base network in `forward` method that 
plugin network input is passed to the network:
```python
if layer_name in self.plugins_dict.keys():
    plugin_layer = self.plugins_dict[layer_name]
    plugin_output = plugin_layer(partial_evidence)
```
And then fuse base network layer output with plugin network output:
```python
def forward(self, x_in):
    ...
    x = layer_output
    x = self.operator(x, plugin_output)
    ...
    return x
```
The implemented example is available in `pluginnet/sun397/partial_evidence.py`
class `AlexNetPartialEvidence`.

## Citation
If you found this paper and code useful please cite:
```
@InProceedings{Koperski_2020_WACV,
author = {Koperski, Michal and Konopczynski, Tomasz and Nowak, Rafal and Semberecki, Piotr and Trzcinski, Tomasz},
title = {Plugin Networks for Inference under Partial Evidence},
booktitle = {The IEEE Winter Conference on Applications of Computer Vision (WACV)},
month = {March},
year = {2020}
}
```
