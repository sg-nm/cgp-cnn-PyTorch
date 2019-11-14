# KAIST CS454 Project - Group 7
KAIST CS454 project originated from cgp-cnn-PyTorch

## Requirement

* Ubuntu 16.04.6 LTS
* Python version 3.6.2
* PyTorch version 0.4.1
* CUDA version 10.0

# cgp-cnn-PyTorch
A Genetic Programming Approach to Designing CNN Architectures, In GECCO 2017 (oral presentation, Best Paper Award)

# Designing Convolutional Neural Network Architectures Based on Cartegian Genetic Programming

This repository contains the code for the following paper:

Masanori Suganuma, Shinichi Shirakawa, and Tomoharu Nagao, "A Genetic Programming Approach to Designing Convolutional Neural Network Architectures," 
Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '17, Best paper award), pp. 497-504 (2017) [[paper]](https://doi.org/10.1145/3071178.3071229) [[arXiv]](https://arxiv.org/abs/1704.00764)

## Requirement
We use the [PyTorch](https://pytorch.org/) framework for neural networks and tested on the following environment:

* PyTorch version 0.2.0_4
* Python version 3.6.2
* CUDA version 8.0
* Ubuntu 14.04 LTS

## Usage

### Run the architecture search
This code can reproduce the experiment for CIFAR-10 dataset with the same setting of the GECCO 2017 paper (by default scenario). The (training) data are split into the training and validation data. The validation data are used for assigning the fitness to the generated architectures.

When you use the multiple GPUs, please specify the `-g` option:

```shell
python exp_main.py -g 2
```

After the execution, the files, `network_info.pickle` and `log_cgp.txt` will be generated. The file `network_info.pickle` contains the information for Cartegian genetic programming (CGP) and `log_cgp.txt` contains the log of the optimization and discovered CNN architecture's genotype lists.

Some parameters (e.g., # rows and columns of CGP, and # epochs) can easily change by modifying the arguments in the script `exp_main.py`.

### Re-training

The discovered architecture is re-trained by the different training scheme (500 epoch training with momentum SGD) to polish up the network parameters. All training data are used for re-training, and the accuracy for the test data set is reported.

```shell
python exp_main.py -m retrain
```
