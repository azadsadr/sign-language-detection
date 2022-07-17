# Sign Language Detection

**Authors:** *Azad Sadr*, *Lorenzo Taroni*, *Thomas Axel Deponte*  
**Date:** *6 July 2022*



**Table of contents**  
<<<<<<< HEAD
- [Problem Statement](#problem-statement)
- [Data](#data)
- [Methodology](#methodology)
- [Results](#results)
- [References](#references)
=======
- [Sign Language Detection](#sign-language-detection)
  - [Install and run the code](#install-and-run-the-code)
    - [prepare the environment (miniconda)](#prepare-the-environment-miniconda)
  - [Problem Statement](#problem-statement)
  - [Data](#data)
  - [Methodology](#methodology)
  - [Conclusion](#conclusion)
  - [References](#references)
>>>>>>> 18fdcc928d56da3c5bff7afbc2385058173120c9

## Install and run the code  
  
### prepare the environment (miniconda)  
  
Assume miniconda is already installed and you already execute `$ conda config --set env_prompt '({name})'`  

```bash
source ~/miniconda3/bin/activate   # entering in base env
(miniconda3)$ cd ~/project_repository_folder     # go to the project folder
(miniconda3)$ conda create --prefix ./env -f environment.txt       # create a new env in a different location
(miniconda3)$ conda activate ./env               # activate the new env

(env)$ conda install ipykernel                          # install a jupyter python kernel
(env)$ python -m ipykernel install --user --name 'sld'  # tell to jupyter where the new kernel is located
(env)$ conda deactivate # exit the local environment
(miniconda3)$ jupyter kernelspec install PREFIX --sys-prefix   # tell to jupyter where the new kernel is located, the prefix is returned by "python -m ipykernel install ..."  
```

## Problem Statement

Communication is an important part of our lives. Deaf and dumb people being unable to speak and listen, experience a lot of problems while communicating with normal people. There are many ways by which people with these disabilities try to communicate. One of the most prominent ways is the use of sign language, i.e. hand gestures. It is necessary to develop an application for recognizing gestures and actions of sign language so that deaf and dumb people can communicate easily with even those who don’t understand sign language. The objective of this work is to take an elementary step in breaking the barrier in communication between the normal people and deaf and dumb people with the help of sign language.

American Sign Language (ASL) is a complete, natural language that has the same linguistic properties as spoken languages, with grammar that differs from English. ASL is expressed by movements of the hands and face. It is the primary language of many North Americans who are deaf and hard of hearing, and is used by many hearing people as well.

## Data

Dataset are from Kaggle website include, 27000 images with $28 \times 18$ resolution in `csv` format.

## Methodology

Here, we are trying to develop a deep learning model to recognize the hand gestures and map them to the English letter.

* Convolutional Neural Network
* Bayesian Convolutional Neural Network

### Convolutional Neural Network

![](images/arch(3).drawio.png)

## Results



## References
<a id="1">https://www.kaggle.com/code/sanikamal/sign-language-detection/notebook</a> 