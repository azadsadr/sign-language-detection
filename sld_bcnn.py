import tyxe
#import numpy as np
#import pandas as pd
import torch
import pyro
from torch import nn
from torch.utils.data import Dataset, DataLoader
import functools
#import torchvision
#from torchvision import datasets, transforms
#import matplotlib.pyplot as plt
import pyro.distributions as dist
from pyro.infer import SVI
import os

output_dir = '/home/thomas/Documents/repositories/sign-language-detection/output'

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print('Device: {}'.format(device))

activ = nn.LeakyReLU 
net = nn.Sequential(nn.Conv2d(in_channels=1,
                              out_channels=64,
                              kernel_size=5,
                              ),
                    activ(),
                    nn.MaxPool2d(2,2),
                    nn.Conv2d(in_channels=64,
                              out_channels=512,
                              kernel_size=3,
                              ),
                    activ(),
                    nn.MaxPool2d(2,2),
                    nn.Flatten(),
                    nn.Linear(512*5*5, 256),
                    activ(),
                    nn.Linear(256, 24)).to(device)

prior_kwargs = dict()#expose_all=False, hide_module_types=(nn.BatchNorm2d,))

likelihood = tyxe.likelihoods.Categorical(27455)

prior = tyxe.priors.IIDPrior(dist.Normal(torch.zeros(1, device=device), torch.ones(1, device=device)), **prior_kwargs)
                                 
guide = functools.partial(tyxe.guides.AutoNormal,
                                  init_loc_fn=tyxe.guides.PretrainedInitializer.from_net(net, prefix="net"), init_scale=1e-4,
                                  max_guide_scale=1)#, train_loc=not scale_only)
bnn = tyxe.VariationalBNN(net, prior, likelihood, guide)

pyro.clear_param_store()
bnn.net.load_state_dict(torch.load(os.path.join(output_dir, "state_dict.pt")))
pyro.get_param_store().load(os.path.join(output_dir, "param_store.pt"), map_location=device)

