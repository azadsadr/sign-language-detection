import torch
import pyro
import pyro.distributions as dist
import functools
import os
import tyxe
import network

def CNN():
    path = "output/cnn/model.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = network.network(in_channels=1, output_size=24, device=device)

    # load the pretrained model
    net.load_state_dict(torch.load(path))

    return net


def BCNN():
    path = "output/bcnn"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = network.network(in_channels=1, output_size=24, device=device)

    # load the pretrained model
    prior_kwargs = dict() # expose_all=False, hide_module_types=(nn.BatchNorm2d,))
    likelihood = tyxe.likelihoods.Categorical(27455)
    prior = tyxe.priors.IIDPrior(
        dist.Normal(torch.zeros(1, device=device), torch.ones(1, device=device)), 
        **prior_kwargs
        )
    guide = functools.partial(
        tyxe.guides.AutoNormal,
        init_loc_fn=tyxe.guides.PretrainedInitializer.from_net(net, prefix="net"), 
        init_scale=1e-4,
        max_guide_scale=1
        ) #, train_loc=not scale_only)
    bnn = tyxe.VariationalBNN(net, prior, likelihood, guide)

    pyro.clear_param_store()
    bnn.net.load_state_dict(torch.load(os.path.join(path, "state_dict.pt")))
    pyro.get_param_store().load(os.path.join(path, "param_store.pt"), map_location=device)
    return bnn
