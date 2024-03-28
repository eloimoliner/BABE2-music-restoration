import os
import re
import json
import hydra
import torch
#from utils.torch_utils import distributed as dist
import utils.setup as setup

import copy


def _main(args):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #assert torch.cuda.is_available()
    #device="cuda"

    global __file__
    __file__ = hydra.utils.to_absolute_path(__file__)
    dirname = os.path.dirname(__file__)
    args.model_dir = os.path.join(dirname, str(args.model_dir))
    if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)


    args.exp.model_dir=args.model_dir

    #dist.init()
    dset=setup.setup_dataset(args)
    diff_params=setup.setup_diff_parameters(args)
    network=setup.setup_network(args, device)
    optimizer=setup.setup_optimizer(args, network)

    network_tester=copy.deepcopy(network)

    tester=setup.setup_tester(args, network=network_tester, diff_params=diff_params,  device=device) #this will be used for making demos during training
    print("setting up trainer")
    trainer=setup.setup_trainer(args, dset=dset, network=network, optimizer=optimizer, diff_params=diff_params, tester=tester, device=device) #this will be used for making demos during training
    print("trainer set up")


    # Print options.
    print()
    print('Training options:')
    print()
    print(f'Output directory:        {args.model_dir}')
    print(f'Network architecture:    {args.network.callable}')
    print(f'Dataset:    {args.dset.callable}')
    print(f'Diffusion parameterization:  {args.diff_params.callable}')
    print(f'Batch size:              {args.exp.batch}')
    print()

    # Train.
    trainer.training_loop()

@hydra.main(config_path="conf", config_name="conf_piano")
def main(args):
    _main(args)

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
