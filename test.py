import os
import hydra
import torch
import utils.setup as setup

def _main(args):

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global __file__
    __file__ = hydra.utils.to_absolute_path(__file__)
    dirname = os.path.dirname(__file__)
    args.model_dir = os.path.join(dirname, str(args.model_dir))
    if not os.path.exists(args.model_dir):
            raise Exception(f"Model directory {args.model_dir} does not exist")

    args.exp.model_dir=args.model_dir

    diff_params=setup.setup_diff_parameters(args)
    network=setup.setup_network(args, device)

    try:
        test_set=setup.setup_dataset_test(args)
    except:
        test_set=None

    tester=setup.setup_tester(args, network=network, diff_params=diff_params, test_set=test_set, device=device) #this will be used for making demos during training
    # Print options.
    print()
    print('Training options:')
    print()
    print(f'Output directory:        {args.model_dir}')
    print(f'Network architecture:    {args.network.callable}')
    print(f'Diffusion parameterization:  {args.diff_params.callable}')
    print(f'Tester:                  {args.tester.callable}')
    print(f'Experiment:                  {args.exp.exp_name}')
    print()

    if args.tester.checkpoint != 'None':
        ckpt_path= args.tester.checkpoint
        print("Loading checkpoint:",ckpt_path)

        try:
            tester.load_checkpoint(ckpt_path) 
        except:
            #maybe it is a relative path
            #find my path
            path=os.path.dirname(__file__)
            print(path)
            tester.load_checkpoint(os.path.join(path,ckpt_path))
        tester.setup_sampler()
    else:
        print("trying to load latest checkpoint")
        tester.load_latest_checkpoint()

    tester.test()

@hydra.main(config_path="conf", config_name="conf")
def main(args):
    _main(args)

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
