import torch
import argparse

from sdf.utils import *
from sdf.provider import SampleBox

def scale_fn(i):
    return 0.99 ** i

def scale_fn(i):
    return 0.99 ** i

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4, help="initial learning rate")
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")

    opt = parser.parse_args()
    print(opt)

    seed_everything(opt.seed)

    if opt.ff:
        assert opt.fp16, "fully-fused mode must be used with fp16 mode"
        from sdf.netowrk_ff import SDFNetwork
    elif opt.tcnn:
        assert opt.fp16, "tcnn mode must be used with fp16 mode"
        from sdf.network_tcnn import SDFNetwork        
    else:
        from sdf.netowrk import SqueezeSDFNetwork as SDFNetwork

    model = SDFNetwork(encoding="hashgrid")
    print(model)

    if opt.test:
        trainer = Trainer('ngp', model, workspace=opt.workspace, fp16=opt.fp16, use_checkpoint='best', eval_interval=1)
        trainer.save_mesh(os.path.join(opt.workspace, 'results', 'output.ply'), 1024)

    else:
        from sdf.provider import SDFDataset
        from loss import mape_loss

        train_dataset = SDFDataset(opt.path, size=100, num_samples=2**14,
                                    sample_boxes=[
                                        SampleBox("./data/sdf/sponza_sample_box_1.obj", num_samples=256),
                                        SampleBox("./data/sdf/sponza_sample_box_2.obj", num_samples=256),
                                        SampleBox("./data/sdf/sponza_sample_box_3.obj", num_samples=256),
                                        SampleBox("./data/sdf/sponza_sample_box_curtains_1.obj", num_samples=256),
                                        SampleBox("./data/sdf/sponza_sample_box_curtains_2.obj", num_samples=256),
                                        SampleBox("./data/sdf/sponza_sample_box_lion.obj", num_samples=256),
                                        SampleBox("./data/sdf/sponza_sample_box_stick_1.obj", num_samples=256),
                                        SampleBox("./data/sdf/sponza_sample_box_stick_2.obj", num_samples=256)
                                    ])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

        valid_dataset = SDFDataset(opt.path, size=1, num_samples=2**18) # just a dummy
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1)

        criterion = mape_loss # torch.nn.L1Loss()

        optimizer = lambda model: torch.optim.Adam(model.get_params(), lr=opt.lr, betas=(0.9, 0.99), eps=1e-15)
        # optimizer = lambda model: torch.optim.LBFGS(model.parameters(), lr=opt.lr)


        scheduler = lambda optimizer: optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

        scheduler = lambda optimizer: optim.lr_scheduler.CyclicLR(
                optimizer, 
                scale_fn=scale_fn, scale_mode="cycle",
                step_size_up=1, step_size_down=9, 
                base_lr=1e-5, max_lr=1e-4,
                cycle_momentum=False,
                # verbose=True
                )

        trainer = Trainer('ngp', model, workspace=opt.workspace, optimizer=optimizer, criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint='latest', eval_interval=100)

        trainer.train(train_loader, valid_loader, 100000)

        # also test
        trainer.save_mesh(os.path.join(opt.workspace, 'results', 'output.ply'), 1024)
