from __future__ import unicode_literals, print_function

import os
# import numpy as np
# import nibabel as nib

from tqdm import tqdm
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference, SliceInferer
from monai.metrics import DiceMetric
from monai.transforms import (
    AsDiscrete
)
from monai.data import (
    decollate_batch,
)

from load_data.swinunetr_dataloader import dataloader
import torch
import math
import nibabel as nib
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from datetime import datetime
from prep.generate_posencs import norm_sinenc
from monai.optimizers import WarmupCosineSchedule

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def parser():
    import argparse
    parser = argparse.ArgumentParser(description='Executable script for training the Mouse Brain Extractor model.')
    parser.add_argument('-d', help='Number of dimensions (2 for 2D model; 3 for 3D model).', required=True, type= int, choices=[2,3])
    parser.add_argument('-m', help='Model type.', required = True, choices=['mod5', 'orig'])
    parser.add_argument('-o', help='Model name and path. What you would like to name the output trained weights. It must end in .pth.', 
                        required= True)
    parser.add_argument('-j', help='Training dataset JSON file.', required= True)
    parser.add_argument('-f', help='The number which indicates which training and validation dataset you would like to use (0 for train_0 and val_0).', 
                        required= True, type=int)
    parser.add_argument('--p', help='Pre-trained model to load to continue training.', required= False)
    parser.add_argument('--iters', help='Maximum number of iterations.', required= False, default=50000, type=int)
    parser.add_argument('--eval', help='Nth iteration to compute validation.', required= False, default=100, type=int)
    parser.add_argument('--n', help='Number of ROI samples per image.', default=4, type=int)
    parser.add_argument('--cachedDW', help='Number of cached data workers for data loading.', default=2, type=int)
    parser.add_argument('--workers', help='Number of data workers for training data.', default=2, type=int)
    parser.add_argument('--batch', help='Batch size.', default=1, type=int)
    parser.add_argument('--cachenum', help='Number of samples to cache for training data.', default=1, type=int)
    parser.add_argument('--valcachenum', help='Number of samples to cache for validation data.', default=1, type=int)
    parser.add_argument('--lr_decay_iternum', help='Nth iteration to for lr decay. Deprecated.', default=50, type=int)
    parser.add_argument('--patch_size', help='Patch size. One int number should be given. Patch size is isotropic.', default=96, type=int)
    parser.add_argument('--lr', help='Starting learning rate. Linear warm up cosine annealing learning rate scheduler is used.', 
                        default=1e-4, type=float)

    return parser

def main():
    args = parser().parse_args()
    
    custom_dataloader = dataloader(args.d, args.j, args.f, args.m, num_samples=args.n, 
                                   batch_size=args.batch, cachenum=args.cachenum, val_cachenum=args.valcachenum, patchsize=args.patch_size)
    train_loader, val_loader = custom_dataloader.prep_data(args.cachedDW, args.workers)

    if args.m == 'mod5':
        from models.swin_unetr_mod5 import SwinUNETR
        pos = 'normcoord'
    else:
        from models.swin_unetr import SwinUNETR

    use_checkpoint = False
    if args.p:
        use_checkpoint = True

    patch_size = args.patch_size
    if args.d == 2:
        imgsize = (patch_size, patch_size)
        coronal_inferer = SliceInferer(
        roi_size=imgsize,
        sw_batch_size=48,
        spatial_dim=1, 
        cval=0,
        progress=False,
        sw_device="cuda", 
        device="cpu",
        overlap=0.25
    )

    if args.d == 3:
        imgsize = (patch_size, patch_size, patch_size)
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 2

    model = SwinUNETR(
        img_size=imgsize,
        in_channels=1,
        out_channels=num_classes,
        feature_size=48,
        use_checkpoint=use_checkpoint,
        spatial_dims=args.d
    ).to(device)

    if args.p:
        checkpoint_params = torch.load(args.p)
        model.load_from(weights=checkpoint_params['model_weights'])
        model.load_state_dict(checkpoint_params['model_weights'])

    modelname = args.o
    model_folder = os.path.dirname(os.path.expanduser(modelname))

    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    
    now = datetime.now()
    metrics_file = os.path.join(model_folder, 'metrics_'+ now.strftime("%Y_%m%d_%H-%M-%S") + '.csv')
    with open(metrics_file, 'w') as f:
        f.write('Iteration,Average Loss,Average Val Dice\n')

    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy('file_system')
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True, include_background=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler()
    # lmbda = lambda epoch: 0.99 ** epoch

    # scheduler = LambdaLR(optimizer, lr_lambda=lmbda)
    scheduler = WarmupCosineSchedule(optimizer, 50, args.iters, end_lr=1e-6)
    if args.p:
        optimizer.load_state_dict(checkpoint_params['optimizer'])
        scaler.load_state_dict(checkpoint_params['scaler'])
        scheduler.load_state_dict(checkpoint_params['scheduler'])

    def validation(epoch_iterator_val):
        model.eval()
        with torch.no_grad():
            i = 0
            for batch in epoch_iterator_val:
                val_inputs, val_labels = (batch["image"], batch["label"])
                
                if (args.m == 'mod5'):
                    norm_coords = batch[pos]
                    global_pos = norm_sinenc(norm_coords, norm_coords.shape)
                    inputs = torch.cat([val_inputs,global_pos],1)
                else:
                    inputs = val_inputs
                with torch.cuda.amp.autocast():
                    if args.d == 3:    
                        val_outputs = sliding_window_inference(inputs, imgsize, 3, model, sw_device="cuda", device="cpu")
                    else:
                        val_outputs = coronal_inferer(inputs=inputs, network=model) 
                val_labels_list = decollate_batch(val_labels)
                val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
                val_outputs_list = decollate_batch(val_outputs.detach())
                val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
                dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))
                del val_outputs
                i += 1
            mean_dice_val = dice_metric.aggregate().detach().item()
            dice_metric.reset()
        return mean_dice_val


    def train(global_step, train_loader, dice_val_best, global_step_best):
        model.train()
        epoch_loss = 0
        step = 0
        epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
        for step, batch in enumerate(epoch_iterator):
            step += 1
            x, y = (batch["image"].to(device), batch["label"].to(device))
            if (args.m == 'mod5'):
                norm_coords = batch[pos]
                global_pos = norm_sinenc(norm_coords, norm_coords.shape)
                global_pos = global_pos.to(device)
                inputs = torch.cat([x,global_pos],1)
            else:
                inputs = x
            
            with torch.cuda.amp.autocast():
                logit_map = model(inputs)
                loss_dicece = loss_function(logit_map, y)
                loss = loss_dicece 
                
            scaler.scale(loss).backward()

            epoch_loss += loss.detach().item()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            epoch_iterator.set_description(f"Training ({global_step} / {max_iterations} Steps) (loss={loss:2.5f})")
            if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
                epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
                dice_val = validation(epoch_iterator_val)

                epoch_loss /= step
                epoch_loss_values.append(epoch_loss)
                metric_values.append(dice_val)
                if dice_val > dice_val_best:
                    dice_val_best = dice_val
                    global_step_best = global_step
                    checkpoint = {
                        'model_weights': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scaler': scaler.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'best_valdice': dice_val_best,
                        'global_step': global_step,
                        'global_step_best': global_step_best,
                        'max_iters': args.iters
                    }
                    torch.save(checkpoint, modelname)
                    print(
                        "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val)
                    )
                else:
                    checkpoint = {
                        'model_weights': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scaler': scaler.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'best_valdice': dice_val_best,
                        'global_step': global_step,
                        'global_step_best': global_step_best,
                        'max_iters': args.iters
                    }
                    torch.save(checkpoint, modelname.split('.pth')[0]+'_latest.pth')
                    print(
                        "Latest model saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                            dice_val_best, dice_val
                        )
                    )
                del epoch_iterator_val
                with open(metrics_file, 'a') as f:
                    f.write(f'{global_step},{epoch_loss},{dice_val}\n')
            # if (global_step % args.lr_decay_iternum == 0 and global_step != 0):
            scheduler.step()
            print("Learning rate:", scheduler.get_lr())
            
            global_step += 1
        return global_step, dice_val_best, global_step_best

    max_iterations = args.iters
    eval_num = args.eval
    post_label = AsDiscrete(to_onehot=num_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    if args.p:
        checkpoint_params = torch.load(args.p)
        global_step = checkpoint_params['global_step']
        print(global_step)
        dice_val_best = checkpoint_params['best_valdice']
        global_step_best = checkpoint_params['global_step_best']
    epoch_loss_values = []
    metric_values = []
    while global_step < max_iterations:
        global_step, dice_val_best, global_step_best = train(global_step, train_loader, dice_val_best, global_step_best)

if __name__ == "__main__":
    main()