'''
Main script to train NMT models
'''

from data_prep import DataTensorLoader
import sys
import os
import argparse
import torch
from transformers import AdamW
from tools import AverageMeter, get_default_device, set_seeds
from models import T5Based
from torch.utils.data import TensorDataset, DataLoader

def train(train_loader, model, output_to_loss, optimizer, epoch, device, print_freq=5):
    '''
    Run one train epoch
    '''
    losses = AverageMeter()

    # switch to train mode
    model.train()

    for i, (inp_id, mask, out_id) in enumerate(train_loader):

        inp_id = inp_id.to(device)
        mask = mask.to(device)
        out_id = out_id.to(device)

        # Forward pass
        outputs = model(input_ids=inp_id, attention_mask=mask, labels=out_id)
        loss = output_to_loss(outputs)

        # Backward pass and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # record loss
        losses.update(loss.item(), inp_id.size(0))

        if i % print_freq == 0:
            text = f'Epoch: [{epoch}][{i}/{len(train_loader)}]\tLoss {losses.val:.4f} ({losses.avg:.4f})'
            print(text)               


if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('OUT', type=str, help='Specify output th file')
    commandLineParser.add_argument('--subset', type=str, default='cs-en', help="Specify translation")
    commandLineParser.add_argument('--arch', type=str, default='T5', help="Specify model architecture")
    commandLineParser.add_argument('--size', type=str, default='t5-base', help="Specify model size")
    commandLineParser.add_argument('--B', type=int, default=16, help="Specify batch size")
    commandLineParser.add_argument('--epochs', type=int, default=2, help="Specify epochs")
    commandLineParser.add_argument('--lr', type=float, default=0.000001, help="Specify learning rate")
    commandLineParser.add_argument('--sch', type=int, default=10, help="Specify scheduler rate")
    commandLineParser.add_argument('--seed', type=int, default=1, help="Specify seed")
    commandLineParser.add_argument('--num_points', type=int, default=-1, help="limit number of datapoints for debugging")
    args = commandLineParser.parse_args()

    set_seeds(args.seed)

    # Save the command run
    text = ' '.join(sys.argv)+'\n'
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(text)
    print(text)

    # Get the device
    device = get_default_device()

    # Initialise model
    model = T5Based(args.size)
    model.to(device)

    # Load the data as tensors
    dataloader = DataTensorLoader(model.tokenizer, subset=args.subset, lang_flip=True, arch=args.arch)
    input_ids, input_mask, output_ids = dataloader.get_train(num_points=args.num_points)

    # Use dataloader to handle batches
    train_ds = TensorDataset(input_ids, input_mask, output_ids)
    train_dl = DataLoader(train_ds, batch_size=args.B, shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.sch])

    # loss from output - criterion is wrapped inside model
    loss_from_output = lambda a: a[0]

    # Train
    for epoch in range(args.epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_dl, model, loss_from_output, optimizer, epoch, device)
        scheduler.step()

    # Save the trained model
    state = model.state_dict()
    torch.save(state, args.OUT)