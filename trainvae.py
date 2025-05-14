""" Training VAE """
import argparse
import os
from os.path import join, exists
from bisect import bisect
from os import listdir
from os.path import join, isdir
from os import makedirs
from tqdm.auto import tqdm
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch
import torch.utils.data
import math
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image

from models.vae import VAE
from utils.misc import save_checkpoint
from utils.misc import LSIZE, RED_SIZE
from utils.learning import EarlyStopping
from utils.learning import ReduceLROnPlateau
from data.loaders import RolloutObservationDataset

torch.manual_seed(42)
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='VAE Trainer')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 1000)')  # num of full passes over data
parser.add_argument('--logdir', type=str, help='Directory where results are logged')
parser.add_argument('--noreload', action='store_true',
                    help='Best model is not reloaded if specified')
parser.add_argument('--nosamples', action='store_true',
                    help='Does not save samples during training if specified')
parser.add_argument('--local_rank', type=int, default=int(os.environ.get('LOCAL_RANK', 0)), help='will be overwritten by torchrun via LOCAL_RANK')
args = parser.parse_args()

dist.init_process_group(backend='nccl', init_method='env://')
# LOCAL_RANK (in env) is the per‐process GPU index when you use torchrun or launch
torch.cuda.set_device(args.local_rank)
device = torch.device('cuda', args.local_rank)
is_master = (dist.get_rank() == 0)

is_distributed = (
    dist.is_available()
    and dist.is_initialized()
    and dist.get_world_size() > 1          # true only when you launched with torchrun -n >1
)


def get_lr(opt):
    return opt.param_groups[0]['lr']


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor(),
])

buffer_size = 256
dataset_train = RolloutObservationDataset('datasets/carracing', transform_train, train=True, buffer_size=buffer_size)
dataset_test = RolloutObservationDataset('datasets/carracing', transform_test, train=False, buffer_size=buffer_size)
num_full_passes = args.epochs
epochs_per_pass = int(math.ceil(len(dataset_train._files) / buffer_size))
num_epochs = num_full_passes * epochs_per_pass
train_sampler = DistributedSampler(dataset_train)
test_sampler = DistributedSampler(dataset_test, shuffle=False)
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, sampler=train_sampler, num_workers=16, pin_memory=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=16, pin_memory=True)

model = VAE(3, LSIZE).to(device)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
optimizer = optim.Adam(model.parameters())
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
earlystopping = EarlyStopping('min', patience=30)


def loss_function(recon_x, x, mu, logsigma):
    """ VAE loss function 
    Reconstruction + KL divergence losses summed over all elements and batch
    """
    BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
    return BCE + KLD


def train(epoch):
    """ One training epoch """
    model.train()
    dataset_train.load_next_buffer()
    train_sampler.set_epoch(epoch)   # shuffle differs per epoch
    train_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch:>4}", leave=False, disable=not is_master)
    for batch_idx, data in enumerate(pbar):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if is_master:
            pbar.set_postfix(loss=loss.item() / len(data))
    avg = train_loss / len(train_loader.dataset)
    return avg


def test():
    """ One test epoch """
    model.eval()
    dataset_test.load_next_buffer()
    test_loss = 0
    with torch.no_grad():
        for data in tqdm(test_loader, desc="valid", leave=False, disable=True):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
    test_loss /= len(test_loader.dataset)
    return test_loss


# check vae dir exists, if not, create it
vae_dir = join(args.logdir, 'vae')
if not exists(vae_dir):
    makedirs(vae_dir, exist_ok=True)
    makedirs(join(vae_dir, 'samples'), exist_ok=True)

reload_file = join(vae_dir, 'best.tar')
if not args.noreload and exists(reload_file):
    state = torch.load(reload_file, map_location=device)
    print("Reloading model at epoch {}"
          ", with test error {}".format(
              state['epoch'],
              state['precision']))
    if is_distributed:
        model.module.load_state_dict(state['state_dict'])
    else:
        model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    earlystopping.load_state_dict(state['earlystopping'])


cur_best = None

epoch_bar = tqdm(range(1, num_epochs + 1), desc='epochs', disable=not is_master)
for epoch in epoch_bar:
    train_loss = train(epoch)
    test_loss = test()
    scheduler.step(test_loss)
    earlystopping.step(test_loss)

    # sync local losses across all ranks
    t_loss_tensor = torch.tensor(train_loss, device=device)
    v_loss_tensor = torch.tensor(test_loss, device=device)
    global_train = all_reduce_mean(t_loss_tensor)
    global_valid = all_reduce_mean(v_loss_tensor)
    if is_master:
        epoch_bar.set_postfix(train=global_train.item(), valid=global_valid.item(), lr=get_lr(optimizer))

    # checkpointing
    best_filename = join(vae_dir, 'best.tar')
    filename = join(vae_dir, 'checkpoint.tar')
    is_best = not cur_best or test_loss < cur_best
    if is_best:
        cur_best = test_loss

    if is_master:
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.module.state_dict() if is_distributed else model.state_dict(),
            'precision': test_loss,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'earlystopping': earlystopping.state_dict()
        }, is_best, filename, best_filename)
        if (epoch % 100 == 0) and (not args.nosamples):
            with torch.no_grad():
                sample = torch.randn(RED_SIZE, LSIZE).to(device)
                sample = model.module.decoder(sample).cpu() if is_distributed else model.decoder(sample).cpu()
                save_image(sample.view(64, 3, RED_SIZE, RED_SIZE), join(vae_dir, 'samples/sample_' + str(epoch) + '.png'))

    if earlystopping.stop:
        print("End of Training because of early stopping at epoch {}".format(epoch))
        break

if is_master:
    print("******** Training complete, tearing down process group ********")
dist.destroy_process_group()
