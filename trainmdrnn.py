""" Recurrent model training """
import argparse
from functools import partial
import os
from os.path import join, exists
from os import makedirs
from tqdm.auto import tqdm
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm.auto import tqdm
from utils.misc import save_checkpoint
from utils.misc import ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE
from utils.learning import EarlyStopping
from utils.learning import ReduceLROnPlateau

from data.loaders import RolloutSequenceDataset
from models.vae import VAE
from models.mdrnn import MDRNN, gmm_loss
torch.manual_seed(42)
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser("MDRNN training")
parser.add_argument('--logdir', type=str,
                    help="Where things are logged and models are loaded from.")
parser.add_argument('--noreload', action='store_true',
                    help="Do not reload if specified.")
parser.add_argument('--include_reward', action='store_true',
                    help="Add a reward modelisation term to the loss.")
parser.add_argument('--local_rank', type=int, default=int(os.environ.get('LOCAL_RANK', 0)), help='automatically filled by torchrun (LOCAL_RANK env var)')
args = parser.parse_args()

dist.init_process_group(backend='nccl', init_method='env://')
torch.cuda.set_device(args.local_rank)
device = torch.device('cuda', args.local_rank)
is_master = (dist.get_rank() == 0)

is_distributed = (
    dist.is_available()
    and dist.is_initialized()
    and dist.get_world_size() > 1          # true only when you launched with torchrun -n >1
)

def all_reduce_mean(t: torch.Tensor) -> torch.Tensor:
    rt = t.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def get_lr(opt):
    return opt.param_groups[0]['lr']


# constants
BSIZE = 128
SEQ_LEN = 32
epochs = 32

# Loading VAE
vae_file = join(args.logdir, 'vae', 'best.tar')
assert exists(vae_file), "No trained VAE in the logdir..."
state = torch.load(vae_file, map_location=device)
print("Loading VAE at epoch {} "
      "with test error {}".format(
          state['epoch'], state['precision']))

vae = VAE(3, LSIZE).to(device)
vae.load_state_dict(state['state_dict'])

# Loading model
rnn_dir = join(args.logdir, 'mdrnn')
rnn_file = join(rnn_dir, 'best.tar')

if not exists(rnn_dir):
    makedirs(rnn_dir, exist_ok=True)

mdrnn = MDRNN(LSIZE, ASIZE, RSIZE, 5)
mdrnn.to(device)
mdrnn = torch.nn.parallel.DistributedDataParallel(mdrnn, device_ids=[args.local_rank], output_device=args.local_rank)
optimizer = torch.optim.RMSprop(mdrnn.parameters(), lr=1e-3, alpha=.9)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
earlystopping = EarlyStopping('min', patience=30)


if exists(rnn_file) and not args.noreload:
    rnn_state = torch.load(rnn_file, map_location=device)
    print("Loading MDRNN at epoch {} "
          "with test error {}".format(
              rnn_state["epoch"], rnn_state["precision"]))
    if is_distributed:
        mdrnn.module.load_state_dict(rnn_state["state_dict"])
    else:
        mdrnn.load_state_dict(rnn_state["state_dict"])
    optimizer.load_state_dict(rnn_state["optimizer"])
    scheduler.load_state_dict(state['scheduler'])
    earlystopping.load_state_dict(state['earlystopping'])


# Data Loading
transform = transforms.Lambda(lambda x: np.transpose(x, (0, 3, 1, 2)) / 255)
train_set = RolloutSequenceDataset('datasets/carracing', SEQ_LEN, transform, buffer_size=256)
test_set = RolloutSequenceDataset('datasets/carracing', SEQ_LEN, transform, train=False, buffer_size=256)
train_sampler = DistributedSampler(train_set)
test_sampler = DistributedSampler(test_set, shuffle=False)
train_loader = DataLoader(train_set, batch_size=BSIZE, sampler=train_sampler, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=BSIZE, sampler=test_sampler, num_workers=8, pin_memory=True)


def to_latent(obs, next_obs):
    """ Transform observations to latent space.

    :args obs: 5D torch tensor (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)
    :args next_obs: 5D torch tensor (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)

    :returns: (latent_obs, latent_next_obs)
        - latent_obs: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
        - next_latent_obs: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
    """
    with torch.no_grad():
        obs, next_obs = [
            f.upsample(x.view(-1, 3, SIZE, SIZE), size=RED_SIZE,
                       mode='bilinear', align_corners=True)
            for x in (obs, next_obs)]

        (obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma) = [
            vae(x)[1:] for x in (obs, next_obs)]

        latent_obs, latent_next_obs = [
            (x_mu + x_logsigma.exp() * torch.randn_like(x_mu)).view(-1, SEQ_LEN, LSIZE)
            for x_mu, x_logsigma in
            [(obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma)]]
    return latent_obs, latent_next_obs


def get_loss(latent_obs, action, reward, terminal,
             latent_next_obs, include_reward: bool):
    """ Compute losses.

    The loss that is computed is:
    (GMMLoss(latent_next_obs, GMMPredicted) + MSE(reward, predicted_reward) +
         BCE(terminal, logit_terminal)) / (LSIZE + 2)
    The LSIZE + 2 factor is here to counteract the fact that the GMMLoss scales
    approximately linearily with LSIZE. All losses are averaged both on the
    batch and the sequence dimensions (the two first dimensions).

    :args latent_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor
    :args action: (BSIZE, SEQ_LEN, ASIZE) torch tensor
    :args reward: (BSIZE, SEQ_LEN) torch tensor
    :args latent_next_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor

    :returns: dictionary of losses, containing the gmm, the mse, the bce and
        the averaged loss.
    """
    latent_obs, action, \
        reward, terminal, \
        latent_next_obs = [arr.transpose(1, 0)
                           for arr in [latent_obs, action,
                                       reward, terminal,
                                       latent_next_obs]]
    mus, sigmas, logpi, rs, ds = mdrnn(action, latent_obs)
    gmm = gmm_loss(latent_next_obs, mus, sigmas, logpi)
    bce = f.binary_cross_entropy_with_logits(ds, terminal)
    if include_reward:
        mse = f.mse_loss(rs, reward)
        scale = LSIZE + 2
    else:
        mse = 0
        scale = LSIZE + 1
    loss = (gmm + bce + mse) / scale
    return dict(gmm=gmm, bce=bce, mse=mse, loss=loss)


def data_pass(epoch, train, include_reward):  # pylint: disable=too-many-locals
    """ One pass through the data """
    if train:
        mdrnn.train()
        loader = train_loader
        train_sampler.set_epoch(epoch)
    else:
        mdrnn.eval()
        loader = test_loader

    loader.dataset.load_next_buffer()

    cum_loss = 0
    cum_gmm = 0
    cum_bce = 0
    cum_mse = 0

    pbar = tqdm(total=len(loader.dataset), desc=f"Epoch {epoch:>3}", disable=not is_master, leave=False)
    for i, data in enumerate(loader):
        obs, action, reward, terminal, next_obs = [arr.to(device) for arr in data]

        # transform obs
        latent_obs, latent_next_obs = to_latent(obs, next_obs)

        if train:
            losses = get_loss(latent_obs, action, reward,
                              terminal, latent_next_obs, include_reward)

            optimizer.zero_grad()
            losses['loss'].backward()
            optimizer.step()
        else:
            with torch.no_grad():
                losses = get_loss(latent_obs, action, reward,
                                  terminal, latent_next_obs, include_reward)

        cum_loss += losses['loss'].item()
        cum_gmm += losses['gmm'].item()
        cum_bce += losses['bce'].item()
        cum_mse += losses['mse'].item() if hasattr(losses['mse'], 'item') else \
            losses['mse']

        if is_master:
            pbar.set_postfix(loss=cum_loss / (i + 1), bce=cum_bce / (i + 1), gmm=cum_gmm / LSIZE / (i + 1), mse=cum_mse / (i + 1))
        pbar.update(BSIZE)
    pbar.close()
    return cum_loss * BSIZE / len(loader.dataset)


train = partial(data_pass, train=True, include_reward=args.include_reward)
test = partial(data_pass, train=False, include_reward=args.include_reward)

cur_best = None
epoch_bar = tqdm(range(epochs), desc='epochs', disable=not is_master)
for e in epoch_bar:
    train_loss = train(e)
    test_loss = test(e)

    # ---- sync losses across ranks ----
    g_train = all_reduce_mean(torch.tensor(train_loss, device=device))
    g_valid = all_reduce_mean(torch.tensor(test_loss, device=device))

    scheduler.step(g_valid.item())
    earlystopping.step(g_valid.item())

    if is_master:
        epoch_bar.set_postfix(train=g_train.item(), valid=g_valid.item(),
                              lr=get_lr(optimizer))

        is_best = (cur_best is None) or g_valid.item() < cur_best
        if is_best:
            cur_best = g_valid.item()

        save_checkpoint({
            "state_dict": mdrnn.module.state_dict() if is_distributed else mdrnn.state_dict(),
            "optimizer": optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'earlystopping': earlystopping.state_dict(),
            "precision": g_valid.item(),
            "epoch": e}, is_best,
            join(rnn_dir, 'checkpoint.tar'), rnn_file)

    if earlystopping.stop:
        if is_master:
            print(f"End of Training because of early stopping at epoch {e}")
        break

if is_master:
    print("******** Training complete, tearing down process group ********")
dist.destroy_process_group()
