import os
import datetime
import glob
import tqdm
from pathlib import Path
import torch
from tensorboardX import SummaryWriter

from configs import config
from utils import common_utils, train_utils
from utils.logger import logger
from datasets import create_dataloader
from model import get_model


# Basic setting
args, cfg = config.parse_config()

start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
# Check directories
output_dir = Path(args.log_dir, args.exper_name, start_time)
common_utils.check_directory(output_dir)
common_utils.check_directory('data')
ckpt_dir = output_dir / 'ckpt'
common_utils.check_directory(ckpt_dir)

# Create logger
logger.initialize(args, output_dir)

# Set random seeds
if args.fix_random_seed:
    logger.info(('Fix random seed as {}. '.format(args.random_seed)))
    common_utils.set_random_seed(args.random_seed)

tensorboard_log = SummaryWriter(common_utils.get_writer_path(args.exper_name, start_time))

# Create dataset
train_dataloaders = create_dataloader.build_dataloaders(cfg['data'], task='train', is_debugging=args.is_debugging)
val_dataloaders = create_dataloader.build_dataloaders(cfg['data'], task='val', is_debugging=args.is_debugging)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = get_model.load_model(cfg['model'])
model.to(device)


optimizer = train_utils.build_optimizer(filter(lambda p: p.requires_grad, model.parameters()), cfg['model']['optimizer'])


total_epochs = cfg['model']['optimizer']['total_epochs']
start_epoch = 0
last_epoch = -1
best_epoch = 0
best_repeatability = 0

if args.resume_training is not None:
    start_epoch, best_repeatability = get_model.load_pretrained_model(
        model=model, filename=args.resume_training, logger=logger, optimizer=optimizer
    )
    best_epoch = start_epoch
    last_epoch = start_epoch + 1

## Count the number of learnable parameters.
logger.info("================ List of Learnable model parameters ================ ")
for n,p in model.named_parameters():
    if p.requires_grad:
        logger.info("{} {}".format(n, p.data.shape))
    else:
        logger.info("\n\n\n None learnable params {} {}".format( n ,p.data.shape))
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([torch.prod(torch.tensor(p.size())) for p in model_parameters])
logger.info("The number of learnable parameters : {} ".format(params.data))
logger.info("==================================================================== ")


scheduler = train_utils.build_scheduler(
    optimizer,
    total_epochs=total_epochs,
    last_epoch=last_epoch,
    optim_cfg=cfg['model']['optimizer'])


if best_epoch == 0:
    with torch.no_grad():
        for val_loader in val_dataloaders:
            repeatability,_,_,_,_ = train_utils.check_val_repeatability(
                val_loader['dataloader'], model=model, device=device, tb_log=None, cur_epoch=-1,
                cell_size=cfg['model']['cell_size'], nms_size=cfg['model']['nms_size'], num_points=25
            )
            best_repeatability = repeatability
            best_epoch = -1
            logger.info(('\n Epoch -1 : Repeatability Validation: {:.3f}.'.format(repeatability)))


logger.info("================ Start Training ================ ")
with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True) as tbar:
    for cur_epoch in tbar:
        for train_loader in train_dataloaders:
            train_utils.train_model(
                cur_epoch=cur_epoch, dataloader=train_loader['dataloader'], model=model,
                optimizer=optimizer, device=device, tb_log=tensorboard_log, tbar=tbar, output_dir=output_dir,
                cell_size=cfg['model']['cell_size'], anchor_loss=cfg['model']['anchor_loss']
            )
        with torch.no_grad():
            for val_loader in val_dataloaders:
                rep_s, rep_m, error_overlap_s, error_overlap_m, possible_matches = train_utils.check_val_repeatability(
                    val_loader['dataloader'], model=model, device=device, tb_log=tensorboard_log, cur_epoch=cur_epoch,
                    cell_size=cfg['model']['cell_size'], nms_size=cfg['model']['nms_size'], num_points=25
                )
                tensorboard_log.add_scalar('repeatability_rep_s', rep_s, cur_epoch)
        logger.info(('Epoch {} (Validation) : Repeatability (rep_s): {:.3f}. '.format(cur_epoch, rep_s)))
        logger.info('\trep_m : {:.3f}, error_overlap_s : {:.3f}, error_overlap_m : {:.3f}, possible_matches : {:.3f}. \n'\
                    .format( rep_m, error_overlap_s, error_overlap_m, possible_matches))

        if best_repeatability < rep_s:
            best_repeatability = rep_s
            best_epoch = cur_epoch

            ckpt_name = ckpt_dir / 'best_model'
            logger.save_model(train_utils.ckpt_state(model, optimizer, cur_epoch, rep_s), ckpt_name)
    

        trained_epoch = cur_epoch + 1
        if trained_epoch % cfg['ckpt_save_interval'] == 0:
            ckpt_list = glob.glob(str(ckpt_dir / 'checkpoint_epoch_*.pth'))
            ckpt_list.sort(key=os.path.getmtime)

            if ckpt_list.__len__() >= cfg['max_ckpt_save_num']:
                for cur_file_idx in range(0, len(ckpt_list) - cfg['max_ckpt_save_num'] + 1):
                    os.remove(ckpt_list[cur_file_idx])

            ckpt_name = ckpt_dir / ('checkpoint_epoch_%d' % trained_epoch)
            logger.save_model(train_utils.ckpt_state(model, optimizer, cur_epoch, best_repeatability), ckpt_name)        

        scheduler.step()

logger.info("Best validation repeatability score : {} at epoch {}. ".format(best_repeatability, best_epoch))
logger.info("================ Enhd Training ================ \n\n")