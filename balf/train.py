import os
import datetime
import glob
import tqdm
from pathlib import Path
import torch
from tensorboardX import SummaryWriter

from .configs import config
from .utils import common_utils, train_utils
from .utils.logger import logger
from .datasets import create_dataloader
from .model import get_model
from .loss import loss_function


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

# usp_loss = loss_function.ScoreLoss(device, cfg['model']['unsuper_loss'])
usp_loss = None

optimizer = train_utils.build_optimizer(filter(lambda p: p.requires_grad, model.parameters()), cfg['model']['optimizer'])


total_epochs = cfg['model']['optimizer']['total_epochs']
start_epoch = 0
last_epoch = -1
best_epoch = 0
best_repeatability = 0

if args.resume_training is not None:
    start_epoch, best_repeatability = get_model.load_pretrained_model(
        model=model, filename=args.resume_training, logger=logger, optimizer=optimizer, device=device
    )
    best_epoch = start_epoch
    last_epoch = start_epoch - 1

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


# scheduler = train_utils.build_scheduler(
#     optimizer,
#     total_epochs=total_epochs,
#     last_epoch=last_epoch,
#     optim_cfg=cfg['model']['optimizer'])
scheduler = train_utils.build_scheduler(
    optimizer,
    total_epochs=total_epochs,
    last_epoch=last_epoch,
    scheduler_cfg=cfg['model']['scheduler'])


if best_epoch == 0:
    with torch.no_grad():
        for val_loader in val_dataloaders:
            repeatability,_,_,_,_,rep_s_nms,_,_,_,_, = train_utils.check_val_repeatability(
                val_loader['dataloader'], model=model, device=device, tb_log=None, cur_epoch=-1,
                cell_size=cfg['model']['cell_size'], nms_size=cfg['model']['nms_size'], num_points=25
            )
            # best_repeatability = repeatability
            best_repeatability = rep_s_nms
            best_epoch = -1
            logger.info(('\n Epoch -1 : Repeatability Validation: {:.3f}.'.format(repeatability)))
            logger.info(('\n Epoch -1 : KeyNet NMS Repeatability Validation: {:.3f}.\n\n'.format(rep_s_nms)))
            # logger.info(('\n Epoch -1 : Position Repeatability Validation: {:.3f}.\n\n'.format(rep_s_pos)))

count = 0
max_counts = 3
logger.info("================ Start Training ================ ")
with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True) as tbar:
    for cur_epoch in tbar:
        for train_loader in train_dataloaders:
            loss = train_utils.train_model(
                cur_epoch=cur_epoch, dataloader=train_loader['dataloader'], model=model, optimizer=optimizer,
                device=device, tb_log=tensorboard_log, tbar=tbar, output_dir=output_dir,cell_size=cfg['model']['cell_size'],
                anchor_loss=cfg['model']['anchor_loss'], usp_loss=usp_loss, repeatability_loss=None
            )
        if cur_epoch % 3 == 0:
            with torch.no_grad():
                for val_loader in val_dataloaders:
                    rep_s, rep_m, error_overlap_s, error_overlap_m, possible_matches,\
                    rep_s_nms, rep_m_nms, error_overlap_s_nms, error_overlap_m_nms, possible_matches_nms =\
                    train_utils.check_val_repeatability(
                        val_loader['dataloader'], model=model, device=device, tb_log=tensorboard_log, cur_epoch=cur_epoch,
                        cell_size=cfg['model']['cell_size'], nms_size=cfg['model']['nms_size'], num_points=25
                    )
                    tensorboard_log.add_scalar('repeatability_rep_s', rep_s, cur_epoch)
                    tensorboard_log.add_scalar('keynet_nms_repeatability_rep_s', rep_s_nms, cur_epoch)
                    # tensorboard_log.add_scalar('position_repeatability_rep_s', rep_s_pos, cur_epoch)
            logger.info(('Epoch {} (Validation) : Repeatability (rep_s): {:.3f}. '.format(cur_epoch, rep_s)))
            # logger.info('\trep_m : {:.3f}, error_overlap_s : {:.3f}, error_overlap_m : {:.3f}, possible_matches : {:.3f}.'\
            #             .format( rep_m, error_overlap_s, error_overlap_m, possible_matches))

            logger.info(('\tEpoch {} (Validation) : KeyNet NMS Repeatability (rep_s_nms): {:.3f}. '.format(cur_epoch, rep_s_nms)))
            # logger.info('\t\trep_m_nms : {:.3f}, error_overlap_s_nms : {:.3f}, error_overlap_m_nms : {:.3f}, possible_matches_nms : {:.3f}. \n\n'\
            #             .format( rep_m_nms, error_overlap_s_nms, error_overlap_m_nms, possible_matches_nms))
        else:
            rep_s_nms = 0

        # logger.info(('\tEpoch {} (Validation) : Position Repeatability (rep_s_nms): {:.3f}. '.format(cur_epoch, rep_s_pos)))
        # logger.info('\t\trep_m_nms : {:.3f}, error_overlap_s_nms : {:.3f}, error_overlap_m_nms : {:.3f}, possible_matches_nms : {:.3f}. \n\n'\
        #             .format( rep_m_pos, error_overlap_s_pos, error_overlap_m_pos, possible_matches_pos))

        # Control the early stopping
        if cur_epoch == 0:
            loss_best = loss
        else:
            # if best_repeatability < rep_s:
                # best_repeatability = rep_s
            if best_repeatability < rep_s_nms:
                best_repeatability = rep_s_nms
                best_epoch = cur_epoch + 1

                ckpt_name = ckpt_dir / 'best_model'
                logger.save_model(train_utils.ckpt_state(model, optimizer, best_epoch, best_repeatability), ckpt_name)

                count = 0
            elif rep_s_nms > 0:
                if loss_best > loss:
                    loss_best = loss
                else:
                    count += 1
    

        trained_epoch = cur_epoch + 1
        if trained_epoch % cfg['ckpt_save_interval'] == 0:
            ckpt_list = glob.glob(str(ckpt_dir / 'checkpoint_epoch_*.pth'))
            ckpt_list.sort(key=os.path.getmtime)

            if ckpt_list.__len__() >= cfg['max_ckpt_save_num']:
                for cur_file_idx in range(0, len(ckpt_list) - cfg['max_ckpt_save_num'] + 1):
                    os.remove(ckpt_list[cur_file_idx])

            ckpt_name = ckpt_dir / ('checkpoint_epoch_%d' % trained_epoch)
            logger.save_model(train_utils.ckpt_state(model, optimizer, trained_epoch, best_repeatability), ckpt_name)        

        scheduler.step()

        if count > max_counts:
            break

logger.info("Best validation repeatability score : {} at epoch {}. ".format(best_repeatability, best_epoch))
logger.info("================ End Training ================ \n\n")

# rep_s_pos, rep_m_pos, error_overlap_s_pos, error_overlap_m_pos, possible_matches_pos =\