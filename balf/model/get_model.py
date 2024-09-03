import os
import torch

from ..model import mlp_ma_decoder

def load_pretrained_model(model, filename, logger, optimizer=None, device='cuda'):
    if not os.path.isfile(filename):
        raise FileNotFoundError

    logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if device=='cpu' else 'GPU'))
    loc_type = torch.device('cpu') if device=='cpu' else None
    checkpoint = torch.load(filename, map_location=loc_type)
    model_state_disk = checkpoint['model_state']
    epoch = checkpoint.get('epoch', -1)
    repeatability = checkpoint.get('repeatability', 0.0)

    update_model_state = {}
    for key, val in model_state_disk.items():
        if key in model.state_dict() and model.state_dict()[key].shape == model_state_disk[key].shape:
            update_model_state[key] = val
            logger.info('Update weight %s: %s' % (key, str(val.shape)))

    state_dict = model.state_dict()
    state_dict.update(update_model_state)
    model.load_state_dict(state_dict)

    for key in state_dict:
        if key not in update_model_state:
            logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

    if optimizer is not None:
        if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
            logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                        % (filename, 'CPU' if device=='cpu' else 'GPU'))
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        else:
            assert filename[-4] == '.', filename
            src_file, ext = filename[:-4], filename[-3:]
            optimizer_filename = '%s_optim.%s' % (src_file, ext)
            if os.path.exists(optimizer_filename):
                optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

    assert len(update_model_state) == len(model.state_dict())

    logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(model.state_dict())))

    return epoch, repeatability


def load_model(model_cfg):
    if model_cfg['name'] == 'mlp_ma':
        model = mlp_ma.MLPMA(model_cfg['network_architecture'])
    elif model_cfg['name'] == 'mlp_ma_decoder':
        model = mlp_ma_decoder.MLP_MA_DECODER(model_cfg['network_architecture'])
    return model