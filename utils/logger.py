import logging

import torch

class logger:
    @classmethod
    def initialize(cls, args, output_dir):
        cls.logpath = output_dir

        logging.basicConfig(filemode='w',
                            filename=cls.logpath / 'log.txt',
                            level=logging.INFO,
                            format='%(message)s',
                            datefmt='%m-%d %H:%M:%S')

        # Console log config
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        # # Tensorboard writer
        # cls.tbd_writer = SummaryWriter(os.path.join(cls.logpath, 'tbd/runs'))

        # # Log arguments
        logging.info('\n================= Motion Blur Local Feature ================')
        for arg_key in args.__dict__:
            logging.info(' %20s: %-24s ' % (arg_key, str(args.__dict__[arg_key])))
        logging.info('============================================================\n')

    @classmethod
    def info(cls, msg):
        r"""Writes message to .txt"""
        logging.info(msg)

    @classmethod
    def save_model(cls, state, ckpt_name='checkpoint'):
        ckpt_file = '{}.pth'.format(ckpt_name)
        torch.save(state, ckpt_file)
        cls.info('%s saved @%d w/ val. Repeability score: %5.2f.\n' % (ckpt_name.stem, state['epoch'], state['repeatability']))