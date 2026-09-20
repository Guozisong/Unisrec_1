import argparse
from logging import getLogger
import torch
from recbole.config import Config
from recbole.trainer.trainer import PretrainTrainer
from recbole.utils import init_seed, init_logger

from unisrec import UniSRec
from recbole_data.dataset import UniSRecDataset
from recbole_data.dataloader import CustomizedTrainDataLoader


def restore_pretrain_checkpoint(checkpoint_path, config, model, trainer):
    checkpoint = torch.load(checkpoint_path, map_location=config['device'])
    required_keys = {'config', 'epoch', 'state_dict', 'optimizer'}
    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        raise ValueError(f'Invalid pretraining checkpoint; missing: {sorted(missing_keys)}')

    checkpoint_dataset = checkpoint['config']['dataset']
    if checkpoint_dataset != config['dataset']:
        raise ValueError(
            f'Checkpoint dataset {checkpoint_dataset!r} does not match {config["dataset"]!r}'
        )

    start_epoch = checkpoint['epoch'] + 1
    if start_epoch >= config['pretrain_epochs']:
        raise ValueError(
            f'Checkpoint already reached epoch {start_epoch}; '
            f'pretrain_epochs must be greater than {start_epoch}'
        )

    model.load_state_dict(checkpoint['state_dict'])
    model.load_other_parameter(checkpoint.get('other_parameter'))
    trainer.optimizer.load_state_dict(checkpoint['optimizer'])
    trainer.start_epoch = start_epoch
    return start_epoch


def pretrain(dataset, resume_checkpoint=None, **kwargs):
    props = ['configs/UniSRec.yaml', 'configs/pretrain.yaml']

    config = Config(model=UniSRec, dataset=dataset, config_file_list=props, config_dict=kwargs)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # 数据处理
    dataset = UniSRecDataset(config)
    logger.info(dataset)
  
    # 训练数据
    pretrain_dataset = dataset.build()[0]
    pretrain_data = CustomizedTrainDataLoader(config, pretrain_dataset, None, shuffle=True)

    # 模型初始化
    model = UniSRec(config, pretrain_data.dataset).to(config['device'])
    logger.info(model)


    # 训练器初始化
    trainer = PretrainTrainer(config, model)

    if resume_checkpoint:
        start_epoch = restore_pretrain_checkpoint(resume_checkpoint, config, model, trainer)
        logger.info(
            'Resuming pretraining from %s at epoch %s of %s',
            resume_checkpoint, start_epoch + 1, config['pretrain_epochs']
        )

    # 模型训练
    trainer.pretrain(pretrain_data, show_progress=True)

    return config['model'], config['dataset'], trainer.saved_model_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, required=True, help='dataset name')
    parser.add_argument('--data-path', required=True, help='parent directory of the dataset')
    parser.add_argument('--checkpoint-dir', required=True)
    parser.add_argument('--checkpoint-path-file', required=True)
    parser.add_argument('--resume-checkpoint', help='pretraining checkpoint to resume from')
    args = parser.parse_args()

    model, dataset, checkpoint = pretrain(args.d, resume_checkpoint=args.resume_checkpoint,
                                          data_path=args.data_path,
                                          checkpoint_dir=args.checkpoint_dir)
    with open(args.checkpoint_path_file, 'w') as file:
        file.write(checkpoint)
