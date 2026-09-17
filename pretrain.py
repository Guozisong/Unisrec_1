import argparse
from logging import getLogger
from recbole.config import Config
from recbole.trainer.trainer import PretrainTrainer
from recbole.utils import init_seed, init_logger

from unisrec import UniSRec
from data.dataset import UniSRecDataset
from data.dataloader import CustomizedTrainDataLoader


def pretrain(dataset, **kwargs):
    props = ['props/UniSRec.yaml', 'props/pretrain.yaml']

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


    # 模型训练
    trainer.pretrain(pretrain_data, show_progress=True)

    return config['model'], config['dataset'], trainer.saved_model_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='lianhua', help='dataset name')
    parser.add_argument('--data-path', required=True, help='parent directory of the dataset')
    parser.add_argument('--checkpoint-dir', required=True)
    parser.add_argument('--checkpoint-path-file', required=True)
    args = parser.parse_args()

    model, dataset, checkpoint = pretrain(args.d, data_path=args.data_path,
                                          checkpoint_dir=args.checkpoint_dir)
    with open(args.checkpoint_path_file, 'w') as file:
        file.write(checkpoint)
