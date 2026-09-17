import argparse
import os
import torch
from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import init_seed, get_trainer

from unisrec import UniSRec
from data.dataset import UniSRecDataset

def finetune(dataset, pretrained_file, fix_enc=True, **kwargs):
    # 配置文件
    props = ['props/UniSRec.yaml', 'props/finetune.yaml']
    print(props)

    # 配置初始化
    config = Config(model=UniSRec, dataset=dataset, config_file_list=props, config_dict=kwargs)
    init_seed(config['seed'], config['reproducibility'])
    dataset = UniSRecDataset(config)


    # 数据集划分
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # 模型初始化&加载预训练参数
    model = UniSRec(config, train_data.dataset).to(config['device'])

    if pretrained_file != '':
        checkpoint = torch.load(pretrained_file, map_location=config['device'])
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        if fix_enc:
            for _ in model.position_embedding.parameters():
                _.requires_grad = False
            for _ in model.trm_encoder.parameters():
                _.requires_grad = False

    # 训练器初始化 config['MODEL_TYPE']: ModelType.SEQUENTIAL, config['model']: UniSRec
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    
    saved_model_file = f'UniSRec-{config["dataset"]}-finetuned.pth'
    trainer.saved_model_file = os.path.join(trainer.checkpoint_dir, saved_model_file)
    
    # 模型训练
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=True # config['show_progress']
    )
    
    return config['model'], config['dataset'], {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='lianhua', help='dataset name for training')
    parser.add_argument('-p', type=str, required=True, help='pre-trained model path')
    parser.add_argument('--no-fix-encoder', action='store_false', dest='fix_enc')
    parser.add_argument('--data-path', required=True, help='parent directory of the dataset')
    parser.add_argument('--checkpoint-dir', required=True)
    args = parser.parse_args()
    
    finetune(args.d, pretrained_file=args.p, fix_enc=args.fix_enc,
             data_path=args.data_path, checkpoint_dir=args.checkpoint_dir)
