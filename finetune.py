import argparse
from logging import getLogger
import torch
from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import init_seed, init_logger, get_trainer, set_color

from unisrec import UniSRec
from data.dataset import UniSRecDataset
import os
import predict

def finetune(dataset, pretrained_file, fix_enc=True, **kwargs):
    # 配置文件
    props = ['props/UniSRec.yaml', 'props/finetune.yaml']
    print(props)

    # 配置初始化
    config = Config(model=UniSRec, dataset=dataset, config_file_list=props, config_dict=kwargs)
    init_seed(config['seed'], config['reproducibility'])
    # logger 初始化
    # init_logger(config)
    # logger = getLogger()
    # logger.info(config)


    dataset = UniSRecDataset(config)


    # 数据集划分
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # 模型初始化&加载预训练参数
    model = UniSRec(config, train_data.dataset).to(config['device'])

    if pretrained_file != '':
        checkpoint = torch.load(pretrained_file)
        # logger.info(f'Loading from {pretrained_file}')
        # logger.info(f'Transfer [{checkpoint["config"]["dataset"]}] -> [{dataset}]')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        if fix_enc:
            # logger.info(f'Fix encoder parameters.')
            for _ in model.position_embedding.parameters():
                _.requires_grad = False
            for _ in model.trm_encoder.parameters():
                _.requires_grad = False
    # logger.info(model)

    # 训练器初始化 config['MODEL_TYPE']: ModelType.SEQUENTIAL, config['model']: UniSRec
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    
    # saved_model_file = "{}-{}.pth".format(trainer.config["model"], "lianhua-finetuned")
    saved_model_file = 'UniSRec-lianhua-finetuned.pth'
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
    parser.add_argument('-p', type=str, default='/ml/output/UniSRec-LIANHUA-6.pth', help='pre-trained model path')
    parser.add_argument('-f', type=bool, default=True)

    parser.add_argument('-fp', type=str, default='/ml/output/UniSRec-lianhua-finetuned.pth', help='finetuned model path')
    parser.add_argument('-t', type=str, default=150, help='top k scoring items')
    parser.add_argument('-sp', type=str, default='/ml/output/result/', help= 'result save path')

    args, unparsed = parser.parse_known_args()
    
    finetune(args.d, pretrained_file=args.p, fix_enc=args.f)
    
    predict.predictor(args.d, args.fp, args.t, args.sp)
