import argparse
from recbole.config import Config
from recbole.data import data_preparation

from unisrec import UniSRec
from data.dataset import UniSRecDataset
import torch
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from odps import ODPS
from odps.models import Schema, Column
from dotenv import load_dotenv

def write_to_odps(df1):
    # 建立链接。
    load_dotenv("/ml/output/.env")
    access_id = os.getenv("ALI_ACCESS_ID")
    access_key = os.getenv("ALI_SECRET_ACCESS_KEY")
    project = 'jxaidataworks'
    endpoint = 'https://service.cn-hangzhou-vpc.maxcompute.aliyun-inc.com/api'

    odps = ODPS(access_id, access_key, project, endpoint=endpoint)
    
    table_name1 = 'lianhua_tmp_Unisrec_uid2simitem'

    columns = [
        Column(name='user_id', type='string', comment='用户索引'),
        Column(name='prod_id', type='string', comment='商品索引'),
        Column(name='similarity_score', type='double', comment='相似分数')]

    schema = Schema(columns=columns)
    
    # 检查表是否存在，如果存在则删除
    if odps.exist_table(table_name1):
        odps.delete_table(table_name1, if_exists=True)
        print(f"Table {table_name1} has been dropped.")
  
    odps.create_table(table_name1, schema, if_not_exists=True)
    print(f"Table {table_name1} already exists.")

    table = odps.get_table(table_name1)

    # 将DataFrame写入ODPS表
    odps.write_table(table_name1, df1, overwrite=True)

    print(f"Data has been written to {table_name1}")

def read_from_odps():
    load_dotenv("/ml/output/.env")
    access_id = os.getenv("ALI_ACCESS_ID")
    access_key = os.getenv("ALI_SECRET_ACCESS_KEY")
    project = 'jxaidataworks'
    endpoint = 'https://service.cn-hangzhou-vpc.maxcompute.aliyun-inc.com/api'

    # 初始化ODPS对象
    odps = ODPS(access_id, access_key, project, endpoint)

    # 商品品类
    sql_1 = '''SELECT prod_id, cate_level5_code FROM unisrec_items_info;'''
    # 在售商品
    sql_2 = '''SELECT prod_id FROM lianhua_recall_station_brand_category_grade_base_tmp GROUP BY prod_id;'''
    
    query_job_1 = odps.execute_sql(sql_1)
    items_info_1 = query_job_1.open_reader(tunnel=True)
    items_info_1 = items_info_1.to_pandas(n_process=4)
    
    query_job_2 = odps.execute_sql(sql_2)
    items_info_2 = query_job_2.open_reader(tunnel=True)
    items_info_2 = items_info_2.to_pandas(n_process=4)
    return (items_info_1, items_info_2)


def diversify_recommendations(rec_items, rec_items_score, prod_cate_info, valid_prod, max_per_cate=10, top_k=50):
    # 建立 prod_id -> cate_id 映射字典
    prod2cate = dict(zip(prod_cate_info['prod_id'], prod_cate_info['cate_level5_code']))
    valid_prod = valid_prod['prod_id'].to_list() 

    diversified_items = [] 
    diversified_scores = []

    for items, scores in zip(rec_items, rec_items_score):
        cate_count = {}
        new_items = []
        new_scores = []

        for item, score in zip(items, scores):

            cate_id = prod2cate.get(item, None)
            if cate_id is None:
                continue 
            # 保证商品有效
            if item not in valid_prod: 
                continue

            if cate_count.get(cate_id, 0) < max_per_cate: # 保证每个指定品类商品数不超过阈值
                new_items.append(item)
                new_scores.append(score)
                cate_count[cate_id] = cate_count.get(cate_id, 0) + 1

            # 提前终止，如果已经到达 top_k 限制
            if len(new_items) >= top_k:
                break

        # 保证返回长度不超过 top_k
        diversified_items.append(new_items[:top_k])
        diversified_scores.append(new_scores[:top_k])

    return diversified_items, diversified_scores

def _full_sort_batch_eval(batched_data, model, device, tot_item_num):
    interaction, history_index, positive_u, positive_i = batched_data

    scores = model.full_sort_predict(interaction.to(device))

    scores = scores.view(-1, tot_item_num)
    scores[:, 0] = -np.inf
    return interaction, scores, positive_u, positive_i


def predictor(dataset, model_file, top_k, result_save_path):
    # 配置文件
    props = ['props/UniSRec.yaml', 'props/finetune.yaml']
    config = Config(model=UniSRec, dataset=dataset, config_file_list=props)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("加载数据 ...")
    dataset_ = UniSRecDataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset_)

    with open(os.path.join(config["data_path"], "index2user.json"), 'r', encoding='utf-8') as f:
        index2user = json.load(f)
    with open(os.path.join(config["data_path"], "index2item.json"), 'r', encoding='utf-8') as f:
        index2item = json.load(f)

    print("加载模型 ...")
    model = UniSRec(config, test_data.dataset).to(config['device'])
    checkpoint = torch.load(model_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))
    model.eval()

    print("预测 ...")
    eval_func = _full_sort_batch_eval
    # item_tensor = test_data._dataset.get_item_feature().to(device)
    tot_item_num = test_data._dataset.item_num
    iter_data = test_data
    num_sample = 0
    
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    # 真实值
    users = []  # 用户
    target_items = []  # 目标商品
    items_seq = [] # 商品序列（模型输入）
    rec_items = []  # 推荐商品 （模型输出）
    rec_items_score = []  # 推荐商品得分（模型输出）
    
    with torch.no_grad():
        for batch_idx, batched_data in enumerate(iter_data):
            num_sample += len(batched_data[0][config["USER_ID_FIELD"]])

            interaction, scores, positive_u, positive_i = eval_func(batched_data, model, device, tot_item_num)

            users_id = interaction[config["USER_ID_FIELD"]]  # 批次用户ID
            items_id_list = interaction[config["ITEM_ID_FIELD"] + config["LIST_SUFFIX"]] # 批次用户商品ID序列
            items_id = interaction[config["ITEM_ID_FIELD"]]  # 批次目标商品ID
            # 批次用户预测结果
            topk_values = (torch.topk(scores, k=800, dim=1)[0]).cpu().numpy()
            topk_idx = (torch.topk(scores, k=800, dim=1)[1]).cpu().numpy()
        
            # 用户、商品转换
            # 用户id转换
            for user_id in users_id:
                users.append(index2user[dataset_.field2id_token["user_id"][user_id]])
            # 目标商品转换
            for item_id in items_id:
                target_items.append(index2item[dataset_.field2id_token["item_id"][item_id]])
            # 用户商品id序列转化
            for i in range(items_id_list.shape[0]):
                T = []
                for j in range(items_id_list.shape[1]):
                    if items_id_list[i][j] != 0:
                        T.append(index2item[dataset_.field2id_token["item_id"][items_id_list[i][j]]])
                items_seq.append(T)
            # 预测商品id转换
            for i in range(topk_idx.shape[0]):
                T = []
                for j in range(topk_idx.shape[1]):
                    T.append(index2item[dataset_.field2id_token["item_id"][topk_idx[i][j]]])
                rec_items.append(T)
            # 预测商品得分
            rec_items_score.extend(topk_values.tolist())


    
    # 推荐商品按照三级品类打散, 同一品类下的商品至多保留 n 个，且均为目前在售商品
    prod_cate_info, valid_prod = read_from_odps()
    rec_items, rec_items_score = diversify_recommendations(rec_items, rec_items_score, prod_cate_info, valid_prod, max_per_cate=2, top_k=top_k)
    n = len(users)
    m = 0
    for i in range(len(users)):
        if target_items[i] in rec_items[i]:
            m += 1
    df = pd.DataFrame({
        "用户": users,
        "历史购买序列": items_seq,
        "目标商品": target_items,
        "推荐商品": rec_items,
        "推荐商品得分": rec_items_score
    })
    
    df.to_csv(os.path.join(result_save_path, "predict_result-{}-{}.csv".format(str(round(m/n, 4)), str(datetime.now().date()))), encoding='utf-8-sig', index=False)
    print("已保存到OSS")
    # 购买序列长度前150000的用户
    n_users = [user for user, seq in sorted(zip(users, items_seq), key=lambda x: len(x[1]), reverse=True)[:150000]]
    
    data = []
    rec_items_score = [[round(x, 6) for x in row] for row in rec_items_score]
    
    # 手动添加试验用户
    n_users.append('2a13f1cb662a4c44854ca823eaa6ec75')
    users.append('2a13f1cb662a4c44854ca823eaa6ec75') 
    i = users.index("a7f4dc8da5164063b9decf3aff6958f9")
    rec_items.append(rec_items[i])
    rec_items_score.append(rec_items_score[i])

    n_users = list(set(n_users))
     

    for user, items, scores in zip(users, rec_items, rec_items_score):
        if user not in n_users:
            continue
        for item, score in zip(items, scores):
            data.append([user, item, score])
    df1 = pd.DataFrame(data, columns=['user_id', 'prod_id', 'similarity_score'])
    write_to_odps(df1) 
    print("已保存到Dataworks")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='lianhua', help='dataset name')
    parser.add_argument('-fp', type=str, default='/ml/output/UniSRec-lianhua-finetuned.pth', help='finetuned model path')
    parser.add_argument('-t', type=str, default=50, help='top k scoring items')
    parser.add_argument('-sp', type=str, default='/ml/output/result/', help= 'result save path')
    args, unparsed = parser.parse_known_args()

    predictor(args.d, args.fp, args.t, args.sp)
