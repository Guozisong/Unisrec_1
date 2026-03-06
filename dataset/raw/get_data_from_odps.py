import os
from odps import ODPS
from dotenv import load_dotenv


def get_df_from_odps(sql, saved_path):
    
    load_dotenv("/ml/output/.env")
    access_id = os.getenv("ALI_ACCESS_ID")
    access_key = os.getenv("ALI_SECRET_ACCESS_KEY")
    project = 'jxaidataworks'
    endpoint = 'https://service.cn-hangzhou-vpc.maxcompute.aliyun-inc.com/api'

    # 初始化ODPS对象
    odps = ODPS(access_id, access_key, project, endpoint)

    # 读取商品属性数据
    query_job = odps.execute_sql(sql)
    prod_attr_df = query_job.open_reader(tunnel=True)
    prod_attr_df = prod_attr_df.to_pandas(n_process=8)
    csv_file_path = os.path.join(saved_path, "lianhua.csv")
    prod_attr_df.to_csv(csv_file_path, index=False, encoding='utf-8')


if __name__ == '__main__':
    sql = '''select user_id, prod_id, purchase_count, dt, attrvalues
             from unisrec_raw_data
             where dt between TO_CHAR(DATEADD(GETDATE(), -91, 'dd'), 'yyyy-MM-dd')
                      AND TO_CHAR(DATEADD(GETDATE(), -1, 'dd'), 'yyyy-MM-dd')
             ;
          '''
    saved_path = "/ml/output/"
    get_df_from_odps(sql, saved_path)