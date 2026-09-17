import argparse
import os
import re
from dotenv import dotenv_values
from odps import ODPS


def connect_odps(env_file):
    settings = dotenv_values(env_file)
    required = ('access_id', 'access_key', 'project', 'endpoint')
    missing = [name for name in required if not settings.get(name)]
    if missing:
        raise ValueError(f'Missing ODPS settings in {env_file}: {", ".join(missing)}')
    return ODPS(settings['access_id'], settings['access_key'], settings['project'],
                endpoint=settings['endpoint'])


def get_df_from_odps(sql, saved_path, env_file, dataset='dataset'):
    odps = connect_odps(env_file)

    # 读取商品属性数据
    query_job = odps.execute_sql(sql)
    prod_attr_df = query_job.open_reader(tunnel=True)
    prod_attr_df = prod_attr_df.to_pandas(n_process=8)
    os.makedirs(saved_path, exist_ok=True)
    csv_file_path = os.path.join(saved_path, f'{dataset}.csv')
    prod_attr_df.to_csv(csv_file_path, index=False, encoding='utf-8')
    return csv_file_path


def export_interactions(table, saved_path, env_file, dataset, query_file=None):
    if query_file:
        with open(query_file, encoding='utf-8') as file:
            sql = file.read()
    else:
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_.]*', table):
            raise ValueError('ODPS table name must contain only letters, digits, underscores, or dots')
        sql = f'SELECT * FROM {table}'
    return get_df_from_odps(sql, saved_path, env_file, dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='outputs/raw/')
    parser.add_argument('--env-file', default='.env')
    parser.add_argument('--dataset', required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input-table')
    source.add_argument('--input-query-file')
    args = parser.parse_args()
    export_interactions(args.input_table, args.output_dir, args.env_file, args.dataset, args.input_query_file)
