"""Normalize an interaction source to the CSV contract consumed by preprocessing."""

import argparse
import csv
import os


FIELDS = ('user_id', 'item_id', 'event_value', 'event_time', 'item_text')
LEGACY_FIELDS = ('user_id', 'prod_id', 'purchase_count', 'dt', 'attrvalues')


def prepare_csv(source, destination):
    destination = os.fspath(destination)
    with open(source, newline='', encoding='utf-8-sig') as incoming:
        reader = csv.DictReader(incoming)
        columns = reader.fieldnames or ()
        if set(FIELDS).issubset(columns):
            source_fields = FIELDS
        elif set(LEGACY_FIELDS).issubset(columns):
            source_fields = LEGACY_FIELDS
        else:
            raise ValueError(f'Interaction CSV needs columns: {", ".join(FIELDS)}')
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        staged = destination + '.tmp'
        with open(staged, 'w', newline='', encoding='utf-8') as outgoing:
            writer = csv.DictWriter(outgoing, fieldnames=FIELDS, lineterminator='\n')
            writer.writeheader()
            for row in reader:
                writer.writerow({field: row[source_field] for field, source_field in zip(FIELDS, source_fields)})
        os.replace(staged, destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output-dir', required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input-csv')
    source.add_argument('--input-table')
    source.add_argument('--input-query-file')
    parser.add_argument('--env-file', default='.env')
    args = parser.parse_args()

    output = os.path.join(args.output_dir, f'{args.dataset}.csv')
    if args.input_table or args.input_query_file:
        from get_data_from_odps import export_interactions
        exported = export_interactions(args.input_table, args.output_dir, args.env_file,
                                       args.dataset + '.source', args.input_query_file)
        try:
            prepare_csv(exported, output)
        finally:
            os.unlink(exported)
    else:
        prepare_csv(args.input_csv, output)
