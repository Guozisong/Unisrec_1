"""Build model item sequences from current user interaction history."""

import csv
import datetime


def parse_event_time(value):
    if len(value) == 13 and value.isdigit():
        return int(value)
    return int(datetime.datetime.strptime(value, '%Y-%m-%d').timestamp() * 1000)


def load_prediction_histories(source, item_ids, max_seq_length):
    if max_seq_length < 1:
        raise ValueError('max_seq_length must be positive')
    histories = {}
    skipped = 0
    with open(source, newline='', encoding='utf-8-sig') as incoming:
        reader = csv.DictReader(incoming)
        if not {'user_id', 'item_id', 'event_time'}.issubset(reader.fieldnames or ()):
            raise ValueError('Prediction CSV needs columns: user_id, item_id, event_time')
        for row in reader:
            user_id, item_id, event_time = (row[name] for name in ('user_id', 'item_id', 'event_time'))
            if not user_id or item_id not in item_ids or not event_time:
                skipped += 1
                continue
            try:
                timestamp = parse_event_time(event_time)
            except ValueError:
                skipped += 1
                continue
            histories.setdefault(user_id, []).append((timestamp, item_ids[item_id]))

    if not histories:
        raise ValueError('No valid prediction histories contain known items')
    sequences = [
        (user_id, [item_id for _, item_id in sorted(events, key=lambda event: event[0])][-max_seq_length:])
        for user_id, events in histories.items()
    ]
    return sequences, skipped
