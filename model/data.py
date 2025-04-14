import glob
import re
import pandas as pd
import numpy as np
import torch
import random
import os
from transformers import AutoTokenizer
if __name__ != '__main__':
    from model.utils import *

def tokenize(text, annos, tokenizer, label2id, use_bioe):
    global anno_type

    # tokenize all entities and text segments between entities and assign tags to them
    annos = sorted(annos, key=lambda x: x[0] * POS_CONST - x[1])

    # convert annotation to BIO format (B=2, I=1, O=0), optional E=3
    result = tokenizer(text, add_special_tokens=False)
    tokens = result.data['input_ids']

    document_pos = [result.token_to_chars(j).start for j in range(len(tokens))]
    document_pos.append(len(text))

    all_type_tags = {}
    for item in annos:
        if item[2] not in all_type_tags:
            all_type_tags[item[2]] = [label2id['O'] for i in range(len(tokens))]
        try:
            id = np.searchsorted(document_pos, item[0], side="right") - 1
            if all_type_tags[item[2]][id] == label2id['O']:
                all_type_tags[item[2]][id] = label2id['B']
                while document_pos[id + 1] < item[1]:
                    id = id + 1
                    all_type_tags[item[2]][id] = label2id['I']
                if all_type_tags[item[2]][id] == label2id['I'] and use_bioe:
                    all_type_tags[item[2]][id] = label2id['E']
        except:
            print(item)

    return tokens, all_type_tags, document_pos

def read_data(path, mode = 'eval_text', tokenizer = None, label2id = None, sep_id = None, use_bioe = False, verbose = False):
    all_data = []
    filelist = glob.glob(path + '**/*.pubtator', recursive=True) if os.path.isdir(path) else glob.glob(path)
    for file in filelist:
        if verbose:
            print('Found data: ', file)

        dataset = file.split('/')[-1].split('.')[0]
        lines = open(file, "r", encoding="utf-8").readlines()

        all_tokens = []
        full_all_type_tags = []
        all_raw_texts = []
        all_types = []
        all_annotations = []
        all_start_pos = []

        pos = 0
        while pos < len(lines):
            if '|t|' in lines[pos]:
                text = lines[pos].split('|t|')[1] + lines[pos + 1].split('|a|')[1].strip('\n')
                raw_text = [lines[pos],lines[pos + 1]]
                pos += 2

                annotations = []
                while pos < len(lines) and lines[pos] != '\n':
                    items = lines[pos].split('\t')
                    annotations.append((int(items[1]), int(items[2]), items[4].strip('\n'), items[3]))
                    if items[4].strip('\n') not in all_types:
                        all_types.append(items[4].strip('\n'))
                    pos += 1

                all_raw_texts.append(raw_text)
                all_annotations.append(annotations)

                if mode != 'eval_text':
                    tokens, all_type_tags, start_pos = tokenize(text, annotations, tokenizer, label2id, use_bioe)
                    all_tokens.append(tokens)
                    full_all_type_tags.append(all_type_tags)
                    all_start_pos.append(start_pos)

            pos += 1

        if mode == 'inference':
            all_types = ['Gene','Chemical','Species','Disease','Variant','CellLine','CellType','Anatomy','CellComponent']

        if mode == 'train':
            # concatenate documents
            new_tokens = []
            all_type_tags = {type: [] for type in all_types}
            for doc_tokens, doc_tags in zip(all_tokens, full_all_type_tags):
                new_tokens += doc_tokens + [sep_id]
                for type in all_types:
                    if type in doc_tags:
                        all_type_tags[type] += doc_tags[type] + [label2id['O']]
                    else:
                        all_type_tags[type] += [label2id['O'] for i in range(len(doc_tokens))] + [label2id['O']]

            for type in all_type_tags:
                all_type_tags[type] = torch.tensor(all_type_tags[type])
            all_data.append({'tokens': torch.tensor(new_tokens), 'tags': all_type_tags, 'dataset': dataset})
        elif mode == 'eval_text':
            all_data.append({'dataset': dataset, 'raw_text': all_raw_texts, 'ent_types': all_types})
        elif mode in ['eval', 'inference']:
            all_data.append({'tokens': all_tokens, 'token_start_pos': all_start_pos, 'dataset': dataset, 'raw_text': all_raw_texts, 'ent_types': all_types})
        else:
            raise TypeError('Unknown argument mode in function read data!')

    return all_data

def get_batch(data_dict, prompt_tokens, batch_size, cxt_len, data_use_percentage):
    batch = []
    all_types_l_batch = {}

    weights = [data['tokens'].size(0) for data in data_dict]
    dataset_id = random.choices([i for i in range(len(data_dict))], weights=weights, k=1)[0]
    for i in range(batch_size):
        data = data_dict[dataset_id]
        if prompt_tokens is None:
            real_len = cxt_len
            begin = torch.randint(0, int(data['tokens'].size(0) * data_use_percentage) - real_len, (1,))
            item = data['tokens'][begin:begin + real_len]
        else:
            real_len = cxt_len - prompt_tokens.size(0)
            begin = torch.randint(0, int(data['tokens'].size(0) * data_use_percentage) - real_len, (1,))
            item = torch.cat((prompt_tokens, data['tokens'][begin:begin + real_len]), 0)

        batch.append(item)
        for type in data['tags']:
            if type not in all_types_l_batch:
                all_types_l_batch[type] = []
            if prompt_tokens is None:
                all_types_l_batch[type].append(data['tags'][type][begin:begin + real_len])
            else:
                all_types_l_batch[type].append(torch.cat((torch.zeros_like(prompt_tokens) - 1, data['tags'][type][begin:begin + real_len]), 0))

    batch = torch.stack(batch, 0).cuda()
    for type in all_types_l_batch:
        all_types_l_batch[type] = torch.stack(all_types_l_batch[type], 0).cuda()

    return batch, all_types_l_batch, data_dict[dataset_id]['dataset']