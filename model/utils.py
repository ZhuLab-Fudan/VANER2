import torch
import numpy as np
POS_CONST = 1000000

def save_predictions(raw_texts, predictions, save_path, with_score = False):
    # Annotation format: [start, end, score, linking]
    file = open(save_path, "w", encoding='utf-8')
    for i in range(len(raw_texts)):
        for line in raw_texts[i]:
            file.write(line.strip('\n') + '\n')
        doc_id = raw_texts[i][0].split('|')[0]
        text = raw_texts[i][0].split('|t|')[-1].strip('\n') + ' ' + raw_texts[i][1].split('|a|')[-1].strip('\n')
        for type in predictions:
            for anno in predictions[type][i]:
                linking = anno[3] if len(anno) >= 4 else 'None'
                out_str = doc_id + '\t' + str(anno[0]) + '\t' + str(anno[1]) + '\t' + text[anno[0]:anno[1]] + '\t' + type + '\t' + linking
                file.write(out_str + '\t' + str(anno[2]) + '\n' if with_score else out_str + '\n')
        file.write('\n')

# Post processing to remove leading or trailing special characters in tokens and remove weak predictions
# prediction format: [[start, end, score, (optional)linking]]
def post_process(raw_text, predictions, dataset, score_threshold = 0):
    text = ''
    for item in raw_text:
        text += item.split('|a|')[1] if '|a|' in item else item.split('|t|')[1]

    strip_list = [' ', '\n', ',', '.', '-']
    opposite = {'(': ')', ')': '(', '[': ']', ']': '[', '{': '}', '}': '{'}

    # remove special characters
    new_predictions = []
    for item in predictions:
        if item[1] > len(text):
            item[1] = len(text)
        if item[0] >= item[1]:
            item[0] = item[1] - 1
        while item[0] < item[1] and text[item[0]] in strip_list:
            item[0] = item[0] + 1
        while item[0] < item[1] and text[item[0]] in opposite and opposite[text[item[0]]] not in text[item[0]:item[1]]:
            item[0] = item[0] + 1
        while item[0] < item[1] and text[item[1] - 1] in strip_list:
            item[1] = item[1] - 1
        while item[0] < item[1] and text[item[1] - 1] in opposite and opposite[text[item[1] - 1]] not in text[item[0]:item[1]]:
            item[1] = item[1] - 1

        # ignore certain words in certain dataset
        if 'medmentions' in dataset.lower() and 'patient' in text[item[0]:item[1]].lower():
            continue
        new_predictions.append(item)

    # Merge same intervals
    sorted_predictions = sorted(new_predictions, key = lambda x: x[0] * POS_CONST + x[1])
    new_predictions = []
    for item in sorted_predictions:
        if new_predictions != []:
            if item[0] == new_predictions[-1][0] and item[1] == new_predictions[-1][1]:
                new_predictions[-1][2] += item[2]
                continue
        new_predictions.append(item)

    # Filter predictions based on threshold
    predictions = new_predictions
    new_predictions = []
    for item in predictions:
        if item[2] > score_threshold:
            new_predictions.append(item)

    return new_predictions

def check_gpu_memory(target_gpu):
    import nvidia_smi
    nvidia_smi.nvmlInit()
    ngpu = nvidia_smi.nvmlDeviceGetCount()
    if ngpu > 0:
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(target_gpu)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    return info.used

def get_bert_base_params(name):

    medium_names = ['bert-base-multilingual-cased-ner-hrl','bert-base-NER','bert-italian-cased-ner','bert-italian-finetuned-ner','bioBIT',
                    'bert-spanish-cased-finetuned-ner','BioLinkBERT-base','bsc-bio-ehr-es','roberta-base-biomedical-clinical-es',
                    'roberta-es-clinical-trials-ner','NuNER-multilingual-v0.1','BiomedBERT-base-uncased-abstract-fulltext',
                    'BiomedNLP-BiomedBERT-base-uncased-abstract']
    special_names = ['NuNER-v2.0', 'mdeberta-v3-base']
    large_names = ['xlm-roberta-large-english-clinical','xlm-roberta-large-spanish-clinical']

    if name == 'ModernBERT-large':
        cxt_len = 4096
        embed_size = 1024
        batch_size = 1
        grad_acc_steps = 8
    elif name == 'ModernBERT-base':
        cxt_len = 4096
        embed_size = 768
        batch_size = 4
        grad_acc_steps = 2
    elif name in large_names:
        cxt_len = 512
        embed_size = 1024
        batch_size = 1
        grad_acc_steps = 8
    elif name in medium_names:
        cxt_len = 512
        embed_size = 768
        batch_size = 16
        grad_acc_steps = 1
    elif name in special_names:
        cxt_len = 512
        embed_size = 768
        batch_size = 4
        grad_acc_steps = 4
    else:
        raise TypeError('Unknown BERT model!')

    return cxt_len, embed_size, batch_size, grad_acc_steps

def get_llm_base_params(name):

    cxt_len = 512
    if name == 'Meta-Llama-3.2-1B-Instruct':
        embed_size = 2048
        batch_size = 4
        grad_acc_steps = 2
    elif name == 'Meta-Llama-3.2-3B-Instruct':
        embed_size = 3072
        batch_size = 1
        grad_acc_steps = 8
    elif name in ['MMedS-Llama-3-8B','Meta-Llama-3.1-8B-Instruct']:
        embed_size = 4096
        batch_size = 1
        grad_acc_steps = 8
    else:
        raise TypeError('Unknown LLM model!')

    return cxt_len, embed_size, batch_size, grad_acc_steps

def get_special_token_ids(tokenizer, model_type, model_name):
    if model_type == 'bert':
        if model_name in ['NuNER-v2.0','xlm-roberta-large-english-clinical', 'xlm-roberta-large-spanish-clinical']:
            pad_id = tokenizer('<pad>', add_special_tokens=False).data['input_ids']
            sep_id = tokenizer('</s>', add_special_tokens=False).data['input_ids']
            unk_id = tokenizer('<unk>', add_special_tokens=False).data['input_ids']
            mask_id = tokenizer('<mask>', add_special_tokens=False).data['input_ids']
        else:
            pad_id = tokenizer('[PAD]', add_special_tokens=False).data['input_ids']
            sep_id = tokenizer('[SEP]', add_special_tokens=False).data['input_ids']
            unk_id = tokenizer('[UNK]', add_special_tokens=False).data['input_ids']
            mask_id = tokenizer('[MASK]', add_special_tokens=False).data['input_ids']
    else:
        pad_id = tokenizer('<|finetune_right_pad_id|>', add_special_tokens=False).data['input_ids']
        sep_id = tokenizer('\n', add_special_tokens=False).data['input_ids']
        unk_id = tokenizer('<|reserved_special_token_1|>', add_special_tokens=False).data['input_ids']
        mask_id = tokenizer('<|reserved_special_token_2|>', add_special_tokens=False).data['input_ids']

    assert len(pad_id) == 1 and len(sep_id) == 1 and len(unk_id) == 1 and len(mask_id) == 1
    return pad_id[0], sep_id[0], unk_id[0], mask_id[0]

# Convert bio predictions to text intervals
# Input: [[tag, score, token_start_pos]], One more empty token should be added at the end
# Output: [[start, end, score]]
def bio2brat(tags, use_bioe):
    tags = sorted(tags, key=lambda x: x[2])
    result_annos = []
    last_j = None
    score = 0
    score_cnt = 0
    for j in range(len(tags)):
        if last_j is not None:
            end = None
            if use_bioe and tags[j][0] == 3:
                end = j + 1
            elif tags[j][0] in [0, 2]:
                end = j
            if end is not None and end < len(tags):
                result_annos.append([tags[last_j][2], tags[end][2], score / score_cnt])
                score = 0
                score_cnt = 0
                last_j = None
        if tags[j][0] == 2:
            last_j = j
            score += tags[j][1]
            score_cnt += 1
        elif tags[j][0] == 1:
            score += tags[j][1]
            score_cnt += 1

    return result_annos