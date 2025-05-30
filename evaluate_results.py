import os
import copy
import torch
from tqdm import tqdm
import argparse
import sys
import shutil
from datetime import datetime
import numpy as np
from model.utils import POS_CONST

def read_pub(file):
    f = open(file, 'r', encoding='utf-8')
    content = f.readlines()
    f.close()

    all_text = []
    all_annos = []
    all_pmid = []
    pos = 0
    while pos < len(content):
        if '|t|' in content[pos]:
            pmid = content[pos].split('|t|')[0]
            if '|' in pmid:
                pmid = pmid.split('|')[1]
            text = content[pos].split('|t|')[-1] + content[pos+1].split('|a|')[-1]
            pos += 2
            annotations = []
            while pos < len(content) and content[pos] != '\n':
                items = content[pos].split('\t')
                if len(items) == 5:
                    items.append('None')
                if items[1] != 'React':
                    annotations.append([int(items[1]), int(items[2]), items[3], items[4].strip('\n'), items[5].strip('\n')])
                pos += 1
            all_pmid.append(pmid)
            all_text.append(text)
            all_annos.append(annotations)
        else:
            pos += 1

    return all_pmid, all_text, all_annos

# Input two list of annotations sets and calculate METRICS
def count_tp(all_doc_annos, all_doc_gold_annos, select_type = None, evaluate_NEN = False):
    total_NER_tp, total_NER_fp, total_NER_fn = 0, 0, 0
    total_NEN_tp, total_NEN_fp, total_NEN_fn = 0, 0, 0
    nwrong = 0
    nmiss = 0
    nmore = 0
    nless = 0
    noverlap = 0

    for annos, gold_annos in zip(all_doc_annos,all_doc_gold_annos):

        new_annos = []
        for anno in annos:
            if select_type is None or anno[3] == select_type:
                new_annos.append(anno)
        annos = new_annos

        new_gold_annos = []
        for anno in gold_annos:
            if select_type is None or anno[3] == select_type:
                new_gold_annos.append(anno)
        gold_annos = new_gold_annos

        NER_tp, NEN_tp = 0, 0
        annos = sorted(annos, key = lambda x:x[0] * POS_CONST + x[1])
        gold_annos = sorted(gold_annos, key=lambda x: x[0] * POS_CONST + x[1])

        if evaluate_NEN:
            NEN_tot_annos = 0
            NEN_tot_gold_annos = 0
            for item in annos:
                if len(item) >= 5 and item[4] != 'None':
                    NEN_tot_annos += 1
            for item in gold_annos:
                if len(item) >= 5 and item[4] != 'None':
                    NEN_tot_gold_annos += 1

        pos = 0
        temp = [0 for i in range(len(gold_annos))]
        for anno in annos:
            while pos < len(gold_annos) and (gold_annos[pos][0] * POS_CONST + gold_annos[pos][1] < anno[0] * POS_CONST + anno[1]):
                pos += 1
            if pos < len(gold_annos) and gold_annos[pos][0] == anno[0] and gold_annos[pos][1] == anno[1]:
                NER_tp += 1
                temp[pos] = 1

                if evaluate_NEN:
                    if len(anno) >= 5 and len(gold_annos[pos]) >= 5:
                        if anno[4] != 'None' and gold_annos[pos][4] != 'None':
                            if anno[4] == gold_annos[pos][4]:
                                NEN_tp += 1
            else:
                # find the interval in labels that overlaps the most with current interval
                if pos < len(gold_annos):
                    comp_pos = pos
                    overlap = np.minimum(gold_annos[comp_pos][1], anno[1]) - np.maximum(gold_annos[comp_pos][0],anno[0])
                    if np.minimum(gold_annos[pos - 1][1], anno[1]) - np.maximum(gold_annos[pos - 1][0],anno[0]) > overlap:
                        comp_pos = pos - 1
                else:
                    comp_pos = pos - 1

                if comp_pos < 0 or comp_pos >= len(gold_annos):
                    continue

                overlap = np.minimum(gold_annos[comp_pos][1], anno[1]) - np.maximum(gold_annos[comp_pos][0], anno[0])
                if overlap <= 0:
                    nwrong += 1
                else:
                    temp[comp_pos] = 1
                    if overlap == anno[1] - anno[0]:
                        nless += 1
                    elif overlap == gold_annos[comp_pos][1] - gold_annos[comp_pos][0]:
                        nmore += 1
                    else:
                        noverlap += 1

        nmiss += len(gold_annos) - np.sum(temp)
        total_NER_tp += NER_tp
        total_NER_fp += len(annos) - NER_tp
        total_NER_fn += len(gold_annos) - NER_tp
        if evaluate_NEN:
            total_NEN_tp += NEN_tp
            total_NEN_fp += NEN_tot_annos - NEN_tp
            total_NEN_fn += NEN_tot_gold_annos - NEN_tp

    all_NER_results = np.array([total_NER_tp, total_NER_fp, total_NER_fn, nwrong, nmiss, nmore, nless, noverlap])
    all_NEN_results = np.array([total_NEN_tp, total_NEN_fp, total_NEN_fn])

    return all_NER_results, all_NEN_results if evaluate_NEN else None

def prf_metrics(total_tp, total_fp, total_fn):
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f_score = 2 * precision * recall / (precision + recall + 1e-6)
    return precision, recall, f_score

def save_metrics(save_path, all_metrics):
    file = open(save_path, "w")
    for dataset in all_metrics:
        cur_metrics = all_metrics[dataset]
        temp_metrics = np.zeros(6)
        for type in cur_metrics:
            temp_metrics += cur_metrics[type][:6]
            line1 = "Result of {} entities in {} dataset:\n".format(type, dataset)
            line2 = "Precision: {:.4f}  Recall: {:.4f}  F-score: {:.4f}\n".format(*list(cur_metrics[type][0:3]))
            file.write(line1)
            file.write(line2)

            if len(cur_metrics[type]) == 11:
                line3 = "Nwrong: {:.0f}  Nmiss: {:.0f}  Nmore: {:.0f}  Nless: {:.0f}  Noverlap: {:.0f}\n\n".format(
                    *list(cur_metrics[type][6:11]))
                file.write(line3)

        precision, recall, f_score = prf_metrics(*list(temp_metrics[3:]))
        line1 = "Micro of {} dataset:\n".format(dataset)
        line2 = "Precision: {:.4f}  Recall: {:.4f}  F-score: {:.4f}\n".format(precision, recall, f_score)
        file.write(line1)
        file.write(line2)

        line1 = "Macro of {} dataset:\n".format(dataset)
        line2 = "Precision: {:.4f}  Recall: {:.4f}  F-score: {:.4f}\n".format(*list(temp_metrics[:3]/len(cur_metrics)))
        file.write(line1)
        file.write(line2)
        file.write('__________________________________________________\n')

def summarize(base_path, input_file, output_file):
    entity_types = ['Gene', 'Chemical', 'Disease', 'Species', 'CellLine', 'Variant', 'Anatomy', 'CellType', 'CellComponent']
    outfile = open(output_file, 'w', encoding='utf-8')

    models = []
    for type in entity_types:
        datasets = []
        all_scores = []
        for folder in os.listdir(base_path):
            if os.path.exists(base_path + folder + '/' + input_file):
                models.append(folder)
                scores = {}
                infile = open(base_path + folder + '/' + input_file, 'r', encoding='utf-8')
                lines = infile.readlines()
                for i in range(len(lines)):
                    if ('of ' + type + ' entities') in lines[i]:
                        if lines[i].split(' ')[-2] not in datasets:
                            datasets.append(lines[i].split(' ')[-2])
                        scores[lines[i].split(' ')[-2]] = (float(lines[i + 1].split(':')[-1].strip('\n')))
                all_scores.append(scores)

        outfile.write(type + '\t' + '\t'.join(datasets) + '\taverage\n')
        for model, scores in zip(models, all_scores):
            outfile.write(model + '\t')
            avg = 0
            cnt = 0
            for dataset in datasets:
                if dataset not in scores:
                    outfile.write('-\t')
                else:
                    avg += scores[dataset]
                    cnt += 1
                    outfile.write('{:.4f}'.format(scores[dataset]) + '\t')
            outfile.write('{:.4f}'.format(avg / (cnt + 1e-6)) + '\n')
        outfile.write('\n')

def parse_arguments():
    # main args
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", type=str,
                        default='Converted_New')  # dataset folder to evaluate, if None, use the test set of the trained dataset
    parser.add_argument("--evaluate_NEN", type=str,
                        default='False')  # dataset folder to evaluate, if None, use the test set of the trained dataset
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_arguments()
    evaluate_NEN = args.evaluate_NEN.lower() == 'true'
    dataset_folder = args.dataset_folder
    base_path = './results/' + dataset_folder + '/'
    gold_annos_path = './data/' + dataset_folder + '/test/'

    # Automatically Evaluate all model predictions for each dataset
    for name in os.listdir(path=base_path):
        preds_path = base_path + name + '/predictions/'
        all_NER_metrics = {}
        all_NEN_metrics = {}

        if os.path.exists(preds_path):
            print('Evaluating results in ' + preds_path)
            for file_name in os.listdir(preds_path):
                dataset_name = file_name.split('.')[0]
                all_NER_metrics[dataset_name] = {}
                all_NEN_metrics[dataset_name] = {}

            for file_name in os.listdir(preds_path):
                dataset_name = file_name.split('.')[0]
                all_pmid, all_text, all_doc_annos = read_pub(preds_path + file_name)
                _, _, all_doc_gold_annos = read_pub(gold_annos_path + dataset_name + '.pubtator')

                # get all entity types in dataset
                all_types = []
                for doc in all_doc_gold_annos:
                    for anno in doc:
                        if anno[3].strip('\n') not in all_types:
                            all_types.append(anno[3].strip('\n'))

                # match results for each type
                for type in all_types:
                    all_NER_results, all_NEN_results = count_tp(all_doc_annos, all_doc_gold_annos, type, evaluate_NEN)
                    all_NER_metrics[dataset_name][type] = np.concatenate(
                        (prf_metrics(*list(all_NER_results[:3])), all_NER_results), 0)
                    if evaluate_NEN:
                        all_NEN_metrics[dataset_name][type] = np.concatenate(
                            (prf_metrics(*list(all_NEN_results)), all_NEN_results), 0)

        save_metrics(base_path + name + '/all_NER_scores.txt', all_NER_metrics)
        if evaluate_NEN:
            save_metrics(base_path + name + '/all_NEN_scores.txt', all_NEN_metrics)

    # Make a table to summarize results
    print('Making results table...')
    summarize(base_path, 'all_NER_scores.txt', 'NER_summary.tsv')
    if evaluate_NEN:
        summarize(base_path, 'all_NEN_scores.txt', 'NEN_summary.tsv')