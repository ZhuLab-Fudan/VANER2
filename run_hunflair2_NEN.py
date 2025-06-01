import os
import argparse
import time
import torch
import flair
import glob
from tqdm import tqdm
from typing import Dict, List, Tuple
from bioc import pubtator
from flair.data import Sentence
from flair.models.entity_mention_linking import EntityMentionLinker

def run_nen(
    linkers: Dict[str, EntityMentionLinker],
    all_ids: List,
    all_texts: List,
    all_annotations: List,
    batch_size: int,
):
    print("Start entity linking")
    start = time.time()

    # Replace certrain characters Hunflair2 cannot process
    for i in range(len(all_texts)):
        all_texts[i] = all_texts[i].replace('_','-').replace(' \n', '. ')
    texts = [Sentence(text) for text in all_texts]

    linker = list(linkers.values())[0]
    linker.preprocessor.initialize(texts)

    for other_linker in list(linkers.values())[1:]:
        other_linker.preprocessor.abbreviation_dict = (
            linker.preprocessor.abbreviation_dict
        )

    print('Preprocessing annotations...')
    all_signle_annos = {}
    all_signle_annos_text = {}
    new_annos = {}
    for i in tqdm(range(len(all_annotations))):
        new_annos[all_ids[i]] = []
        for anno in all_annotations[i]:
            if anno[3] not in all_signle_annos:
                all_signle_annos[anno[3]] = []
                all_signle_annos_text[anno[3]] = []
            all_signle_annos[anno[3]].append((anno, all_ids[i]))
            processed = linker.preprocessor.process_mention(entity_mention=anno[2].replace('_','-'), sentence=texts[i])
            all_signle_annos_text[anno[3]].append(processed)

    print('Running linking models...')
    for ent_type in all_signle_annos:
        if ent_type in linkers:
            linker = linkers[ent_type]
            annos_cnt = len(all_signle_annos[ent_type])
            for batch_start in tqdm(range(0,annos_cnt,batch_size)):

                batch_annos = all_signle_annos[ent_type][batch_start: batch_start + batch_size]
                batch_annos_text = all_signle_annos_text[ent_type][batch_start: batch_start + batch_size]

                batch_candidates = linker.candidate_generator.search(
                    entity_mentions = batch_annos_text, top_k=1
                )

                for (anno, doc_id), mention_candidates in zip(batch_annos, batch_candidates):
                    if len(mention_candidates) > 0:
                        top_candidate = mention_candidates[0]
                        anno[4] = top_candidate[0]
                    else:
                        anno[4] = 'None'
                    new_annos[doc_id].append(anno)
        else:
            for (anno, doc_id) in all_signle_annos[ent_type]:
                anno[4] = 'None'
                new_annos[doc_id].append(anno)

    elapsed = round(time.time() - start, 2)
    print(f"Entity linking took: {elapsed}s")

    return new_annos

def read_pub(file):
    f = open(file, 'r', encoding='utf-8')
    content = f.readlines()
    f.close()

    all_texts = []
    all_annos = []
    all_lines = []
    all_ids = []
    pos = 0
    while pos < len(content):
        if '|t|' in content[pos]:
            all_ids.append(content[pos].split('|')[0])
            all_lines.append(content[pos] + content[pos + 1])

            text = content[pos].split('|t|')[-1] + content[pos+1].split('|a|')[-1].strip('\n')
            all_texts.append(text)

            pos += 2
            annotations = []
            while pos < len(content) and content[pos] != '\n':
                items = content[pos].split('\t')
                if len(items) == 5:
                    items.append('None')
                if items[1] != 'React':
                    annotations.append([int(items[1]), int(items[2]), items[3], items[4].strip('\n'), items[5].strip('\n')])
                pos += 1
            all_annos.append(annotations)
        else:
            pos += 1

    return all_ids, all_lines, all_texts, all_annos

def save2file(all_ids, all_lines, all_annos, save_name):
    file = open(save_name, 'w')
    for doc_id, lines in zip(all_ids, all_lines):
        file.write(lines)
        for anno in all_annos[doc_id]:
            file.write(doc_id + '\t' + '\t'.join([str(x) for x in anno]) + '\n')
        file.write('\n')

def parse_arguments():
    # main args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_names", nargs="*", type=str,
                        default=None)
    parser.add_argument("--dataset_folder", type=str,
                        default='Converted_New')  # dataset folder to evaluate, if None, use the test set of the trained dataset
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    flair.device = torch.device('cuda:0')

    args = parse_arguments()
    base_path = './results/' + args.dataset_folder + '/'
    entity_types = ['Gene', 'Chemical', 'Disease', 'Species']

    print(f"Loading entity linking models: {entity_types}")

    linkers = {et: EntityMentionLinker.load(f"{et.lower()}-linker") for et in entity_types}

    # linker_path = './Hunflair2_pretrained_models/'
    # linkers = {et: EntityMentionLinker.load(linker_path + f"{et.lower()}_linker/pytorch_model.bin") for et in entity_types}

    model_names = args.model_names
    if model_names is None:
        model_names = [name for name in os.listdir(path=base_path)]

    # Automatically do NEN based on NER results
    for name in model_names:
        file_path = base_path + name + '/predictions/'
        if os.path.exists(file_path):
            for file_name in glob.glob(file_path + '*.pubtator'):
                print('filepath:', file_path)
                all_ids, all_lines, all_texts, all_annos = read_pub(file_path + file_name)
                all_annos = run_nen(linkers, all_ids, all_texts, all_annos, batch_size = 128)

                # Save NEN results
                save2file(all_ids, all_lines, all_annos, file_path + file_name)

