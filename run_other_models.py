import bisect
import copy
import re
import numpy as np
from model.data import *
from model.utils import *
from tqdm import tqdm
import os
import math
import sys
import argparse
import time
import shutil
temp_num = 0

def Hunflair2_predict(tagger, text, offsets, doc_ids):
    from flair.data import Sentence
    from flair.tokenization import SciSpacyTokenizer
    sentence = Sentence(text, use_tokenizer=SciSpacyTokenizer())
    tagger.predict(sentence)

    predictions = []
    for entity in sentence.get_spans('ner'):
        doc_pos = bisect.bisect_right(offsets, entity.start_position) - 1
        predictions.append((entity.tag, 'None', doc_ids[doc_pos], entity.start_position - offsets[doc_pos], entity.end_position - offsets[doc_pos], entity.score))
    return predictions

def BERN2_predict(text, offsets, doc_ids):
    global temp_num
    import requests

    headers = {}
    map_dict = {'gene': 'Gene', 'drug': 'Chemical', 'species': 'Species', 'disease': 'Disease', 'mutation': 'Variant', 'cell_line': 'CellLine', 'cell_type': 'CellType'}

    splitted_texts = []
    splitted_offsets = []

    occurs = [m.start() for m in re.finditer(re.escape(' '), text)]
    last_pos = 0
    for i in range(len(occurs)):
        if occurs[i] - last_pos > 4000:
            splitted_texts.append(text[last_pos:occurs[i]])
            splitted_offsets.append(last_pos)
            last_pos = occurs[i]
    splitted_texts.append(text[last_pos:])
    splitted_offsets.append(last_pos)

    predictions = []
    for split_text, split_offset in zip(splitted_texts, splitted_offsets):
        assert len(split_text) <= 5000
        retry_num = 0
        max_retry_num = 5
        while (retry_num < max_retry_num):
            try:
                reply = requests.post("http://bern2.korea.ac.kr/plain", headers=headers, json={'text': split_text}, timeout=60)
                reply = reply.json()
                break
            except:
                # BERN2 server seems to block text with certain SQL characters, so replace them if fail
                print("Error in Reply! Retrying...")
                split_text = split_text.replace('select', 'choose')
                split_text = split_text.replace('group', 'merge')
                text = text.replace('having', ' have ')
                split_text = split_text.replace('into', ' in ')
                split_text = split_text.replace('=', '-')
                split_text = split_text.replace('?', '.')
                split_text = split_text.replace('(', '/')
                split_text = split_text.replace(')', '/')
                split_text = split_text.replace('\'', ' ')
                split_text = split_text.replace('PS', '**')
                retry_num += 1
        if retry_num >= max_retry_num:
            print("Submission failed!")
            file = open('BERN2_error' + str(temp_num) + '.txt', 'w')
            file.write(split_text)
            temp_num += 1
            continue

        processed_annos = []
        for anno in reply['annotations']:
            occurs = [m.start() for m in re.finditer(re.escape(anno['mention']), split_text)]
            if occurs != []:
                occurs = sorted(occurs, key=lambda x: abs(x - anno['span']['begin']))
                for occur in occurs:
                    start = occur
                    end = start + len(anno['mention'])
                    if anno['obj'] in map_dict:
                        entity_type = map_dict[anno['obj']]
                        entity_linking = anno['id'][0] if 'id' in anno else 'None'
                        if (start, end, entity_type) not in processed_annos:
                            processed_annos.append((start, end, entity_type))
                            new_begin = start + split_offset
                            new_end = end + split_offset
                            doc_pos = bisect.bisect_right(offsets, new_begin) - 1
                            score = 1.0
                            if 'prob' in anno:
                                if anno['prob'] is not None:
                                    if not math.isnan(anno['prob']):
                                        score = anno['prob']
                            predictions.append((entity_type, entity_linking, doc_ids[doc_pos], new_begin - offsets[doc_pos], new_end - offsets[doc_pos], score))
                            break

    return predictions

def Pubtator3_submit(text, input_type):
    global temp_num
    import requests

    text = '1|t|\n1|a|' + text
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    retry_num = 0
    while (retry_num < 5):
        try:
            time.sleep(1)  # sleep to avoid submitting too often to overload the server
            type = input_type
            data = 'text={}&bioconcept={}'.format(text, type)
            response = requests.post('https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/RESTful/request.cgi', headers=headers,
                                     data=data)
            response = response.json()
            return response['id']
        except:
            print("Error in Reply! Retrying...")
            retry_num += 1
            time.sleep(10)
            # Pubtator3 server seems to block text with certain SQL characters, so replace them if fail
            text = text.replace('select', 'choose')
            text = text.replace('group', 'merge')
            text = text.replace('having', ' have ')
            text = text.replace('into', ' in ')
            text = text.replace('=', '-')
            text = text.replace('?', '.')
            text = text.replace('(', '/')
            text = text.replace(')', '/')

    if retry_num >= 5:
        print("Submission failed!")
        file = open('Pubtator3_error' + str(temp_num) + '.txt', 'w')
        file.write(text)
        temp_num += 1

    return None

def Pubtator3_retrieve(session_id, offsets, doc_ids):
    import requests
    if session_id is None:
        return []

    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    data = 'id='+ session_id

    time.sleep(10)  # sleep to avoid submitting too often to overload the server
    try:
        response = requests.post('https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/RESTful/retrieve.cgi', headers=headers, data=data)
        status_code = response.status_code
    except:
        return None

    if status_code != 200:
        return None

    response = response.content.decode("utf-8", errors="ignore")
    predictions = []
    lines = response.split('\n')[2:]
    for line in lines:
        items = line.split('\t')
        if len(items) == 6:
            _, start_pos, end_pos, entity, tag, linking = items
        elif len(items) == 5:
            _, start_pos, end_pos, entity, tag = items
            linking = 'None'
        else:
            continue

        if 'Mutation' in tag or 'SNP' in tag:
            tag = 'Variant'

        start_pos = int(start_pos) - 1  # -1 because pubtator format assumes a space in the beginning of abstract
        end_pos = int(end_pos) - 1
        doc_pos = bisect.bisect_right(offsets, start_pos) - 1
        predictions.append((tag, linking, doc_ids[doc_pos], start_pos - offsets[doc_pos], end_pos - offsets[doc_pos], 1.0))

    return predictions

def Mmeds_Llama3_predict(sdk_api, input_types, text, offsets, doc_ids):
    map_dict = {'Gene': 'Gene', 'Chemical': 'Chemical', 'Organism': 'Species', 'Disease': 'Disease'}

    splitted_texts = []
    splitted_offsets = []

    occurs = [m.start() for m in re.finditer(re.escape('.'), text)]
    last_pos = 0
    for i in range(len(occurs)):
        if occurs[i] - last_pos > 100:
            splitted_texts.append(text[last_pos:occurs[i]])
            splitted_offsets.append(last_pos)
            last_pos = occurs[i]
    splitted_texts.append(text[last_pos:])
    splitted_offsets.append(last_pos)

    predictions = []
    for split_text, split_offset in zip(splitted_texts, splitted_offsets):
        # assert len(split_text) <= 500

        annos = []
        split_text_no_space = ''
        char_pos = []
        for i in range(len(split_text)):
            if split_text[i] != ' ':
                split_text_no_space += split_text[i]
                char_pos.append(i)
        char_pos.append(len(split_text))

        for type in input_types:
            INSTRUCTION = f"Given a document, recognize the names of {type}. Be sure to output only {type} entities, and ignore other entities. There might be multiple correct answers. If none exist, output \"There is no related entity.\"."
            results = sdk_api.chat([], split_text, INSTRUCTION)
            for ent in results.strip('.').split(','):
                ent = ent.replace(' ','')
                if (type, ent) not in annos:
                    annos.append((type, ent))

        processed_annos = []
        for anno in annos:
            occurs = [m.start() for m in re.finditer(re.escape(anno[1]), split_text_no_space)]
            if occurs != []:
                for occur in occurs:
                    start = char_pos[occur]
                    end = char_pos[occur + len(anno[1])]
                    if anno[0] in map_dict:
                        entity_type = map_dict[anno[0]]
                        entity_linking = 'None'
                        if (start, end, entity_type) not in processed_annos:
                            processed_annos.append((start, end, entity_type))
                            new_begin = start + split_offset
                            new_end = end + split_offset
                            doc_pos = bisect.bisect_right(offsets, new_begin) - 1
                            predictions.append((entity_type, entity_linking, doc_ids[doc_pos],
                                                new_begin - offsets[doc_pos], new_end - offsets[doc_pos], 1.0))

    return predictions

def Scispacy_predict(spacy_models, text, offsets, doc_ids):
    predictions = []
    accept_types_dict = {'GENE_OR_GENE_PRODUCT': 'Gene', 'CHEMICAL': 'Chemical', 'DISEASE': 'Disease', 'TAXON': 'Species', 'CELL_LINE': 'CellLine', 'CELL': 'CellType',
                         'ORGANISM_SUBDIVISION': 'Anatomy', 'DEVELOPING_ANATOMICAL_STRUCTURE': 'Anatomy', 'ORGAN': 'Anatomy', 'TISSUE': 'Anatomy', 'ANATOMICAL_SYSTEM': 'Anatomy'}
    for model in spacy_models:
        doc = model(text)
        for ent in doc.ents:
            if ent.label_ in accept_types_dict:
                doc_pos = bisect.bisect_right(offsets, ent.start_char) - 1
                predictions.append((accept_types_dict[ent.label_], 'None', doc_ids[doc_pos], ent.start_char - offsets[doc_pos], ent.end_char - offsets[doc_pos], 1.0))

    return predictions

def evaluate_with_text_only(method, base_path, dataset_folder, score_threshold):
    if method == 'Pubtator3':
        print('Evaluating Pubtator3...')
        input_types = ['Gene', 'Chemical', 'Disease', 'Species', 'All']
    elif method == 'Hunflair2':
        from flair.models.prefixed_tagger import PrefixedSequenceTagger
        print('Evaluating Hunflair2...')
        tagger = PrefixedSequenceTagger.load("./finetuned_models/hunflair2-ner/pytorch_model.bin")
    elif method == 'BERN2':
        print('Evaluating BERN2...')
    elif method == 'Mmeds_Llama3':
        sys.path.insert(0, '../related_works/Mmeds_Llama3/')
        from mmeds_llama3 import MedS_Llama3
        print('Evaluating Mmeds_Llama3...')
        input_types = ['Gene', 'Chemical', 'Disease', 'Organism']
        sys.path.insert(0, '../related_works/Mmeds_Llama3/')
        sdk_api = MedS_Llama3(model_path="../related_works/Mmeds_Llama3/finetuned_models/MMedS-Llama-3-8B", gpu_id=0)
    elif method == 'Scispacy':
        import scispacy
        import spacy
        print('Evaluating Scispacy...')
        model_names = ["en_ner_craft_md", "en_ner_jnlpba_md", "en_ner_bc5cdr_md", "en_ner_bionlp13cg_md"]
        spacy_models = []
        for name in model_names:
            spacy_models.append(spacy.load(name))
    else:
        raise TypeError('Unknown method name!')
    prediction_types = ['Gene', 'Chemical', 'Disease', 'Species', 'CellLine', 'Variant']

    save_path = base_path + method + '/'
    if os.path.exists(save_path):
        if os.path.exists(save_path+'predictions/'):
            print('Result already exists!')
            return
    else:
        os.makedirs(save_path + 'predictions/')

    test_path = './data/Medical_NER_datasets/' + dataset_folder + '/test/'
    test_data = read_data(test_path, mode='eval_text')

    for data in test_data:
        print('Evaluating on {}...'.format(data['dataset']))

        batch_len = 100000
        text_batchs = []

        cat_text = ''
        offsets = []
        doc_ids = []
        pos = 0
        for doc_id in range(len(data['raw_text'])):
            raw_text = data['raw_text'][doc_id]
            real_text = raw_text[0].split('|t|')[1].strip('\n') + ' ' + raw_text[1].split('|a|')[1].strip('\n')
            offsets.append(pos)
            doc_ids.append(doc_id)
            doc_id += 1
            pos += len(real_text)
            cat_text += real_text

            if pos >= batch_len:
                text_batchs.append((cat_text, offsets, doc_ids))
                cat_text = ''
                offsets = []
                doc_ids = []
                pos = 0

        if cat_text != '':
            text_batchs.append((cat_text, offsets, doc_ids))

        all_predictions = []
        all_session_ids = []
        for (text_batch, batch_offsets, doc_ids) in tqdm(text_batchs, total=len(text_batchs)):
            if method == 'Hunflair2':
                predictions = Hunflair2_predict(tagger, text_batch, batch_offsets, doc_ids)
                all_predictions.append(predictions)
            elif method == 'BERN2':
                predictions = BERN2_predict(text_batch, batch_offsets, doc_ids)
                all_predictions.append(predictions)
            elif method == 'Pubtator3':
                all_type_ids = []
                for input_type in input_types:
                    out_id = Pubtator3_submit(text_batch, input_type)
                    all_type_ids.append(out_id)
                all_predictions.append([])
                all_session_ids.append(all_type_ids)
            elif method == 'Mmeds_Llama3':
                predictions = Mmeds_Llama3_predict(sdk_api, input_types, text_batch, batch_offsets, doc_ids)
                all_predictions.append(predictions)
            elif method == 'Scispacy':
                predictions = Scispacy_predict(spacy_models, text_batch, batch_offsets, doc_ids)
                all_predictions.append(predictions)

        if method == 'Pubtator3':
            print('Waiting for server response...')
            finished = 0
            max_retry_num = 1000
            retry_num = 0
            status = [[False for j in range(len(input_types))] for i in range(len(all_session_ids))]
            while finished < len(all_session_ids) * len(input_types) and retry_num < max_retry_num:
                for i in range(len(all_session_ids)):
                    for j in range(len(all_session_ids[i])):
                        if not status[i][j]:
                            batch_offsets = text_batchs[i][1]
                            doc_ids = text_batchs[i][2]
                            result = Pubtator3_retrieve(all_session_ids[i][j], batch_offsets, doc_ids)
                            if result is not None:
                                status[i][j] = True
                                finished += 1

                                # select CellLine and Variant from All type
                                target_type = ['CellLine', 'Variant'] if input_types[j] == 'All' else [input_types[j]]
                                for item in result:
                                    if item[0] in target_type:
                                        all_predictions[i].append(item)
                            else:
                                retry_num += 1

            file = open(save_path + 'batch_status.txt', 'w')
            file.write(data['dataset'] + ': ')
            for item in status:
                for xx in item:
                    file.write(str(int(xx)))
                    file.write('\n')
            print('success batch: ' + str(finished) + '/' + str(len(all_session_ids) * len(input_types)))

        all_doc_predictions = {x: [[] for id in range(len(data['raw_text']))] for x in prediction_types}
        for i in range(len(text_batchs)):
            for (tag, linking, doc_id, start, end, score) in all_predictions[i]:
                if tag in prediction_types:
                    all_doc_predictions[tag][doc_id].append([start, end, score, linking])

        for type in all_doc_predictions:
            for i in range(len(all_doc_predictions[type])):
                all_doc_predictions[type][i] = post_process(data['raw_text'][i], all_doc_predictions[type][i], data['dataset'], score_threshold)

        name = data['dataset'] + '.pubtator'
        save_predictions(data['raw_text'], all_doc_predictions, save_path + 'predictions/' + name)

def parse_arguments():
    # main args
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", type=str,
                        default='Converted_New')  # dataset folder to evaluate

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()
    base_path = './results/' + args.dataset_folder + '/'

    # evaluate_with_text_only('Mmeds_Llama3', base_path, args.dataset_folder, score_threshold = 0.7)

    # evaluate_with_text_only('Pubtator3', base_path, args.dataset_folder, score_threshold = 0.7)
    # evaluate_with_text_only('BERN2', base_path, args.dataset_folder, score_threshold = 0.7)
    # evaluate_with_text_only('Hunflair2', base_path, args.dataset_folder, score_threshold = 0.7)
    evaluate_with_text_only('Scispacy', base_path, args.dataset_folder, score_threshold=0.7)




