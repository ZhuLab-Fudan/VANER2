# Load model directly
import os
import copy
import time

from transformers import AutoModel, AutoTokenizer
from model.data import *
from model.utils import *
import torch
from tqdm import tqdm
import argparse
from model.models import UnmaskingLlamaForTokenClassification, MLP_Head
import sys
import shutil
from datetime import datetime
from peft import PeftModel
import torch._dynamo
torch._dynamo.config.suppress_errors = True

@torch.no_grad()
def batch_inference(encoder, type_heads, data, tokenizer, eval_params):

    tokens = data['tokens']
    start_pos = data['token_start_pos']

    if not eval_params['use_prompt']:
        prompt = '.'
    elif eval_params['model_type'] == 'bert':
        prompt = 'Extract all biomedical entities from the following text: '
    else:
        f = open('./LLAMA3.1_prompt.txt', 'r')
        prompt = f.read()

    prompt_tokens = tokenizer(prompt, add_special_tokens=False).data['input_ids']
    prompt_tokens = torch.tensor(prompt_tokens)
    real_len = eval_params['cxt_len'] - prompt_tokens.size(0)
    step_size = real_len // 2

    # concat all test data into one string and divide into batchs for faster evaluation
    concated_token = []
    concated_start_pos = []
    concated_doc_id = []
    for i in range(len(tokens)):
        concated_token += tokens[i] + [eval_params['sep_id']]
        concated_start_pos += start_pos[i]
        concated_doc_id += [i for x in start_pos[i]]

    batch = []
    input = torch.tensor(concated_token)
    sub_batch = []

    # use sliding window to divide a long token sequence into batches
    pos = 0
    while pos + real_len < input.size(0):
        sub_batch.append(torch.cat((prompt_tokens, input[pos:pos + real_len]),0))
        pos = pos + step_size

    # Pad the last window
    temp = input[pos:]
    last = torch.zeros(real_len, dtype=input.dtype) + eval_params['pad_id']
    last[:temp.size(0)] = temp

    sub_batch.append(torch.cat((prompt_tokens, last),0))
    batch += sub_batch
    batch = torch.stack(batch, 0).cuda()

    # Get model predictions
    sbatch_size = 16
    snum = (batch.size(0) + sbatch_size - 1) // sbatch_size

    # get the output embeddings of each batch of text
    sbatchs = []
    for sid in range(snum - 1):
        sbatchs.append(batch[sid * sbatch_size:(sid + 1) * sbatch_size])
    sbatchs.append(batch[(snum - 1) * sbatch_size:])

    all_type_best_tags = {et:[] for et in type_heads}
    all_type_probs = {et:[] for et in type_heads}
    for sbatch in tqdm(sbatchs, total = len(sbatchs)):

        mask = torch.ones_like(sbatch)
        mask[sbatch == 0] = 0
        output_embed = encoder(sbatch, attention_mask=mask).last_hidden_state

        # for each embedding, apply a specific prediction head for each type of entity
        all_type_all_doc_predictions = {}
        for type in type_heads:
            _, btemp, ptemp = type_heads[type](output_embed)
            all_type_best_tags[type].append(btemp[:, -real_len:])
            all_type_probs[type].append(ptemp[:, -real_len:])

    for type in type_heads:
        best_tags = torch.cat(all_type_best_tags[type], 0)
        probs = torch.cat(all_type_probs[type],0)

        pos2 = copy.deepcopy(pos)
        pred_tags = torch.zeros(input.size(0))
        pred_probs = torch.zeros(input.size(0))
        half = (real_len - step_size) // 2
        rest_len = pred_tags.size(0) - pos2

        for j in range(best_tags.size(0)):
            # merging predictions ignoring the beginning and end region of each window
            if j == 0:
                pred_tags[pos2:] = best_tags[best_tags.size(0) - j - 1, :rest_len]
                pred_probs[pos2:] = probs[best_tags.size(0) - j - 1, :rest_len]
            else:
                pred_tags[pos2:pos2 + real_len - half] = best_tags[best_tags.size(0) - j - 1, :-half]
                pred_probs[pos2:pos2 + real_len - half] = probs[best_tags.size(0) - j - 1, :-half]
            pos2 = pos2 - step_size

        texts = [raw_text[0].split('|t|')[1] + raw_text[1].split('|a|')[1].strip('\n') for raw_text in data['raw_text']]
        try:
            # Run with pybind11 for faster inference
            import NER_helper_functions
            tot_text_lens = [len(text) for text in texts]
            pred_tags = list(pred_tags.cpu().numpy().astype(np.int32))
            pred_probs = list(pred_probs.cpu().numpy().astype(np.float64))
            all_doc_predictions = NER_helper_functions.bio2brat(pred_tags, pred_probs, concated_doc_id, concated_start_pos, tot_text_lens, eval_params['use_bioe'])
        except:
            # Run with python
            print("Using python for bio2brat")
            all_doc_tags = {}
            all_doc_predictions = {}
            for tag, prob, doc_id, start_pos in zip(pred_tags, pred_probs, concated_doc_id, concated_start_pos):
                if doc_id not in all_doc_tags:
                    all_doc_tags[doc_id] = []
                all_doc_tags[doc_id].append((tag, prob, start_pos))

            for doc_id in all_doc_tags:
                all_doc_tags[doc_id].append((0, 1, len(texts[doc_id])))  # add one empty token to the last
                all_doc_predictions[doc_id] = bio2brat(all_doc_tags[doc_id], use_bioe = eval_params['use_bioe'])

        all_type_all_doc_predictions[type] = all_doc_predictions

    return all_type_all_doc_predictions

@torch.no_grad()
def get_results(test_data, all_model_predictions, use_score_threshold, save_path):

    # set the score threshold to filter predictions
    score_threshold = (len(all_model_predictions) + 1) * 0.35 if use_score_threshold else 0

    # merge the output of multiple given models
    merged = all_model_predictions[0]
    for data in test_data:
        for model in all_model_predictions[1:]:
            for type in model[data['dataset']]:
                for doc_id in range(len(model[data['dataset']][type])):
                    merged[data['dataset']][type][doc_id] += model[data['dataset']][type][doc_id]

    # post process prediction results (also merges same intervals)
    for data in test_data:
        for type in merged[data['dataset']]:
            for doc_id in range(len(merged[data['dataset']][type])):
                merged[data['dataset']][type][doc_id] = post_process(data['raw_text'][doc_id], merged[data['dataset']][type][doc_id], data['dataset'], score_threshold)

    # Overwrite old results
    if os.path.exists(save_path + 'predictions/'):
        shutil.rmtree(save_path + 'predictions/')
    os.makedirs(save_path + 'predictions/')

    # Save prediction results
    for data in test_data:
        dataset_predictions = merged[data['dataset']]
        name = data['dataset'] + '.pubtator'
        save_predictions(data['raw_text'], dataset_predictions, save_path + 'predictions/' + name, with_score = True)

@torch.no_grad()
def run_single_model(model_path, dataset_folder):
    print("Evaluating model in path:", model_path)

    saved = torch.load(model_path + 'saved.pt', weights_only=True)
    model_config = saved['train_config'] if 'train_config' in saved else saved['model_config']
    base_model_path = saved["base_model_path"]
    model_type = 'llm' if 'Llama' in base_model_path else 'bert'
    use_bioe = model_config['num_classes'] == 4
    use_prompt = model_config['use_prompt']

    # prepare datasets
    label2id = {'O': 0, 'I': 1, 'B': 2, 'E': 3} if use_bioe else {'O': 0, 'I': 1, 'B': 2}
    id2label = {v: k for k, v in label2id.items()}
    if dataset_folder is not None:
        test_path = './data/' + dataset_folder + '/test/'
    else:
        test_path = './data/' + saved['dataset_folder'] + '/test/'

    print("Evaluating datasets in path: ")
    all_paths = glob.glob(test_path + '**/*.pubtator', recursive=True)
    for path in all_paths:
        print('\t' + path)

    # define model
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if model_type == 'llm':
        if os.path.exists(model_path + 'pretrain/'):
            # first merge pretraining weights then finetuning weights
            encoder = UnmaskingLlamaForTokenClassification.from_pretrained(
                base_model_path, num_labels=len(label2id), id2label=id2label, label2id=label2id
            ).bfloat16().to('cuda')
            encoder = PeftModel.from_pretrained(encoder, model_path + 'pretrain/')
            encoder = encoder.merge_and_unload()
            encoder = PeftModel.from_pretrained(encoder, model_path)
        else:
            encoder = UnmaskingLlamaForTokenClassification.from_pretrained(
                base_model_path, num_labels=len(label2id), id2label=id2label, label2id=label2id
            ).bfloat16().to('cuda')
            encoder = PeftModel.from_pretrained(encoder, model_path)

        encoder.eval()
        cxt_len, embed_size, _, _ = get_llm_base_params(base_model_path.split('/')[-1])
    elif model_type == 'bert':
        encoder = AutoModel.from_pretrained(model_path).bfloat16().cuda()
        cxt_len, embed_size, _, _ = get_bert_base_params(base_model_path.split('/')[-1])
        encoder.eval()
    else:
        raise TypeError('Unknown model type!')

    torch.set_float32_matmul_precision('high')
    encoder = torch.compile(encoder)
    type_heads = {}
    for type in saved['type_heads_state_dict']:
        type_heads[type] = MLP_Head(model_config,embed_size).bfloat16().cuda()
        type_heads[type] = torch.compile(type_heads[type])
        type_heads[type].load_state_dict(saved['type_heads_state_dict'][type])

    pad_id, sep_id, _, _ = get_special_token_ids(tokenizer, model_type, base_model_path.split('/')[-1])

    # preprocessing data for testing
    test_data = read_data(test_path, mode = 'eval', tokenizer=tokenizer,  label2id = label2id, use_bioe = use_bioe, verbose = True)

    eval_params = {
        'model_type': model_type,
        'use_prompt': use_prompt,
        'use_bioe': use_bioe,
        'cxt_len': cxt_len,
        'pad_id': pad_id,
        'sep_id': sep_id
    }

    # Run NER models
    all_data_all_type_all_doc_predictions = {}
    for data in test_data:
        all_data_all_type_all_doc_predictions[data['dataset']] = batch_inference(encoder, type_heads, data, tokenizer, eval_params)

    return test_data, all_data_all_type_all_doc_predictions

def parse_arguments():
    # main args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_names", nargs="*", type=str,
                        default=None)
    parser.add_argument("--dataset_folder", type=str,
                        default='Converted_New')  # dataset folder to evaluate
    parser.add_argument("--merge_all_model_outputs", type=str,
                        default='False')  # Whether to merge outputs of selected models by voting
    parser.add_argument("--use_score_threshold", type=str,
                        default='True')  # Whether to filter results with score threshold

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_arguments()
    args.use_score_threshold = args.use_score_threshold.lower() == 'true'
    args.merge_all_model_outputs = args.merge_all_model_outputs.lower() == 'true'
    model_names = args.model_names

    model_path = './finetuned_models/'
    save_path = './results/' + args.dataset_folder + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    timestamp = datetime.now().strftime('%m%d%H%M%S')

    # evaluate all finetuned models that hasn't been evaluated on given dataset
    if model_names is None:
        model_names = []
        for model_name in os.listdir(path=model_path):
            if not os.path.exists(save_path + model_name):
                model_names.append(model_name)

    if args.merge_all_model_outputs:
        # output a merged result of all input models
        merged_save_path = save_path + timestamp + '_merged/'
        os.makedirs(merged_save_path + '/predictions/')
        f = open(merged_save_path + 'merged_models.txt', 'w')
        all_model_predictions = []
        for name in model_names:
            if os.path.exists(model_path + name + '/saved.pt'):
                test_data, all_data_all_type_all_doc_predictions = run_single_model(model_path + name + '/', args.dataset_folder)
                all_model_predictions.append(all_data_all_type_all_doc_predictions)
                f.write(name + '\n')
        f.close()
        get_results(test_data, all_model_predictions, args.use_score_threshold, merged_save_path)
    else:
        # output result for each input model
        for name in model_names:
            if os.path.exists(model_path + name + '/saved.pt'):
                test_data, all_data_all_type_all_doc_predictions = run_single_model(model_path + name + '/', args.dataset_folder)
                get_results(test_data, [all_data_all_type_all_doc_predictions], args.use_score_threshold, save_path + name + '/')




