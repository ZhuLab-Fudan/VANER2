# Load model directly
import os
import copy
import random
import time
from transformers import AutoModel, AutoConfig, AutoTokenizer, AutoModelForCausalLM
import sys
from run_VANER2_NER import *
import torch
import wandb
from tqdm import tqdm
import numpy as np
from einops import rearrange
import argparse
from model.models import UnmaskingLlamaForTokenClassification, MLP_Head
from model.utils import *
from evaluate_results import count_tp, prf_metrics
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import bitsandbytes
import psutil
import torch._dynamo
torch._dynamo.config.suppress_errors = True
from peft import PeftModel, get_peft_model, LoraConfig, TaskType

sys.path.append(os.path.join(os.getcwd(), "model/peft/src/"))

def create_model(train_config, pretrained_encoder = None):
    # define model
    print('Loading model: ' + base_model_path)
    type_heads = {}
    betas = [0.9, 0.999]
    if model_type == 'llm':
        encoder = UnmaskingLlamaForTokenClassification.from_pretrained(
            base_model_path, num_labels=len(label2id), id2label=id2label, label2id=label2id
        ).bfloat16().to('cuda') if pretrained_encoder is None else pretrained_encoder
        if llm_tuner == 'Lora':
            peft_config = LoraConfig(task_type=TaskType.TOKEN_CLS, inference_mode=False, r=12, lora_alpha=32,
                                     lora_dropout=0.1)
            encoder = get_peft_model(encoder, peft_config)
            encoder.print_trainable_parameters()
        else:
            train_config['encoder_learning_rate'] = 2e-5

        cxt_len, embed_size, batch_size, grad_acc_steps = get_llm_base_params(args.base_model_path.split('/')[-1])
        for type in train_config['all_type_class_weight']:
            type_heads[type] = MLP_Head(train_config, embed_size).bfloat16().cuda()

    elif model_type == 'bert':
        encoder = AutoModel.from_pretrained(base_model_path).cuda() if pretrained_encoder is None else pretrained_encoder
        cxt_len, embed_size, batch_size, grad_acc_steps = get_bert_base_params(base_model_path.split('/')[-1])
        for type in train_config['all_type_class_weight']:
            type_heads[type] = MLP_Head(train_config, embed_size).cuda()
    else:
        raise TypeError('Unknown model type!')

    # Compile model to improve speed
    torch.set_float32_matmul_precision('high')
    encoder = torch.compile(encoder)
    for type in train_config['all_type_class_weight']:
        type_heads[type] = torch.compile(type_heads[type])

    # Define optimizer
    use_quantization = model_type == 'llm' and llm_tuner == 'Full'
    encoder_optimizer = get_encoder_optimizer(encoder, train_config['encoder_learning_rate'], betas, weight_decay,
                                              quantization=use_quantization)
    all_head_params = []
    for type in train_config['all_type_class_weight']:
        all_head_params += list(type_heads[type].parameters())
    head_optimizer = torch.optim.AdamW(all_head_params, lr=train_config['head_learning_rate'],
                                       weight_decay=weight_decay)

    return encoder, type_heads, encoder_optimizer, head_optimizer, cxt_len, embed_size, batch_size, grad_acc_steps

def get_encoder_optimizer(encoder, lr, betas, weight_decay, quantization = False):
    param_dict = {pn: p for pn, p in encoder.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]

    if quantization:
        encoder_optimizer = bitsandbytes.optim.AdamW8bit(optim_groups, betas=betas, lr=lr)
    else:
        encoder_optimizer = torch.optim.AdamW(optim_groups, betas=betas, lr=lr)

    return encoder_optimizer

# pretraining based on masked next token prediction
def pre_train(save_path, data_use_percentage):
    pre_train_config = copy.deepcopy(train_config)
    pre_train_config['num_classes'] = len(tokenizer)
    pre_train_config['all_type_class_weight'] = {'Token':None}
    pre_train_config['num_epoches'] = 1
    mask_prob = 0.15
    encoder, type_heads, encoder_optimizer, head_optimizer, cxt_len, embed_size, batch_size, grad_acc_steps = create_model(pre_train_config)

    # begin training
    print('--------------------------------------------------')
    print("Start pretraining model")
    print('--------------------------------------------------')
    last_t = time.time()
    avg_loss = 0
    loss_cnt = 0

    # Convert number of epochs to number of steps
    tot_len = 0
    for item in train_data:
        tot_len += item['tokens'].size(0)
    tot_len = int(tot_len * data_use_percentage)
    real_len = cxt_len - 10
    num_steps = int(
        tot_len // (batch_size * real_len) * pre_train_config['num_epoches'])
    print("Number of training steps:", num_steps)

    for step in range(num_steps):
        batch, _, dataset_name = get_batch(train_data, None, batch_size, cxt_len, data_use_percentage)
        # Mask
        labels = copy.deepcopy(batch)
        mask_roll = torch.rand(batch.size()).cuda()
        labels[mask_roll >= mask_prob] = -1
        labels = torch.cat((labels[:, 1:], torch.zeros((batch.size(0), 1)).cuda() - 1),1)
        labels = labels.to(torch.int64)

        # Following BERT pretraining, among mask tokens, 10% don't mask, 10% random token, 80% mask.
        random_mask = torch.logical_and(mask_roll >= mask_prob * 0.1, mask_roll < mask_prob * 0.2)
        normal_mask = torch.logical_and(mask_roll >= mask_prob * 0.2, mask_roll < mask_prob)
        token_roll = torch.randint(0,len(tokenizer),batch.size()).cuda()
        batch[random_mask] = token_roll[random_mask]
        batch[normal_mask] = mask_id

        # Forward model
        output_embed = encoder(batch).last_hidden_state
        logits, _, _ = type_heads['Token'](output_embed)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1),
                                                 reduction='none',
                                                 ignore_index=-1).mean()

        avg_loss = avg_loss + loss.item()
        loss_cnt = loss_cnt + 1
        loss = loss / grad_acc_steps
        loss.backward()

        if (step + 1) % grad_acc_steps == 0:
            encoder_optimizer.step()
            head_optimizer.step()
            encoder_optimizer.zero_grad(set_to_none=True)
            head_optimizer.zero_grad(set_to_none=True)

        # Logging
        if (step + 1) % log_interval == 0:
            torch.cuda.synchronize()
            dt = time.time() - last_t
            last_t = time.time()

            lossf = avg_loss / loss_cnt
            avg_loss = 0
            loss_cnt = 0

            mapping = {0: 1, 1: 2, 2: 0}
            GPU_used = check_gpu_memory(mapping[batch.get_device()]) // 2 ** 20
            RAM_used = tot_RAM - psutil.virtual_memory().available / (2 ** 20)
            if RAM_used > 100000:
                print('Too much Disk memory used, Terminating...')
                exit()

            print(
                f"iter {step + 1}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, Peak GPU memory {GPU_used:.2f}MB, RAM used {RAM_used:.2f}MB")

    if not os.path.exists(save_path + 'pretrain/'):
        os.makedirs(save_path + 'pretrain/')
    encoder.save_pretrained(save_path + 'pretrain/')

    print('--------------------------------------------------')
    print("Finished pretraining model")
    print('--------------------------------------------------')
    return encoder.merge_and_unload()

def train(save_path, data_use_percentage, pretrained_encoder = None):
    # Calculate weight to adjust loss based on number of instances each class
    print("Classes:", train_config['classes'])
    all_type_class_weight = {}
    for item in train_data:
        for type in item['tags']:
            if type != 'CellComponent':
                if type not in all_type_class_weight:
                    all_type_class_weight[type] = torch.zeros(train_config['num_classes'])
                for i in range(all_type_class_weight[type].size(0)):
                    all_type_class_weight[type][i] += torch.numel(item['tags'][type]) - torch.count_nonzero(
                        item['tags'][type] - i)
    for type in all_type_class_weight:
        all_type_class_weight[type] = torch.mean(all_type_class_weight[type]) / (all_type_class_weight[type] + 1)
        all_type_class_weight[type] = all_type_class_weight[type].cuda()
        print("Class weight of {}:".format(type),
              [float('{:.4f}'.format(x.item())) for x in all_type_class_weight[type]])

    train_config['all_type_class_weight'] = all_type_class_weight
    encoder, type_heads, encoder_optimizer, head_optimizer, cxt_len, embed_size, batch_size, grad_acc_steps = create_model(train_config, pretrained_encoder)

    if not use_prompt:
        prompt = '.'
    elif model_type == 'bert':
        prompt = 'Extract all biomedical entities from the following text: '
    else:
        f = open('./LLAMA3.1_prompt.txt', 'r')
        prompt = f.read()

    prompt_tokens = tokenizer(prompt, add_special_tokens=False).data['input_ids']
    prompt_tokens = torch.tensor(prompt_tokens)

    # Begin training
    print('--------------------------------------------------')
    print("Start finetuning model")
    print('--------------------------------------------------')
    last_t = time.time()
    avg_loss = 0
    loss_cnt = 0

    # Convert number of epochs to number of steps
    tot_len = 0
    for item in train_data:
        tot_len += item['tokens'].size(0)
    tot_len = int(tot_len * data_use_percentage)
    real_len = cxt_len - 10
    num_steps = int(tot_len // (batch_size * real_len) * train_config['num_epoches'])
    print("Number of training steps:", num_steps)

    cur_encoder_lr = train_config['encoder_learning_rate']
    cur_head_lr = train_config['head_learning_rate']
    lr_change_interval = num_steps // 100
    if lr_change_interval == 0:
        lr_change_interval = 1
    if num_steps > 100000:
        lr_change_gamma = 0.98
    elif num_steps > 10000:
        lr_change_gamma = 0.99
    else:
        lr_change_gamma = 1.00

    if use_wandb:
        wandb_config = copy.deepcopy(train_config)
        wandb_config['data_path'] = train_path
        wandb_config['encoder_learning_rate'] = train_config['encoder_learning_rate']
        wandb_config['head_learning_rate'] = train_config['head_learning_rate']
        wandb_config['weight_decay'] = weight_decay
        wandb_config['training_steps'] = num_steps
        wandb.init(
            project='VANER2',
            config=wandb_config
        )

    cur_batch_size = batch_size
    cur_cxt_len = cxt_len
    for step in range(num_steps):

        batch, all_types_l_batch, dataset_name = get_batch(train_data, prompt_tokens, cur_batch_size, cur_cxt_len, data_use_percentage)

        # Data augmentation while training
        if train_config['data_augmentation_prob'] > 0:
            mask = torch.rand(batch.size()).cuda() < train_config['data_augmentation_prob']
            batch[mask] = unk_id

        # Forward model
        output_embed = encoder(batch).last_hidden_state
        tot_loss = torch.tensor([0], dtype=torch.float).cuda()
        for type in all_types_l_batch:
            if type in type_heads:
                logits, _, _ = type_heads[type](output_embed)
                labels = all_types_l_batch[type]
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1),
                                                         reduction='none',
                                                         ignore_index=-1)
                class_weight = all_type_class_weight[type]
                if use_class_weight and class_weight is not None:
                    all_preds = torch.argmax(logits, dim=-1)
                    loss = loss * (class_weight[all_preds.view(-1)] + class_weight[labels.view(-1)]) / 2
                tot_loss += loss.mean()

        out_loss = tot_loss.item() / len(all_types_l_batch)
        avg_loss = avg_loss + out_loss
        loss_cnt = loss_cnt + 1
        if use_wandb:
            wandb.log({"loss": out_loss})

        tot_loss = tot_loss / grad_acc_steps
        tot_loss.backward()

        if (step + 1) % grad_acc_steps == 0:
            encoder_optimizer.step()
            head_optimizer.step()
            encoder_optimizer.zero_grad(set_to_none=True)
            head_optimizer.zero_grad(set_to_none=True)

            # Randomly vary context length for each batch
            cur_cxt_len = random.randint(prompt_tokens.size(0) + 1, cxt_len)
            cur_batch_size = batch_size * cxt_len // cur_cxt_len

        if (step + 1) % lr_change_interval == 0:
            cur_encoder_lr *= lr_change_gamma
            cur_head_lr *= lr_change_gamma
            for g in encoder_optimizer.param_groups:
                g['lr'] = cur_encoder_lr
            for g in head_optimizer.param_groups:
                g['lr'] = cur_head_lr

        # Logging
        if (step + 1) % log_interval == 0:
            torch.cuda.synchronize()
            dt = time.time() - last_t
            last_t = time.time()

            lossf = avg_loss / loss_cnt
            avg_loss = 0
            loss_cnt = 0

            mapping = {0: 1, 1: 2, 2: 0}
            GPU_used = check_gpu_memory(mapping[batch.get_device()]) // 2 ** 20
            RAM_used = tot_RAM - psutil.virtual_memory().available / (2 ** 20)
            if RAM_used > 100000:
                print('Too much Disk memory used, Terminating...')
                exit()

            print(
                f"iter {step + 1}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, Peak GPU memory {GPU_used:.2f}MB, RAM used {RAM_used:.2f}MB")

        # Evaluate model
        if (step + 1) % eval_interval == 0:
            with torch.no_grad():
                encoder.eval()
                for type in type_heads:
                    type_heads[type].eval()

                all_results = np.zeros(8)
                for sample in tqdm(range(eval_samples)):
                    batch, all_types_l_batch, dataset_name = get_batch(val_data, prompt_tokens, batch_size, cxt_len, 1.0)
                    output_embed = encoder(batch).last_hidden_state

                    for type in all_types_l_batch:
                        if type in type_heads:
                            _, preds, probs = type_heads[type](output_embed)
                            size0, size1 = preds.size()
                            preds = preds.cpu().numpy()
                            probs = probs.float().cpu().numpy()
                            labels = all_types_l_batch[type].cpu().numpy()
                            for bn in range(size0):
                                zip_pred = [[preds[bn][x], probs[bn][x], x] for x in range(size1)]
                                zip_pred.append([0,1,size1])
                                zip_label = [[labels[bn][x], 1, x] for x in range(size1)]
                                zip_label.append([0,1,size1])
                                single_pred = bio2brat(zip_pred, use_bioe)
                                single_label = bio2brat(zip_label, use_bioe)
                                NER_results = count_tp([single_pred], [single_label])
                                all_results += NER_results

                precision, recall, f_score = prf_metrics(*list(all_results[:3]))
                print(f"val precision: {precision:.4f}, val recall: {recall:.4f}, val f_score: {f_score:.4f}")
                print("Nwrong: {:.0f}  Nmiss: {:.0f}  Nmore: {:.0f}  Nless: {:.0f}  Noverlap: {:.0f}".format(
                    *list(all_results[3:])))

                if use_wandb:
                    wandb.log({"val_precision": precision})
                    wandb.log({"val_recall": recall})
                    wandb.log({"val_f_score": f_score})
                    wandb.log({"val_Nwrong": all_results[3]})
                    wandb.log({"val_Nmiss": all_results[4]})
                    wandb.log({"val_Nmore": all_results[5]})
                    wandb.log({"val_Nless": all_results[6]})
                    wandb.log({"val_Noverlap": all_results[7]})

                encoder.train()
                for type in type_heads:
                    type_heads[type].train()

    # Save model
    encoder.save_pretrained(save_path)
    torch.save({
        'base_model_path': base_model_path,
        'type_heads_state_dict': {type: type_heads[type].state_dict() for type in type_heads},
        'train_config': train_config,
        'dataset_folder': args.dataset_folder
    }, save_path + 'saved.pt')

    print('--------------------------------------------------')
    print("Finished finetuning model")
    print('--------------------------------------------------')

def parse_arguments():
    # main args
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", help='',
                        default='./base_models/Meta-Llama-3.1-8B-Instruct')
    parser.add_argument("--dataset_folder", help='',
                        default='Converted_New')
    parser.add_argument("--num_epochs", type=int,
                        default=4)
    parser.add_argument("--data_use_percentage", type=float,
                        default=1.0)
    parser.add_argument("--train_single_datasets", type=str,
                        default='False') # train a separate model on each dataset instead of training on all datasets

    parser.add_argument("--pretrain", type=str,
                        default='True')  # Only works for LLM
    parser.add_argument("--use_prompt", type=str,
                        default='True')
    parser.add_argument("--use_class_weight", type=str,
                        default='True')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    torch.manual_seed(1234567)
    np.random.seed(1234567)
    tot_RAM = psutil.virtual_memory().available / (2**20)

    args = parse_arguments()
    class_type = 'bio' # 'bio', 'bioe'
    model_type = 'llm' if 'Llama' in args.base_model_path else 'bert'
    pretrain = (args.pretrain.lower() == 'true')
    train_single_datasets = (args.train_single_datasets.lower() == 'true')
    use_class_weight = (args.use_class_weight.lower() == 'true')
    use_prompt = (args.use_prompt.lower() == 'true')

    if model_type != 'llm':
        pretrain = False

    use_wandb = False
    log_interval = 100
    eval_interval = 500
    eval_samples = 25

    train_config = {
        'num_epoches': args.num_epochs,
        "class_type": class_type,
        "num_classes": 3 if class_type == 'bio' else 4,
        "classes": ['O','I','B'] if class_type == 'bio' else ['O','I','B','E'],
        "number_of_head_layers": 2,
        "encoder_learning_rate": 3e-5,
        "head_learning_rate": 3e-5,
        "hidden_dropout_prob": 0.1,
        "data_augmentation_prob": 0.1,  # prob to randomly mask entity tokens in training data
        "weight_decay": 0.1,
        'use_prompt': use_prompt
    }
    use_bioe = class_type == 'bioe'
    base_model_path = args.base_model_path
    weight_decay = train_config['weight_decay']
    label2id = {'O': 0, 'I': 1, 'B': 2, 'E': 3} if use_bioe else {'O': 0, 'I': 1, 'B': 2}
    id2label = {v: k for k, v in label2id.items()}

    llm_tuner = 'Lora'  # ‘Lora', 'Full'

    dataset_folder = args.dataset_folder
    if train_single_datasets:
        all_model_train_path = glob.glob('./data/Medical_NER_datasets/' + dataset_folder + '/train/*.pubtator')
        all_model_test_path = [item.replace('train','test') for item in all_model_train_path]
    else:
        all_model_train_path = ['./data/Medical_NER_datasets/' + dataset_folder + '/train/']
        all_model_test_path = ['./data/Medical_NER_datasets/' + dataset_folder + '/test/']

    for train_path, test_path in zip(all_model_train_path, all_model_test_path):
        # Initialize time and wandb
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        _, sep_id, unk_id, mask_id = get_special_token_ids(tokenizer, model_type, base_model_path.split('/')[-1])

        # Load training data
        print("Processing train data...")
        train_data = read_data(train_path, mode = 'train', tokenizer = tokenizer,  label2id = label2id, sep_id = sep_id, use_bioe = use_bioe, verbose = True)
        dataset_heads = {}
        for item in train_data:
            dataset_heads[item['dataset']] = None

        print("Processing validation data...")
        val_data = read_data(test_path, mode = 'train', tokenizer = tokenizer, label2id = label2id, sep_id = sep_id, use_bioe = use_bioe, verbose = True)

        # Create save path
        save_args_str = '_{}_{}_{}_{}_{:.2f}'.format(str(use_prompt), str(pretrain), str(use_class_weight), str(args.num_epochs), args.data_use_percentage)
        if train_single_datasets:
            save_args_str = '_' + train_path.split('/')[-1].split('__')[0] + save_args_str

        save_path = './finetuned_models/' + base_model_path.split('/')[-1].replace('-','_') + '_' + dataset_folder + save_args_str + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        pretrained_encoder = None
        if pretrain:
            pretrained_encoder = pre_train(save_path, args.data_use_percentage)

        train(save_path, args.data_use_percentage, pretrained_encoder)






