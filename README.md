# VANER2: Towards more general biomedical named entity recognition using multi-task large language model encoders

## Preparation
1. Create python environment with requirements.txt.  
2. Download the train and test data from https://zenodo.org/records/15209913 and unzip to VANER2/data folder.  
3. Download the base models from the following links and put them under VANER2/base_models:  
Llama-3.1-8B-Instruct: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct  
PubmedBERT: https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext  
BiolinkBERT: https://huggingface.co/michiyasunaga/BioLinkBERT-base

Pretrained parameters of VANER2 can be downloaded at https://zenodo.org/records/15210322, and unziped to VANER2/finetuned_models  

## Prediction
For VANER2 prediction, run run_VANER2_NER.py --model_names [folder names of trained models in VANER2/results]

Example: run_VANER2_NER.py --model_names Meta_Llama_3.1_8B_Instruct_Converted_New_True_True_True_4_1.00

The model predictions should be saved in VANER2/results/[dataset folder name]/[model name]

Note: the pybind11 module is to speed up prediction, without it the prediction will be slower. To install it, run pip install ./pybind11_module under VANER2 folder.

## Evaluation
For Evaluation of the results, run evaluate_results.py, which evaluates all results under VANER2/results/[dataset folder name] and generate a NER_summary.tsv file containing the results.

More detailed scores are saved in VANER2/results/[dataset folder name]/[model name]/all_NER_scores.txt 

## Baseline methods
For prediction of other baseline methods, run run_other_models.py, the results can then be evaluated using evaluate_results.py.

Note that for Hunflair2, you need to download the model from https://huggingface.co/hunflair/hunflair2-ner and save to VANER2/finetuned_models/hunflair2-ner

For scispacy, you need to install four scispacy models en_ner_craft_md, en_ner_jnlpba_md, en_ner_bc5cdr_md, en_ner_bionlp13cg_md. see https://github.com/allenai/scispacy for guidelines.


