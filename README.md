# VANER2: Towards more general biomedical named entity recognition using multi-task large language model encoders

## Preparation
1. Create python 3.10 environment with requirements.txt.
2. Due to possible copyright issues, the training and testing dataset will not be directly available, but is available upon reasonable request. We provided example data in the data folder.
3. Download the Llama-3.1-8B-Instruct base model from https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct and put them under base_models folder.
4. Pretrained parameters of VANER2 can be downloaded at https://huggingface.co/droidlyx866/VANER2, and unziped to finetuned_models folder.

## Prediction
For VANER2 prediction, run run_VANER2_NER.py --model_names [folder names of trained models in the results folder]

Example: python run_VANER2_NER.py --model_names VANER2

The model predictions should be saved in results/[dataset folder name]/[model name]

Note: 20GB GPU memory is required. The pybind11 module is to speed up prediction, without it the prediction will be slower. To install it, run pip install ./pybind11_module

## Evaluation
For Evaluation of the results, run evaluate_results.py, which evaluates all results under results/[dataset folder name] and generate a NER_summary.tsv file containing the results.

More detailed scores are saved in results/[dataset folder name]/[model name]/all_NER_scores.txt 

## Baseline methods
For prediction of other baseline methods, run run_other_models.py, the results can then be evaluated using evaluate_results.py.

Note that for Hunflair2, you need to download the model from https://huggingface.co/hunflair/hunflair2-ner and save to finetuned_models/hunflair2-ner

For scispacy, you need to install four scispacy models en_ner_craft_md, en_ner_jnlpba_md, en_ner_bc5cdr_md, en_ner_bionlp13cg_md. see https://github.com/allenai/scispacy for guidelines.

AIONER needs a separate python environment, thus are not included in our code, please refer to https://github.com/ncbi/AIONER to reproduce their results.
Similarly, for VANER, please refer to https://github.com/Eulring/VANER.

## Entity Linking

Additionally, you can run the code run_hunflair2_NEN.py to automatically add Entity Linking results to the NER results of VANER2 using the pretrained models of Hunflair2 [1]. The code will automatically download the EL models from the flair repository. Alternatively, you can download the EL models from the following links manually and modify run_hunflair2_NEN.py to load the downloaded models:

Gene-linker: https://huggingface.co/hunflair/biosyn-sapbert-bc2gn

Chemical-linker: https://huggingface.co/hunflair/biosyn-sapbert-bc5cdr-chemical

Disease-linker: https://huggingface.co/hunflair/biosyn-sapbert-ncbi-disease

Species-linker: https://huggingface.co/hunflair/sapbert-ncbi-taxonomy

After that, run evaluate_results.py --evaluate_NEN True to evaluate NER and NEN simutaneously, if NEN labels are provided in the test dataset.

[1] Sänger M, Garda S, Wang X D, et al. HunFlair2 in a cross-corpus evaluation of biomedical named entity recognition and normalization tools[J]. Bioinformatics, 2024, 40(10): btae564.


