# InSPEcT - Eliciting Textual Descriptions from Representations of Continuous Prompts
This is the official code for [Eliciting Textual Descriptions from Representations of Continuous Prompts](https://arxiv.org/abs/2410.11660) by Dana Ramati, Daniela Gottesman and Mor Geva. 2024.


## Run Configurations

### Prompt Tuning
To Run Prompt Tuning on `SetFit/sst2` Dataset with `Meta-Llama-3-8B-Instruct` use the following command:
```
python -m scripts.train_prompt -m meta-llama/Meta-Llama-3-8B-Instruct -d SetFit/sst2 -t text -l label_text -lr 8e-4 -es validation -o trained_prompts
```

To Run Prompt Tuning on `stanfordnlp/sst2` Dataset with `Meta-Llama-3-8B-Instruct` use the following command:
```
python -m scripts.train_prompt -m meta-llama/Meta-Llama-3-8B-Instruct -d stanfordnlp/sst2 -t sentence -l label -lr 8e-4 -es validation -o trained_prompts -mt 50000 -tl
```

### Running Patchscopes on Trained Soft Prompts
To Run Patchscopes on trained soft prompts using `Meta-Llama-3-8B-Instruct` and SST2 Dataset, use the following command:
```
python -m scripts.create_patching_outputs -m meta-llama/Meta-Llama-3-8B-Instruct -d SetFit/sst2 -n 7 -c trained_prompts/Meta-Llama-3-8B-Instruct_sst2_lr0.0008_8_epochs_pt_n7 -t description_and_classes -i 1
```

### Calculate Scores on the elicited descriptions of the Soft Prompts
```
python -m scripts.calculate_scores -i patching_output/Meta-Llama-3-8B-Instruct/SetFit/sst2/n7_target_description_and_classes_1 -o scores/Meta-Llama-3-8B-Instruct_sst2_lr0.0008_8_epochs_pt_n7 -t sst2
```





## Citation
```
@article{ramati2024inspect,
      title={Eliciting Textual Descriptions from Representations of Continuous Prompts}, 
      author={Dana Ramati and Daniela Gottesman and Mor Geva},
      year={2024},
      journal={arXiv preprint arXiv:2410.11660},
      url={https://arxiv.org/abs/2410.11660}, 
}
```
