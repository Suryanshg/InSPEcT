# InSPEcT - Eliciting Textual Descriptions from Representations of Continuous Prompts
This is the official code for [Eliciting Textual Descriptions from Representations of Continuous Prompts](https://arxiv.org/abs/2410.11660) by Dana Ramati, Daniela Gottesman and Mor Geva. 2024.


## Run Configurations
---

### Prompt Tuning
To Run Prompt Tuning on SST2 Dataset with Meta-Llama-3-8B-Instruct use the following command:
```
python -m scripts.train_prompt -m meta-llama/Meta-Llama-3-8B-Instruct -d SetFit/sst2 -t text -l label_text -lr 8e-4 -es validation -ee 1000
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
