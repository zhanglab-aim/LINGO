# :shipit:Domain-Shift-Prompting-for-Parameter-Efficient-Fine-Tuning-of-Open-Pre-trained-Transformers-for-Genomic-Sequence

Parameter-Efficient Fine-Tuning (PEFT) has become the de facto approach to fine-tune PFMs while decreasing the computational costs. The current status of PEFT includes:

1. Prefix Tuning methods, e.g., [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://aclanthology.org/2021.acl-long.353/), [P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks](https://arxiv.org/pdf/2110.07602.pdf)
2. Prompt Tuning methods, e.g., [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691)
3. Low-rank adaptation method, e.g.,  [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/abs/2106.09685) and AdaLoRA: [Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2303.10512)
Among these methods, we opt for AdaLoRA+random sampling (AdaLoRA+RS) to deal with the data heterogeneous issue and domain shift prompting (DSP) to leverage the in-context learning ability of LLMs. The framework is as follows:
<img src="/figures/PLM_figure.png" alt="The framework" style="width:8cm; height:auto;"/>
<!-- ![image](/figures/PLM_figure.png) -->

The repository is organized as follows:

1. dataset/: the directory of data sets. We applied our AdaLoRA+RS for a comprehensive set of genome understanding tasks on various LLMs, i.e., promoter detection, epigenetic marks prediction in yeast, and in multiple human cell types. the link is [Here](https://drive.google.com/drive/folders/12FAujYJIT-XR9PCKECvHmLEeTykLkmo9?usp=share_link)
2. finetune/: fine-tuning LLMs and pre-trained DNA foundation models for single label task and multiple label tasks using DSP with BBPE tokenized embeddings and one-hot embeddings.
3. peftnew/: Coupling RS with AdaLoRA method
4. scripts/: SLURM batch script to run the .py files.
5. demos/: Some minimal demos to run AdaLoRA + RS with DSP on OPT and 4-bit quantized Llama. See [llama_dna_sequential_finetune_QLoRA.ipynb](https://github.com/zhanglab-aim/finetune-llm/blob/main/llama_dna_sequential_finetune_QLoRA.ipynb)
 Besides, this link contains 2 fine-tuned checkpoints. See [link](https://drive.google.com/drive/folders/1pDPujSbqzOVxz8OeWtzOTgvjOKInC4nV?usp=share_link)

## Setting up environment 
<pre>
conda env create -f dna_llm.yml
</pre>

## For fine-tune
<pre>
sbatch run_llm_lora.sh data_path
</pre>

## Models support matrix

Find models that are supported out of the box below. 
| Model        | LoRA | AdaLoRA  | AdaLoRA + RS  |AdaLoRA +RS + one-hot w. (DSP)  ||AdaLoRA +RS + BBPE  w. (DSP) |
|--------------| ---- | ---- | ---- | ----  | ----  |
| 1000G-500M        | ✅  | ✅  | ✅  |   | ✅  |
| DNABERT-2        | ✅  | ✅  | ✅  |   | ✅  |
| OPT          | ✅  | ✅  | ✅  | ✅  | ✅  |
| LLaMA        | ✅  | ✅  | ✅  |  |   |


## Figures
<img src="/figures/pareto_front.png" alt="The Pareto front" style="width:8cm; height:auto;"/>
<img src="/figures/mcc.png" alt="The MCC Changes over time" style="width:8cm; height:auto;"/>
<!-- ![The Pareto front](/figures/pareto_front.png) -->
<!-- ![The MCC Changes over time](/figures/mcc.png) -->

## Cite
<pre>
@inproceedings{zhan2023parameter,
  title={Parameter-Efficient Fine-Tune on Open Pre-trained Transformers for Genomic Sequence},
  author={Zhan, Huixin and Zhang, Zijun Frank},
  booktitle={NeurIPS 2023 Generative AI and Biology (GenBio) Workshop},
  year={2023}
}
</pre>

