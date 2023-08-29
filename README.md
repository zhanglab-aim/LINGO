# finetune-llm

##  Data link 

[Here](https://drive.google.com/drive/folders/12FAujYJIT-XR9PCKECvHmLEeTykLkmo9?usp=share_link)

## Setting up environment 
<pre>
conda env create -f dna_llm.yml
</pre>

## An example for fine-tuning the OPT-125M model using H3 data.
See [DNA_llm_test.ipynb](https://github.com/zhanglab-aim/finetune-llm/blob/main/DNA_llm_test.ipynb)

## An example for fine-tuning the LLama2 model using H3 data.
See [llama_dna_sequential_finetune_QLoRA.ipynb](https://github.com/zhanglab-aim/finetune-llm/blob/main/llama_dna_sequential_finetune_QLoRA.ipynb)

## Fine-tuning the OPT-125M model on all datasets
<pre>
sbatch run_llm_lora.sh data_path
</pre>

## Figures
![The Pareto front](https://github.com/zhanglab-aim/finetune-llm/edit/main/Figures_llm/pareto_front.png)
![The MCC Changes over time](https://github.com/zhanglab-aim/finetune-llm/edit/main/Figures_llm/test_performance.png)

