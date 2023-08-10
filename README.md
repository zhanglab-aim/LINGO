# finetune-llm

##  Data link 

[Here](https://drive.google.com/drive/folders/12FAujYJIT-XR9PCKECvHmLEeTykLkmo9?usp=share_link)

## Setting up environment 
<pre>
conda env create -f dna_llm.yml
</pre>

## An example for fine-tuning the OPT-125M model using H3 data.
See [DNA_llm_test.ipynb](https://github.com/zhanglab-aim/finetune-llm/blob/main/DNA_llm_test.ipynb)

## Fine-tuning the OPT-125M model on all datasets
<pre>
sbatch run_llm_lora.sh data_path
</pre>

