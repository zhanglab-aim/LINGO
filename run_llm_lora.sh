#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --gpus=a100:1
#SBATCH --job-name=test_zhanh
#SBATCH -t 48:00:00
#SBATCH --mem=10000MB
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=huixin.zhan@cshs.org
#SBATCH -o /common/zhanh/DNABERT_2/finetune/out_gpu_lora_llm_budget
#SBATCH -e /common/zhanh/DNABERT_2/finetune/err_gpu_lora_llm_budget
source /common/zhanh/anaconda3/etc/profile.d/conda.sh
conda activate dna_lora
/common/zhanh/anaconda3/envs/dna_lora/bin/python

data_path=$1
lr=3e-5

echo "The provided data_path is $data_path"

for seed in 42
do
    for data in H3 H3K14ac H3K36me3 H3K4me1 H3K4me2 H3K4me3 H3K79me3 H3K9ac H4 H4ac
    do
        python train_llm.py \
            --model_name_or_path facebook/opt-125m \
            --data_path  $data_path/GUE/EMP/$data \
            --kmer -1 \
            --run_name DNABERT2_${vocab}_${lr}_EMP_${data}_seed${seed} \
            --model_max_length 128 \
            --use_lora \
            --lora_target_modules 'k_proj,q_proj,v_proj,fc1,fc2,output_proj' \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 10 \
            --fp16 \
            --save_steps 200 \
            --output_dir /common/zhanh/DNABERT_2/output_llm \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --find_unused_parameters False
    done



    for data in prom_core_all prom_core_notata
    do
        python train_llm.py \
            --model_name_or_path facebook/opt-125m \
            --data_path  $data_path/GUE/prom/$data \
            --kmer -1 \
            --run_name DNABERT2_${vocab}_${lr}_prom_${data}_seed${seed} \
            --model_max_length 20 \
            --use_lora \
            --lora_target_modules 'k_proj,q_proj,v_proj,fc1,fc2,output_proj' \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 5 \
            --fp16 \
            --save_steps 400 \
            --output_dir /common/zhanh/DNABERT_2/output_llm \
            --evaluation_strategy steps \
            --eval_steps 400 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --find_unused_parameters False
    done


    for data in prom_core_tata
    do
        python train_llm.py \
            --model_name_or_path facebook/opt-125m \
            --data_path  $data_path/GUE/prom/$data \
            --kmer -1 \
            --run_name DNABERT2_${vocab}_${lr}_prom_${data}_seed${seed} \
            --model_max_length 20 \
            --usr_lora \
            --lora_target_modules ‘k_proj,q_proj,v_proj,fc1,fc2,output_proj' \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 10 \
            --fp16 \
            --save_steps 200 \
            --output_dir /common/zhanh/DNABERT_2/output_llm \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --find_unused_parameters False
    done

    for data in prom_300_all prom_300_notata
    do
        python train_llm.py \
            --model_name_or_path facebook/opt-125m \
            --data_path  $data_path/GUE/prom/$data \
            --kmer -1 \
            --run_name DNABERT2_${vocab}_${lr}_prom_${data}_seed${seed} \
            --model_max_length 70 \
            --use_lora \
            --lora_target_modules ‘k_proj,q_proj,v_proj,fc1,fc2,output_proj' \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 4 \
            --fp16 \
            --save_steps 400 \
            --output_dir /common/zhanh/DNABERT_2/output_llm \
            --evaluation_strategy steps \
            --eval_steps 400 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --find_unused_parameters False
    done



    for data in prom_300_tata
    do 
        python train_llm.py \
            --model_name_or_path facebook/opt-125m \
            --data_path  $data_path/GUE/prom/$data \
            --kmer -1 \
            --run_name DNABERT2_${vocab}_${lr}_prom_${data}_seed${seed} \
            --model_max_length 70 \
            --use_lora \
            --lora_target_modules ‘k_proj,q_proj,v_proj,fc1,fc2,output_proj' \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 10 \
            --fp16 \
            --save_steps 200 \
            --output_dir /common/zhanh/DNABERT_2/output_llm \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --find_unused_parameters False
    done 


#    for data in reconstructed
#    do
#        python train.py \
#            --model_name_or_path zhihan1996/DNABERT-2-117M \
#            --data_path  $data_path/GUE/splice/$data \
#            --kmer -1 \
#            --run_name DNABERT2_${vocab}_${lr}_splice_${data}_seed${seed} \
#            --model_max_length 80 \
#            --use_lora \
#            --lora_target_modules 'query,value,key' \
#            --per_device_train_batch_size 8 \
#            --per_device_eval_batch_size 16 \
#            --gradient_accumulation_steps 1 \
#            --learning_rate ${lr} \
#            --num_train_epochs 5 \
#            --fp16 \
#            --save_steps 200 \
#            --output_dir output/dnabert2 \
#            --evaluation_strategy steps \
#            --eval_steps 200 \
#            --warmup_steps 50 \
#            --logging_steps 100000 \
#            --overwrite_output_dir True \
#            --log_level info \
#            --find_unused_parameters False
#    done



#    for data in covid
#    do
#        python train.py \
#            --model_name_or_path zhihan1996/DNABERT-2-117M \
#            --data_path  $data_path/GUE/virus/$data \
#            --kmer -1 \
#            --run_name DNABERT2_${vocab}_${lr}_virus_${data}_seed${seed} \
#            --model_max_length 256 \
#            --use_lora \
#            --lora_target_modules 'query,value,key,dense' \
#            --per_device_train_batch_size 32 \
#            --per_device_eval_batch_size 32 \
#            --gradient_accumulation_steps 1 \
#            --learning_rate ${lr} \
#            --num_train_epochs 8 \
#            --fp16 \
#            --save_steps 200 \
#            --output_dir output/dnabert2 \
#            --evaluation_strategy steps \
#            --eval_steps 200 \
#            --warmup_steps 50 \
#            --logging_steps 100000 \
#            --overwrite_output_dir True \
#            --log_level info \
#            --find_unused_parameters False
#    done

#    for data in 0 1 2 3 4
#    do 
#        python train.py \
#            --model_name_or_path zhihan1996/DNABERT-2-117M \
#            --data_path  $data_path/GUE/mouse/$data \
#            --kmer -1 \
#            --run_name DNABERT2_${vocab}_${lr}_mouse_${data}_seed${seed} \
#            --model_max_length 30 \
#            --use_lora \
#            --lora_target_modules 'query,value,key,dense' \
#            --per_device_train_batch_size 8 \
#            --per_device_eval_batch_size 64 \
#            --gradient_accumulation_steps 1 \
#            --learning_rate ${lr} \
#            --num_train_epochs 5 \
#            --max_steps 1000 \
#            --fp16 \
#            --save_steps 200 \
#            --output_dir output/dnabert2 \
#            --evaluation_strategy steps \
#            --eval_steps 200 \
#            --warmup_steps 30 \
#            --logging_steps 100000 \
#            --overwrite_output_dir True \
#            --log_level info \
#            --find_unused_parameters False
#    done


#    for data in 0 1 2 3 4
#    do 
#        python train.py \
#            --model_name_or_path zhihan1996/DNABERT-2-117M \
#            --data_path  $data_path/GUE/tf/$data \
#            --kmer -1 \
#            --run_name DNABERT2_${vocab}_${lr}_tf_${data}_seed${seed} \
#            --model_max_length 30 \
#            --use_lora \
#            --lora_target_modules 'query,value,key,dense' \
#            --per_device_train_batch_size 8 \
#            --per_device_eval_batch_size 64 \
#            --gradient_accumulation_steps 1 \
#            --learning_rate ${lr} \
#            --num_train_epochs 3 \
#            --fp16 \
#            --save_steps 200 \
#            --output_dir output/dnabert2 \
#            --evaluation_strategy steps \
#            --eval_steps 200 \
#            --warmup_steps 30 \
#            --logging_steps 100000 \
#            --overwrite_output_dir True \
#            --log_level info \
#            --find_unused_parameters False
#    done
done
