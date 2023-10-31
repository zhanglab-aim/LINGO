#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --gpus=a100:1
#SBATCH --job-name=test_
#SBATCH -t 48:00:00
#SBATCH --mem=10000MB
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=
#SBATCH -o /common/
#SBATCH -e /common/
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
        python train_opt350m.py \
            --data_path  $data_path/GUE/EMP/$data \
            --kmer -1 \
            --run_name DNABERT2_${vocab}_${lr}_EMP_${data}_seed${seed} \
            --model_max_length 128 \
            --use_lora False \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 5 \
            --fp16 \
            --save_steps 200 \
            --output_dir output/opt_350m \
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
        python train_opt350m.py \
            --data_path  $data_path/GUE/prom/$data \
            --kmer -1 \
            --run_name DNABERT2_${vocab}_${lr}_prom_${data}_seed${seed} \
            --model_max_length 20 \
            --use_lora False \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 5 \
            --fp16 \
            --save_steps 400 \
            --output_dir output/opt_350m \
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
        python train_opt350m.py \
            --data_path  $data_path/GUE/prom/$data \
            --kmer -1 \
            --run_name DNABERT2_${vocab}_${lr}_prom_${data}_seed${seed} \
            --model_max_length 20 \
            --usr_lora False \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 10 \
            --fp16 \
            --save_steps 200 \
            --output_dir output/opt_350m \
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
        python train_opt350m.py \
            --data_path  $data_path/GUE/prom/$data \
            --kmer -1 \
            --run_name DNABERT2_${vocab}_${lr}_prom_${data}_seed${seed} \
            --model_max_length 70 \
            --use_lora False \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 4 \
            --fp16 \
            --save_steps 400 \
            --output_dir output/opt_350m \
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
        python train_opt350m.py \
            --data_path  $data_path/GUE/prom/$data \
            --kmer -1 \
            --run_name DNABERT2_${vocab}_${lr}_prom_${data}_seed${seed} \
            --model_max_length 70 \
            --use_lora False \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 10 \
            --fp16 \
            --save_steps 200 \
            --output_dir output/opt_350m \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --find_unused_parameters False
    done 

done
