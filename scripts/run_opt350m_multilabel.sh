#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --gpus=a100:2
#SBATCH --job-name=deepsea_
#SBATCH -t 148:00:00
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=
#SBATCH -o /common/
#SBATCH -e /common/
source /common/zhanh/anaconda3/etc/profile.d/conda.sh
conda activate dna_peft
/common/zhanh/anaconda3/envs/dna_peft/bin/python

lr=3e-5

for seed in 42
do

        python DNA_llm_multilabel_350m.py \
            --kmer -1 \
            --run_name OPT_${vocab}_${lr}_deppsea_seed${seed} \
            --model_max_length 512 \
            --use_lora \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
            --num_train_epochs 5 \
            --fp16 \
            --save_steps 2000 \
            --output_dir output/opt_350m_deepsea \
            --evaluation_strategy steps \
            --eval_steps 2000 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --find_unused_parameters False

done
