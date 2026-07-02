master_port=$((RANDOM % (65535 - 49152 + 1) + 49152))
filename=$(basename "$0" | cut -f 1 -d '.')
timestamp=$(date +"%Y%m%d_%H%M")
dir_path=
output_dir=outputs/JING_train_stage2/${filename}_${timestamp}
model_name_or_path=outputs/JING_train_stage1/JING_stage1
anno_path=
data_path=

cd $dir_path
PYTHONPATH=$dir_path:$PYTHONPATH \
python buildingmllm/train/train_mem.py \
    --model_name_or_path $model_name_or_path \
    --data_path $data_path \
    --anno_path ${ANNO_PATH:-data/anno_data/} \
    --output_dir $output_dir \
    --version v1 \
    --model_max_length 2048 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --eval_steps 100 \
    --save_strategy "no" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 3e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 100 \
    --bf16 True \
    --fix_llm True \
    --fix_pointnet True \
    --report_to wandb \
    --run_name $filename \
    --gradient_checkpointing True \
    --stage_2 True \
    --conversation_types ${CONV_TYPES:-"0_simple_conversation" "9_1-8"} \
    --use_color True \
    --force_fsdp False \
    --use_lora True \
    --tune_layer 0 \
    --use_lora_only True \
    --use_ia3_only False \
    --use_adalora_only False \
    --use_lokr_only False \
    --use_loha_only False \
    --use_mt_prompt_only False \
    --use_prompt_only False

    
    #"0_simple_conversation"  #"9_1-8" 
    # "1_General_Visual_Recognition" 
    #"2_Knowledge_Capability" "3_Commonsense_Reasoning" "4_Advanced_Engineering_Reasoning" "5_Functional_Comparison" "6_Constraint_Reasoning" "7_Spatial_Relationship" "8_Embodied_Interaction" 
    
