master_port=$((RANDOM % (65535 - 49152 + 1) + 49152))
filename=$(basename "$0" | cut -f 1 -d '.')
timestamp=$(date +"%Y%m%d_%H%M")
dir_path=
model_name_or_path=RunsenXu/PointLLM_7B_v1.2  
output_dir=outputs/JING_train_stage1/${filename}_${timestamp}
anno_path=data/anno_data
data_path=data

cd $dir_path
PYTHONPATH=$dir_path:$PYTHONPATH
python buildingmllm/train/train_mem.py \
    --model_name_or_path $model_name_or_path \
    --data_path $data_path \
    --anno_path $anno_path \
    --output_dir $output_dir \
    --version v1 \
    --model_max_length 2048 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tune_layer 2 \
    --bf16 True \
    --fix_llm True \
    --fix_pointnet True \
    --gradient_checkpointing True \
    --report_to wandb \
    --run_name $filename \
    --use_color True \

