#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

# modify these augments if you want to try other datasets, splits or methods
# dataset: ['isles']

dataset='isles'
ratio='0.1'

# mode=val #train/val
mode=${1:-train}


script_name='train_model_semi_heri_isles22.py'


#baseline methods
method='debug'
#method='unet_heri_crossfusion_baseline_0.1loss_noisecons_0.1loss_predcons_0.1loss_fused2shared_spec_cons4'


save_path=logs/$dataset/$ratio/$method
mkdir -p $save_path

log_file=$save_path/$now.log

echo "===============================" | tee -a $log_file
echo "Experiment config:"              | tee -a $log_file
echo "Dataset: $dataset"              | tee -a $log_file
echo "Semi ratio: $ratio"             | tee -a $log_file
echo "Method: $method"                | tee -a $log_file
echo "Start time: $(date)"            | tee -a $log_file
echo "Save path: $save_path"          | tee -a $log_file
echo "script_name: $script_name"      | tee -a $log_file
echo "===============================" | tee -a $log_file

export PYTHONPATH=/users/fxzhou/scratch/project/multi-modality/semi-HDCL:$PYTHONPATH

# SCRIPT_DST=$save_path/$script_name
SCRIPT_DST=$script_name


python "$SCRIPT_DST" \
    --semi_rate $ratio \
    --save_path $save_path \
    --mode $mode \
    2>&1 | tee -a $log_file

echo "===============================" | tee -a $log_file
echo "Finished at: $(date)"           | tee -a $log_file
echo "===============================" | tee -a $log_file