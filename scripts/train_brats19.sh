#!/bin/bash
# modify these augments if you want to try other datasets, splits or methods

now=$(date +"%Y%m%d_%H%M%S")
dataset='brats19'
# script_name='train_twoencoderbaseline_brats19.py'
script_name='train_model_semi_heri_brats19.py'

# mode=train
mode=${1:-train}

ratio='0.1'


## baseline methods
method='debug'
# method='twoencoderbaseline'
# method='twoencoderbaseline_0.5FP'

## ablation methods
# method='unet_loss_sup_fused'
# method='unet_loss_sup'
# method='unet_baseline0.01'
# method='unet_baseline0.1'
# method='supervised_withdisentangle'

# method='unet_heri_baseline_0.1loss_noisecons_0.1loss_predcons_0.1loss_fused2shared_spec_cons'
# method='unet_heri_crossfusion_baseline_0.1loss_noisecons_0.1loss_predcons_0.1loss_fused2shared_spec_cons2'
# method='unet_heri_crossfusion_baseline_0.1loss_noisecons_0.1loss_predcons_0.1loss_fused2shared_spec_cons4'


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