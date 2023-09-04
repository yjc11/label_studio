scene_folder=/mnt/disk0/youjiachen/workspace/ELLM_datasets/output/短文档标注第一期
output_dir=./output
model_file=/mnt/disk0/youjiachen/label_studio/uie_x/entity/model.xlsx
sheet_name='23-7卡证表单'

do_convert(){
    python /mnt/disk0/youjiachen/label_studio/uie_x/entity/ocrstudio_to_socr_format/convert_ocrstudio_to_socr_format.py \
    --input_folder $scene_folder \
    --output_dir $output_dir \
    --model_file $model_file \
}

do_convert