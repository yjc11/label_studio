scene_folder=./scene_folder
output_dir=./output
model_file=/mnt/disk0/youjiachen/label_studio/uie_x/entity/model.xlsx
sheet_name="短文档一期"

do_convert(){
    python ./convert_ls_longtext_to_socr_format.py \
    --input_folder $scene_folder \
    --output_dir $output_dir \
    --model_file $model_file \
    --sheet_name $sheet_name
}

do_convert