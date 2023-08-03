scene_folder=./scene_folder
output_dir=./output
model_file=$scene_folder/model.xlsx

do_convert(){
    python ./convert_ls_longtext_to_socr_format.py \
    --input_folder $scene_folder \
    --output_dir $output_dir \
    --model_file $model_file
}

do_convert