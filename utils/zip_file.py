import os
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

# 定义要压缩的文件夹列表
src = '/Users/youjiachen/Downloads/v2_label_modified'
img_list = list(Path(src).glob('**/Labels'))


def zip_folder(folder_path, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                breakpoint()
                zipf.write(file_path, os.path.relpath(file_path, folder_path))
                



# 调用示例
# folder_path = img_list[0]  # 文件夹路径
for img_folder in img_list:
    basename = img_folder.parent.stem
    output_path = f'/Users/youjiachen/Desktop/projects/label_studio_mgr/output/zip_file/{basename}_labels__.zip'  # 输出zip文件路径
    zip_folder(img_folder, output_path)


# 压缩文件夹
# for img_folder in tqdm(img_list):
# def zip_file(img_folder):
#     shutil.make_archive(img_folder, 'zip', root_dir=img_folder, base_dir=img_folder)


# with ThreadPoolExecutor(max_workers=10) as e:
#     futures = [e.submit(zip_file, task) for task in img_list]
#     for future in as_completed(futures):
#         future.result()
