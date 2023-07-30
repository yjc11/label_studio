import hashlib
import json
import os
import random
import re
import shutil
import urllib
from pathlib import Path

import cv2
import imagehash
from PIL import Image
from tqdm import tqdm


def check_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        shutil.rmtree(folder)
        os.makedirs(folder)


def sample_files(source_dir, target_dir, sample_nums):
    # Convert path strings to Path objects
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # Get list of all files in source directory
    file_list = list(source_path.glob('[!.]*'))

    # Compute number of files to sample
    sample_size = min(sample_nums, len(file_list))

    # Randomly select files to sample
    sample_index = random.sample(range(len(file_list)), sample_size)
    sample_result = [file_list[i] for i in sample_index]

    # Copy sampled files to target directory
    for file in sample_result:
        if file.is_file():
            source_file_path = source_path / file.name
            target_file_path = target_path / file.name
            shutil.copyfile(source_file_path, target_file_path)
        elif file.is_dir():
            source_file_path = source_path / file.name
            target_file_path = target_path / file.name
            shutil.copytree(source_file_path, target_file_path)


def image_deduplication(file_paths):
    filenames = list(Path(file_paths).glob('**/[!.]*'))
    # 读取图片并计算哈希值
    hashes = list()
    for filename in tqdm(filenames):
        if filename.is_file():
            hash = get_img_hash(filename)
            if hash in hashes:
                filename.unlink()
                print(f'del {filename.name}')
            else:
                hashes.append(hash)


def get_md5(file):
    file = open(file, 'rb')
    md5 = hashlib.md5(file.read())
    file.close()
    md5_values = md5.hexdigest()
    return md5_values


def get_img_hash(file):
    with Image.open(file) as img:
        hash = imagehash.phash(img)

    return hash


def rename_files():
    """
    短文档定制，因为短文档需要后缀有_page_xxx
    """
    src = '/mnt/disk0/youjiachen/label_studio/output/询证函-去摩尔纹'
    file_list = list(Path(src).glob('**/[!.]*.*'))

    for file in file_list:
        file_stem = file.stem
        file_suffix = file.suffix
        new_name = file_stem + '_page_000' + file_suffix
        file.rename(file.with_name(new_name))


def is_page_number(string):
    pattern = r"_page_\d+$"
    if re.search(pattern, string):
        return True
    else:
        return False


if __name__ == "__main__":
    """sample"""
    sp_size = 150
    src = '/Users/youjiachen/Desktop/长文档5个场景/MSDS'
    dst = '/Users/youjiachen/Desktop/长文档5个场景/MSDS采样'
    sample_files(src, dst, sp_size)
