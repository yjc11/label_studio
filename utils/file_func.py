import os
import cv2
import urllib
import shutil
import random
import imagehash
import hashlib

from pathlib import Path
from PIL import Image
from tqdm import tqdm


def check_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        shutil.rmtree(folder)
        os.makedirs(folder)


def sample_files(source_dir, target_dir, sampling_rate):
    # Convert path strings to Path objects
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # Get list of all files in source directory
    file_list = [f.name for f in source_path.iterdir() if f.is_file()]

    # Compute number of files to sample
    sample_size = int(len(file_list) * sampling_rate)

    # Randomly select files to sample
    sample_index = random.sample(range(len(file_list)), sample_size)
    sample_result = [file_list[i] for i in sample_index]

    # Copy sampled files to target directory
    for file_name in sample_result:
        source_file_path = source_path / file_name
        target_file_path = target_path / file_name
        shutil.copyfile(source_file_path, target_file_path)


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


if __name__ == "__main__":
    # src = '/Users/youjiachen/Desktop/projects/label_studio_mgr/workspace/gouxuankuang/test/check_img'
    # dst = '/Users/youjiachen/Desktop/projects/label_studio_mgr/workspace/gouxuankuang/test/sample_100'
    # check_folder(dst)
    # sample_files(src, dst, 0.1)

    raw_images_path = (
        '/Users/youjiachen/Desktop/projects/label_studio_mgr/data/银行流水评测集_images'
    )
    rotate_images_path = (
        '/Users/youjiachen/Desktop/projects/label_studio_mgr/data/test_rotate/水平/Images'
    )

    raw_images = list(Path(raw_images_path).glob('[!.]*'))
    raw_images_set = {_.name for _ in raw_images}

    rotate_images = list(Path(rotate_images_path).glob('[!.]*'))
    rotate_images_set = {_.name for _ in rotate_images}

    del_img = raw_images_set - rotate_images_set
    for img in raw_images:
        if img.name in del_img:
            im_show = cv2.imread(str(img))
            decode_name = urllib.parse.quote(img.name, safe='://')
            print(decode_name)
            # cv2.imshow(f'{decode_name}', im_show)
            # cv2.waitKey(0)
