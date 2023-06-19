import os
import shutil

from file_func import check_folder


from pathlib import Path
from tqdm import tqdm


def copy_file(src_file, dest_dir):
    """
    复制单个文件到目标目录。

    参数：
    src_file：要复制的源文件的完整路径和文件名。
    dest_dir：目标目录的完整路径。

    返回值：
    无返回值。
    """
    if not os.path.isfile(src_file):
        raise ValueError("源文件不存在或者不是文件。")
    # if not os.path.isdir(dest_dir):
    #     raise ValueError("目标目录不存在或者不是目录。")

    shutil.copy(src_file, dest_dir)


def copy_directory(src_dir, dest_dir):
    """
    复制整个目录（包括其中的所有文件和子目录）到目标目录。

    参数：
    src_dir：要复制的源目录的完整路径。
    dest_dir：目标目录的完整路径。

    返回值：
    无返回值。
    """
    if not os.path.isdir(src_dir):
        raise ValueError("源目录不存在或者不是目录。")
    if not os.path.isdir(dest_dir):
        raise ValueError("目标目录不存在或者不是目录。")

    shutil.copytree(src_dir, dest_dir)


def copy_images(src_dir, dst_dir):
    """
    Copies all non-hidden files in subdirectories of `src_dir` to `dst_dir`.
    The copied files are renamed by concatenating their parent directory name
    and their original file name with an underscore.

    Args:
        src_dir (str or Path): Path to the source directory.
        dst_dir (str or Path): Path to the destination directory.
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    img_files = list(src_dir.glob('*/[!.]*'))
    for img_file in tqdm(img_files):
        prefix = img_file.parent.name
        page = img_file.name
        filename = f"{prefix}_{page}"
        shutil.copyfile(img_file, dst_dir / filename)


if __name__ == "__main__":
    # cwd = Path().cwd()
    # src = Path('/Volumes/T7-500G/数据/基础检测模型数据整理0131/长文本数据集/第一批标注数据')
    # dst = Path('/Volumes/T7-500G/数据/基础检测模型数据整理0131/长文本数据集/Images')

    # img_list = list(src.glob('*/*/[!.]*'))
    # print(img_list)
    # for img in tqdm(img_list):
    #     prefix = img.parents[0].name
    #     filename = prefix + '_' + img.name
    #     # print(filename)
    #     # print(prefix)
    #     copy_file(img, dst / filename)
    # check_folder(dst)

    # copy_images(img_folers_path, dst)

    # selected_list = ['银行流水', '征信报告', '受托报告-表格', '基金招募书-表格', '中粮糖价报盘表']
    # img_src = list(
    #     Path('/Users/youjiachen/Desktop/projects/基础检测模型/datasets/txts').glob('[!.]*')
    # )
    # dst = '/Users/youjiachen/Desktop/projects/基础检测模型/shuya/txts'
    # for i in tqdm(img_src):
    #     if i.name.split('_')[0] in selected_list:
    #         shutil.copy(i, dst)
    # print(len(img_src))
    pass
