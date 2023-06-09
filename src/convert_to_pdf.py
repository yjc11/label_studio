from io import BytesIO
from pathlib import Path

from PIL import Image
from PyPDF2 import PdfMerger


def images_to_pdf(image_paths, output_path):
    """
    image_paths : image_paths / xxx.png
    output_path :  xxx.pdf
    """
    # 创建一个PdfFileMerger对象
    pdf_merger = PdfMerger()

    # 遍历所有图片路径
    for image_path in image_paths:
        # 打开图片
        image = Image.open(image_path)

        # 将图片转换为PDF文件
        pdf_file = image.convert('RGB')

        # 将PDF文件转换为BytesIO对象
        pdf_bytes = BytesIO()
        pdf_file.save(pdf_bytes, format='pdf')

        # 将BytesIO对象添加到PdfMerger对象中
        pdf_merger.append(pdf_bytes)

    # 将所有PDF文件合并为一个PDF文件
    with open(output_path, 'wb') as f:
        pdf_merger.write(f)


if __name__ == "__main__":
    img_path = Path(
        '/Users/youjiachen/Desktop/projects/label_studio_mgr/data/test_rotate/merge'
    ).glob('[!.]*')
    oup_path = './ccc.pdf'
    # images_to_pdf(img_path, oup_path)
    import urllib

    img_path = '/Users/youjiachen/Desktop/projects/label_studio_mgr/data/test_rotate/水平/check_img'
    imgs = Path(img_path).glob('[!.]*')
    for img in imgs:
        basename = img.name
        url_basename = urllib.parse.quote(basename, safe='://')
        try:
            img.rename(img.with_name(url_basename))
        except:
            continue
