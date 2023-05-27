import argparse
import glob
import os

import fitz


def transpdf2png(filename, output_dir):
    doc = fitz.open(filename)
    basename = os.path.basename(filename)
    filename = basename.rsplit('.', 1)[0]
    for page in doc:
        dpis = [72, 144, 200]
        pix = None
        for dpi in dpis:
            pix = page.get_pixmap(dpi=dpi)
            if min(pix.width, pix.height) >= 1600:
                break

        out_name = "page_{:03d}.png".format(page.number)
        out_dir = os.path.join(output_dir, filename)
        out_file = os.path.join(output_dir, filename, out_name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        pix.save(out_file)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--data_dir', help='input data path', type=str, default=None
    )
    parser.add_argument(
        '-o', '--output_dir', help='output file path', type=str, default=None
    )
    return parser.parse_args()


def main():
    args = get_args()
    inputs = glob.glob(os.path.join(args.data_dir, "*.pdf"))
    for file in inputs:
        transpdf2png(file, args.output_dir)


if __name__ == '__main__':
    main()
