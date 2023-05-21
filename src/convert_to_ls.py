import os
import json
import urllib.parse
import argparse


def trans_long_textie(data_dir, output_file, url_prefix):
    content = []
    for dirname in os.listdir(data_dir):
        sub_dir = os.path.join(data_dir, dirname)
        pages = os.listdir(sub_dir)
        pages = sorted(pages)
        document = {'document': [], 'Name': dirname}
        for page in pages:
            page_url = "{}/{}/{}".format(url_prefix, dirname, page)
            page_url = urllib.parse.quote(page_url, safe='://')
            page_info = {"page": page_url}
            document['document'].append(page_info)
        content.append(document)

    with open(output_file, 'w') as fout:
        json.dump(content, fout,  indent = 4)


def trans_textie(data_dir, output_file, url_prefix):
    content = []
    for dirname in os.listdir(data_dir):
        sub_dir = os.path.join(data_dir, dirname)
        pages = os.listdir(sub_dir)
        pages = sorted(pages)
        
        for idx, page in enumerate(pages):
            document = {}
            page_index = "{}_P{:03d}".format(dirname, idx)
            page_url = "{}/{}/{}".format(url_prefix, dirname, page)
            page_url = urllib.parse.quote(page_url, safe='://')
            document['ocr'] = page_url
            document['Index'] = page_index
            content.append(document)

    with open(output_file, 'w') as fout:
        json.dump(content, fout,  indent = 4)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--data_dir',
                        help='input data path',
                        type=str,
                        default=None)
    parser.add_argument('-o',
                        '--output_file',
                        help='output file path',
                        type=str,
                        default=None)
    parser.add_argument('-u',
                        '--url_prefix',
                        help='url prefix path',
                        type=str,
                        default='http://192.168.106.8/datasets')

    parser.add_argument('-t',
                        '--type',
                        help='annotation task type, long or short',
                        type=str,
                        default='long')
    
    return parser.parse_args()


def main():
    args = get_args()
    url_prefix = args.url_prefix
    basename = os.path.basename(args.data_dir)
    url_prefix_task = "{}/{}".format(url_prefix, basename)

    if args.type == 'long':
        trans_long_textie(args.data_dir, args.output_file, url_prefix_task)
    else:
        trans_textie(args.data_dir, args.output_file, url_prefix_task)


if __name__ == '__main__':
    main()
