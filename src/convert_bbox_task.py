import argparse
import json
import os
import random
import urllib.parse
from collections import defaultdict

import requests

random.seed(123)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--data_dir', help='input data path', type=str, default=None
    )
    parser.add_argument(
        '-o', '--output_file', help='output file path', type=str, default=None
    )
    parser.add_argument(
        '-u',
        '--url_prefix',
        help='url prefix path',
        type=str,
        default='http://192.168.106.8/datasets',
    )

    parser.add_argument(
        '-n', '--cnt', help='max count for each category', type=int, default=10000
    )

    parser.add_argument(
        '-s', '--start', help='start index of dataset', type=int, default=0
    )

    return parser.parse_args()


def list_names(dir_path):
    dirs = os.listdir(dir_path)

    file_paths = []
    for d in dirs:
        par_dir = os.path.join(dir_path, d)
        if os.path.isdir(par_dir):
            files = os.listdir(par_dir)
            for f in files:
                file_path = os.path.join(par_dir, f)
                file_paths.append(file_path)

    return file_paths


def covnert_bboxes_tasks(data_dir, output_file, url_prefix, n=100, s=0):
    files = []
    if os.path.isfile(data_dir):
        files = [f.rstrip() for f in open(data_dir).readlines()]
    else:
        files = list_names(data_dir)

    # sample and shuffle
    tmp = defaultdict(list)
    for ind, f in enumerate(files):
        document = {}
        local_dir, par_dir, filename = f.rsplit('/', 2)
        tmp[par_dir].append(f)

    part1_files = []
    for k in tmp.keys():
        random.shuffle(tmp[k])
        e = s + n
        part1_files.extend(tmp[k][s:e])

    random.shuffle(part1_files)
    content1 = []
    for ind, f in enumerate(part1_files):
        document = {}
        local_dir, par_dir, filename = f.rsplit('/', 2)
        page_url = "{}/{}/{}".format(url_prefix, par_dir, filename)
        page_url = urllib.parse.quote(page_url, safe='://')
        document['Image'] = page_url
        document['Index'] = ind + 1
        document['Tag'] = par_dir
        content1.append(document)

    with open(output_file, 'w') as fout:
        json.dump(content1, fout, indent=4)


def split_data(url_prefix, files):
    # split data:
    cnt = 0
    for ind, f in enumerate(files):
        document = {}
        local_dir, task_name, par_dir, filename = f.rsplit('/', 3)
        page_url = "{}/{}/{}".format(url_prefix, par_dir, filename)
        out_dir = os.path.join('./data', task_name, "dataset_v1", par_dir)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_file = os.path.join(out_dir, filename)
        response = requests.get(page_url)
        bin_image = None
        if response.status_code != 200:
            continue
        cnt += 1
        with open(out_file, 'wb') as fout:
            fout.write(response.content)

    print('part1 cnt', cnt)


def main():
    args = get_args()
    url_prefix = args.url_prefix
    n = args.cnt
    start = args.start

    if not os.path.isfile(args.data_dir):
        basename = os.path.basename(args.data_dir)
    else:
        basename = open(args.data_dir).readlines()[0].rsplit('/', 3)[-3]

    url_prefix_task = "{}/{}".format(url_prefix, basename)
    covnert_bboxes_tasks(args.data_dir, args.output_file, url_prefix_task, n, start)


if __name__ == '__main__':
    main()
