import json
from collections import Counter, defaultdict

import gradio as gr
import requests

U_TOKEN = 'cf4d118901d4fa2df115cc44cc51656838399513'
LS_EP = 'http://192.168.106.7:8080'


class LSClient(object):
    def __init__(self):
        pass

    def get_user_info(self, uid):
        url = '{}/api/users/{}'.format(LS_EP, uid)
        headers = {'Authorization': 'Token cf4d118901d4fa2df115cc44cc51656838399513'}
        r = requests.get(url, headers=headers).json()
        return r

    def get_export(self, pid):
        url = '{}/api/projects/{}/export?exportType=JSON'.format(LS_EP, pid)
        headers = {'Authorization': 'Token cf4d118901d4fa2df115cc44cc51656838399513'}
        r = requests.get(url, headers=headers).json()
        return r


LSC = LSClient()


def calc_statistic(data):
    # 定义标注计数器
    label_count = defaultdict(Counter)
    uid_info_map = dict()
    # 遍历标注数据
    for annotation in data:
        for anno_ in annotation['annotations']:
            labels = anno_['result']
            anno = anno_
            # labels = annotation['annotations'][0]['result']
            # anno = annotation['annotations'][0]
            uid = anno['completed_by']
            uinfo = uid_info_map.get(uid, None)
            if uinfo is None:
                uinfo = LSC.get_user_info(uid)
                uid_info_map[uid] = uinfo

            email = uinfo['email']
            username = uinfo['username']
            fname = uinfo['first_name']
            lname = uinfo['last_name']
            if fname and lname:
                username = "{}{}".format(fname, lname)

            aid = '{}'.format(username)
            for label in labels:
                if label.get('type', '') in ['choices', 'relation']:
                    if label['type'] == 'choices':
                        choice = label.get('value', {}).get('choices', [])
                        if choice:
                            label_types = ['质检:{}'.format(choice[0])]
                    else:
                        label_types = ['关系']

                    label_count[aid].update(label_types)
                elif 'value' in label:
                    label_types = label['value'].get('labels', [])
                    if isinstance(label_types, str):
                        label_types = [label_types]

                    label_count[aid].update(label_types)

    # 输出结果
    all_keys = []
    for aid, counter in label_count.items():
        for k, v in counter.items():
            all_keys.append(k)

    keys = sorted(list(set(all_keys)))
    header = ['标注员', '所有'] + keys
    rows = []
    for aid, counter in label_count.items():
        total = sum(list(counter.values()))
        row = [aid, total]
        for k in keys:
            row.append(counter.get(k, 0))

        rows.append(row)

    header_str = '| ' + ' | '.join(header) + ' |'
    sep_str = '| ' + ('--- | ' * len(header)).strip()
    rows_str = ''
    for r in rows:
        r_str = map(str, r)
        row_str = '| ' + ' | '.join(r_str) + ' |'
        if not rows_str:
            rows_str = row_str
        else:
            rows_str = rows_str + '\n' + row_str

    return '{}\n{}\n{}'.format(header_str, sep_str, rows_str)


def calc_count(input_file):
    # 读取导出的JSON文件
    with open(input_file.name, 'r') as f:
        data = json.load(f)

    return calc_statistic(data)


def calc_count1(pid):
    if len(pid) == 0:
        return '填写正确的ProjectID'

    data = LSC.get_export(pid)
    return calc_statistic(data)


def clear():
    return None, None


def main():
    with gr.Blocks() as demo:
        # gr.Markdown("## 框标注数量统计一")
        # input_file = gr.File(label="上传LabelStudio导出的标注结果(JSON格式)")
        # output_text = gr.Textbox(label="标注框数量:", lines=2)
        # btn = gr.Button("计算")
        # btn.click(calc_count, [input_file], output_text)

        gr.Markdown("## 框标注数量统计二")
        input_text1 = gr.Textbox(label="填写ProjectID")
        gr.Markdown("### 标注框数量:")
        # output_text1 = gr.Textbox(label="标注框数量:", lines=2)
        output_text1 = gr.Markdown(label="result")

        with gr.Row():
            btn1 = gr.Button("计算")
            btn1.click(calc_count1, [input_text1], output_text1)

            btn2 = gr.Button("重置")
            btn2.click(clear, [], [input_text1, output_text1])

        demo.launch(server_name="192.168.106.7", server_port=7070)


if __name__ == "__main__":
    main()
