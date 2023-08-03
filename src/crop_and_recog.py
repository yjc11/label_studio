import base64
import copy
import json
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import requests
import tritongrpcclient
from tqdm import tqdm

host = "192.168.106.7"
port = "8502"
http_url = f'{host}:{port}'
grpc_port = "6001"
grpc_url = host + ':' + grpc_port


# convert from LS percent units to pixels
def convert_from_ls(result):
    if 'original_width' not in result or 'original_height' not in result:
        return None

    value = result['value']
    w, h = result['original_width'], result['original_height']

    if all([key in value for key in ['x', 'y', 'width', 'height']]):
        return (
            w * value['x'] / 100.0,
            h * value['y'] / 100.0,
            w * value['width'] / 100.0,
            h * value['height'] / 100.0,
        )


# convert from pixels to LS percent units
def convert_to_ls(x, y, width, height, original_width, original_height):
    return (
        x / original_width * 100.0,
        y / original_height * 100.0,
        width / original_width * 100.0,
        height / original_height * 100,
    )


def convert_rect(rrect):
    x, y, w, h, theta = rrect
    norm_theta = theta * math.pi / 180
    if w > 0 and h > 0:
        w_angle, h_angle = [norm_theta, norm_theta + math.pi / 2]
    elif w < 0 and h > 0:
        w, h = [h, -w]
        w_angle, h_angle = [norm_theta + math.pi / 2, norm_theta + math.pi]
    elif w < 0 and h < 0:
        [w, h] = [-w, -h]
        [w_angle, h_angle] = [norm_theta + math.pi, norm_theta + 1.5 * math.pi]
    else:
        [w, h] = [-h, w]
        [w_angle, h_angle] = [norm_theta + 1.5 * math.pi, norm_theta + 2 * math.pi]

    horiV = np.array([math.cos(w_angle), math.sin(w_angle)]) / np.linalg.norm(
        [math.cos(w_angle), math.sin(w_angle)]
    )
    vertV = np.array([math.cos(h_angle), math.sin(h_angle)]) / np.linalg.norm(
        [math.cos(h_angle), math.sin(h_angle)]
    )

    p0 = np.array([x, y])
    p1 = (p0 + w * horiV).astype(np.float32)
    p2 = (p1 + h * vertV).astype(np.float32)
    p3 = (p2 - w * horiV).astype(np.float32)

    return [p0.tolist(), p1.tolist(), p2.tolist(), p3.tolist()]


def l2_norm(pt0, pt1):
    return np.sqrt((pt0[0] - pt1[0]) ** 2 + (pt0[1] - pt1[1]) ** 2)


def crop_images(img, bboxes):
    n = bboxes.shape[0]
    mats = []
    for i in range(n):
        bbox = bboxes[i]
        ori_w = np.round(l2_norm(bbox[0], bbox[1]))
        ori_h = np.round(l2_norm(bbox[1], bbox[2]))
        new_w = ori_w
        new_h = ori_h
        src_3points = np.float32([bbox[0], bbox[1], bbox[2]])
        dest_3points = np.float32([[0, 0], [new_w, 0], [new_w, new_h]])
        M = cv2.getAffineTransform(src_3points, dest_3points)
        m = cv2.warpAffine(img, M, (int(new_w), int(new_h)))
        mats.append(m)

    return mats


def SendReqWithRest(client, ep, req_body):
    def post(ep, json_data=None, timeout=10000):
        url = 'http://{}/v2/idp/ocr_app/infer'.format(ep)
        r = client.post(url=url, json=json_data, timeout=timeout)
        return r

    try:
        r = post(ep, req_body)
        return r
    except Exception as e:
        print('Exception: ', e)


def grpc_client(model_name, image, boxes):
    url = grpc_url
    triton_client = tritongrpcclient.InferenceServerClient(url=url)

    image = image.astype("uint8")
    inputs = []
    outputs = []
    inputs.append(tritongrpcclient.InferInput("image", image.shape, 'UINT8'))
    inputs.append(tritongrpcclient.InferInput('boxes', boxes.shape, 'FP32'))

    inputs[0].set_data_from_numpy(image)
    inputs[1].set_data_from_numpy(boxes)

    outputs.append(tritongrpcclient.InferRequestedOutput("texts"))
    outputs.append(tritongrpcclient.InferRequestedOutput("texts_score"))

    results = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    texts = results.as_numpy("texts")
    text_score = results.as_numpy("texts_score")
    texts = [str(text, 'utf-8') for text in texts]
    return texts[0]


def exec_transformer(image):
    recog = "transformer-v2.8-gamma-faster"

    client = requests.Session()
    ep = http_url
    bytes_data = cv2.imencode('.jpg', image)[1].tobytes()
    b64enc = base64.b64encode(bytes_data).decode()
    params = {
        'sort_filter_boxes': True,
        'rotateupright': False,
        'support_long_image_segment': True,
        'refine_boxes': True,
        'recog': recog,
    }
    req_data = {'param': params, 'data': [b64enc]}
    r = SendReqWithRest(client, ep, req_data)

    if r.status_code != 200:
        print("ERROR: can't get text result after transformer-v2.8-gamma")
        return None
    res = r.json()
    return res["result"]["texts"][0]


def long_text_crop_and_recog(label_path, output_path, max_workers=10):
    """
    适用长文档标注格式,抠小碎图,然后获取识别结果并写入label_studio格式json文件中
    """
    Path(output_path).mkdir(exist_ok=True, parents=True)
    with open(label_path, 'r') as f:
        raw_result = json.load(f)

    total_anno_num = 0
    for i in raw_result:
        for j in i['annotations'][0]['result']:
            if j['type'] == 'labels':
                total_anno_num += 1
    pbar = tqdm(total=total_anno_num)

    def process_task(task, pbar):
        page_info = task['data']['document']
        cur_url = [i['page'] for i in page_info]

        # Parse bboxes
        id2text = dict()
        for label in task['annotations'][0]['result']:
            if label['type'] == 'rectangle':
                id = label['id']
                num = int(label['to_name'].split('_')[1])
                page = f"page_{num:03d}"

                x, y, w, h = convert_from_ls(label)
                angle = label['value']['rotation']
                box = convert_rect([x, y, w, h, angle])

                image_url = next(filter(lambda x: page in x, cur_url))
                response = requests.get(image_url)
                if response.status_code != 200:
                    break
                bytes_data = response.content
                bytes_arr = np.frombuffer(bytes_data, np.uint8)
                img = cv2.imdecode(bytes_arr, cv2.IMREAD_COLOR)
                croped_img = crop_images(img, np.array([box]))[0]
                res = exec_transformer(croped_img)
                id2text[id] = res
                label['meta'] = {'text': [res]}

            elif label['type'] == 'labels':
                id = label['id']
                res = id2text[id]
                label['meta'] = {'text': [res]}
                pbar.update(1)

    with ThreadPoolExecutor(max_workers) as executor:
        futures = [executor.submit(process_task, task, pbar) for task in raw_result]
        for future in as_completed(futures):
            future.result()

    pbar.close()

    basename = Path(label_path).stem
    oup_path = Path(output_path) / f'recog_{basename}.json'
    with open(oup_path, 'w') as f:
        json.dump(raw_result, f, ensure_ascii=False, indent=2)

    return oup_path


def short_text_crop_and_recog(label_path, output_path, max_workers=10):
    """
    适用短文档标注格式,抠小碎图,然后获取识别结果并写入label_studio格式json文件中
    """
    Path(output_path).mkdir(exist_ok=True, parents=True)
    with open(label_path, 'r') as f:
        raw_result = json.load(f)

    total_anno_num = 0
    for i in raw_result:
        for j in i['annotations'][0]['result']:
            if j['type'] == 'labels':
                total_anno_num += 1
    pbar = tqdm(total=total_anno_num)

    def process_task(task):
        # Parse bboxes
        image_url = task['data']['Image']
        new_result_list = list()
        for label in task['annotations'][0]['result']:
            if label['type'] == 'textarea':
                continue

            elif label['type'] == 'labels':
                text_label = copy.deepcopy(label)
                text_label['from_name'] = 'transcription'
                text_label['type'] = 'textarea'
                del text_label['value']['labels']

                x, y, w, h = convert_from_ls(label)
                angle = label['value']['rotation']
                box = convert_rect([x, y, w, h, angle])

                response = requests.get(image_url)
                if response.status_code != 200:
                    break
                bytes_data = response.content
                bytes_arr = np.frombuffer(bytes_data, np.uint8)
                img = cv2.imdecode(bytes_arr, cv2.IMREAD_COLOR)
                croped_img = crop_images(img, np.array([box]))[0]

                recog_res = []
                shape = croped_img.shape
                boxes_model = (
                    np.array([0, 0, shape[1], 0, shape[1], shape[0], 0, shape[0]])
                    .reshape((-1, 4, 2))
                    .astype(np.float32)
                )
                recog_res.append(exec_transformer(croped_img))
                recog_res.append(grpc_client("crnn_dataelem", croped_img, boxes_model))
                recog_res.append(grpc_client("ctc_revive_1.2", croped_img, boxes_model))
                counts = Counter(recog_res).most_common(1)
                if counts[0][1] > 1:
                    res = recog_res[0]
                else:
                    # print('wrong')
                    res = recog_res[0]
                    res = '#wrong#' + res

                text_label['value']['text'] = [res]

                new_result_list.append(label)
                new_result_list.append(text_label)

                pbar.update(1)

            else:
                new_result_list.append(label)

        task['annotations'][0]['result'] = new_result_list

    with ThreadPoolExecutor(max_workers) as executor:
        futures = [executor.submit(process_task, task) for task in raw_result]
        for future in as_completed(futures):
            future.result()

    pbar.close()

    basename = Path(label_path).stem
    with open(Path(output_path) / f'recog_{basename}.json', 'w') as f:
        json.dump(raw_result, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    label_path = '/home/youjiachen/workspace/yangxiaojing/drawdown.json'
    dst = '/home/youjiachen/workspace/yangxiaojing/res/'
    Path(dst).mkdir(parents=True, exist_ok=True)

    # long_text_crop_and_recog(label_path, dst, 30)
    short_text_crop_and_recog(label_path, dst, 10)
