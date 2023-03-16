from typing import Tuple

import cv2
import json
import re
from pathlib import Path

import numpy
import numpy as np
import pandas as pd
import pickle
import os
import torch
from torch import Tensor


def load_keypoints_info(info_path, video_file_name):
    """
    根据输入的视频文件名称，
    在info_path文件夹内寻找相应的关键点检测信息（关键点、meatadata），
    载入关键点检测信息
    :return:
    bodyparts: index object, shape: (number of parts, )
    裁剪的坐标信息集合，df_x, df_y, df_likelihood, numpy array, shape: (number of parts, number of frame)
    cropping_parameters = [x1, x2, y1, y2] 检测时的裁剪区域
    """
    video_name = str(Path(video_file_name).stem)  # 获取视频去掉文件后缀的名称
    # print(len(video_name))
    # print(video_name)
    # path下含有analyze，metadata两个文件夹存放检测数据
    analyze_path = os.path.join(info_path, 'analyze')
    analyze_list = os.listdir(analyze_path)
    meta_path = os.path.join(info_path, 'metadata')
    meta_list = os.listdir(meta_path)
    if len(analyze_list) == 0 or len(meta_list) == 0:
        print('文件夹内文件数量为0，请检查保存检测文件的analyze与metadata文件夹')

    # 遍历analyze与metadata中的文件
    for an_name, meta_name in zip(analyze_list, meta_list):
        # 正则匹配与视频名称对应的检测文件，载入关键点检测数据
        # 正则匹配对应的metadata文件，获取检测时的裁剪参数
        # print(an_name)
        # print(meta_name)
        # print(re.match(video_name, an_name))
        # print(re.match(video_name, meta_name))
        if re.match(video_name, an_name) and re.match(video_name, meta_name):
            df = pd.read_hdf(os.path.join(analyze_path, an_name))
            nframes = len(df.index)
            bodyparts = df.columns.get_level_values("bodyparts")[::3]
            df_x, df_y, df_likelihood = df.values.reshape((nframes, -1, 3)).T

            with open(os.path.join(meta_path, meta_name), "rb") as handle:
                metadata = pickle.load(handle)
            cropping = metadata["data"]["cropping"]
            [x1, x2, y1, y2] = metadata["data"]["cropping_parameters"]
            cropping_parameters = [x1, x2, y1, y2]
            # print(cropping)
            # print(x1, x2, y1, y2)

            # 置信率较低的点的坐标调整为上一帧该点的坐标
            for n in range(0, len(bodyparts)):
                for i in range(1, nframes):
                    if df_likelihood[n, i] < 0.5:
                        df_x[n, i] = df_x[n, i - 1]
                        df_y[n, i] = df_y[n, i - 1]
                        df_likelihood[n, i] = 1.0

            '''
            提取ROI数量不同时需要修改的地方
            一块ROI就要用下面四行
            '''
            # df_x = np.expand_dims(np.sum(df_x, axis=0) / 2, axis=0)
            # df_y = np.expand_dims(np.sum(df_y, axis=0) / 2, axis=0)
            # df_likelihood = np.expand_dims(np.sum(df_likelihood, axis=0) / 2, axis=0)
            # bodyparts = ['front']

            return bodyparts, df_x, df_y, df_likelihood, cropping, cropping_parameters

    print('未找到与输入视频对应的检测文件，请检查保存检测文件的analyze与metadata文件夹')
    return IOError


# 保存RGB图像与ST图像数据，每个keypoint都有有一张RGB+一张ST图像，自动分文件夹保存数据，json内保存标签信息
def save_img_data(path, file_name, ST_arr, save_size, video_id, label_name):
    """
    提取ROI数量不同时需要修改的地方
    注意看下面有注释的地方
    """
    label_name_dict = {'0': 0, '头部': 1, '前足': 2, '前中足': 3, '后中足': 4,
                      '后足': 5, '腹部': 6, '翅膀': 7}
    # label_name_dict = {'0': 0, 'head': 1, 'foreleg': 2, 'front-mid': 3, 'hind-mid': 4,
    #                   'hindleg': 5, 'abdomen': 6, 'wing': 7}
    dict_label = {'video_id': int(video_id), 'label_name': label_name, 'label': int(label_name_dict[str(label_name)])}
    file_name = file_name + '_' + str(label_name_dict[str(label_name)])
    OutputPath = path + '/label/' + file_name

    if str(label_name) == '0':
        front_ST_img_path = path + '/ST/front/' + str(label_name) + '/' + file_name + '0.png'
        posterior_ST_img_path = path + '/ST/posterior/' + str(label_name) + '/' + file_name + '1.png'
        cv2.imencode('.png', cv2.resize(cv2.cvtColor(ST_arr[0] * 255, cv2.COLOR_RGB2BGR),
                                        (save_size, save_size)))[1].tofile(front_ST_img_path)
        # 一块ROI就要注释下面一行
        cv2.imencode('.png', cv2.resize(cv2.cvtColor(ST_arr[1] * 255, cv2.COLOR_RGB2BGR),
                                        (save_size, save_size)))[1].tofile(posterior_ST_img_path)

    elif label_name in ['0', '头部', '前足', '前中足']:
        front_ST_img_path = path + '/ST/front/' + str(label_name) + '/' + file_name + '.png'
        cv2.imencode('.png', cv2.resize(cv2.cvtColor(ST_arr[0] * 255, cv2.COLOR_RGB2BGR),
                                        (save_size, save_size)))[1].tofile(front_ST_img_path)

    elif label_name in ['0', '后足', '后中足', '后足', '腹部', '翅膀', '产卵器']:
        posterior_ST_img_path = path + '/ST/posterior/' + str(label_name) + '/' + file_name + '.png'
        # 两块ROI，底下就是ST_arr[1]；一块ROI，就要改成ST_arr[0]
        cv2.imencode('.png', cv2.resize(cv2.cvtColor(ST_arr[1] * 255, cv2.COLOR_RGB2BGR),
                                        (save_size, save_size)))[1].tofile(posterior_ST_img_path)
    '''
    if label_name in ['0', '头部', '前足', '前中足']:
        for nb in range(0, len(bodyparts)):
            img_path = path + '/front/' + str(label_name) + '/' + file_name + '.png'
            cv2.imencode('.png', cv2.resize(RGB_arr[..., nb], (save_size, save_size)))[1].tofile(img_path)
            cv2.imencode('.png', cv2.resize(ST_arr[..., nb], (save_size, save_size)))[1].tofile(img_path)
    elif label_name in ['0', '后足', '后中足', '后足', '腹部', '翅膀']:
        for nb in range(0, len(bodyparts)):
            img_path = path + '/posterior/' + str(label_name) + '/' + file_name + '.png'
            cv2.imencode('.png', cv2.resize(RGB_arr[..., nb], (save_size, save_size)))[1].tofile(img_path)
            cv2.imencode('.png', cv2.resize(ST_arr[..., nb], (save_size, save_size)))[1].tofile(img_path)
    '''
    # print(img_path)
    with open(OutputPath + '.json', 'w+') as f:
        json.dump(dict_label, f)


def show_st_image(st_img_rec: Tensor, point_name: dict, is_show=False) -> numpy.ndarray:
    # st Shape: (num_parts, roi_size, roi_size, 3) with RGB mode
    st = st_img_rec.cpu().numpy()
    if is_show:
        for i in range(0, st_img_rec.shape[0]):
            cv2.imshow(point_name[i], st[i][:, :, ::-1])
        cv2.waitKey(1)
    return st


def get_display_label(confidence: Tensor, pred_label: Tensor) -> Tuple[int, int, float]:
    confidence[confidence < 0.6] = 0
    pred_label[confidence < 0.6] = 0
    display_label: int = 0
    display_part: int = -1
    if torch.equal(pred_label.cpu(), torch.zeros_like(pred_label, dtype=torch.int64).cpu()):
        display_conf = confidence.cpu().numpy()[0]
        return display_part, display_label, display_conf
    else:
        confidence[pred_label < 1] = 0
        _, parts = torch.max(confidence.data, 0)
        display_part = parts.data.item()
        display_conf = _.data.item()
        display_label = pred_label[display_part].data.item()
        return display_part, display_label, display_conf


def frames_to_timecode(framerate, frames):  # 将帧数转换为时间，返回值为三个整数：分钟\秒\毫秒
    """
    视频 通过视频帧转换成时间
    :param framerate: 视频帧率
    :param frames: 当前视频帧数
    :return:时间（00:01:01）
    """
    time = '{0:02d}-{1:02d}-{2:02d}'.format(int(frames / (60 * framerate) % 60),
                                            int(frames / framerate % 60),
                                            int((frames % framerate) * (1000 / framerate)))
    time = re.split('-', time)
    time = list(map(int, time))
    if time[0] == 0:
        del time[0]
        length = 2
        return length, time
    else:
        length = 3
        return length, time


def frames_to_timecode2(framerate, frames):  # 将帧数转换为时间，返回值为三个整数：分钟\秒\帧
    """
    视频 通过视频帧转换成时间
    :param framerate: 视频帧率
    :param frames: 当前视频帧数
    :return:时间（00:01:01）
    """
    time = '{0:02d}-{1:02d}-{2:02d}'.format(int(frames / (60 * framerate) % 60),
                                            int(frames / framerate % 60),
                                            int(frames % framerate))
    time = re.split('-', time)
    time = list(map(int, time))
    min = time[0]
    sec = time[1]
    divide_frame = time[2]

    return min, sec, divide_frame


def get_frameID(time, framerate):  # 根据时间获取帧数
    time = re.split('-', time)
    time = list(map(int, time))
    frameID = time[0] * 60 * framerate + time[1] * framerate + time[2] / (1000 / framerate)
    frameID = int(frameID)
    return frameID


def generate_csv(start: list, end: list, motion: list, save_path, video_name, framerate=25):
    label_name_dict = {0: '0',
                       1: '头部',
                       2: '前足',
                       3: '前中足',
                       4: '后中足',
                       5: '后足',
                       6: '腹部',
                       7: '翅膀'}
    detection_res = pd.DataFrame(columns=['s分',
                                          's秒',
                                          's帧',
                                          'e分',
                                          'e秒',
                                          'e帧',
                                          '持续时间',
                                          '持续帧数',
                                          '行为',
                                          '开始帧',
                                          '结束帧'])
    motion_name = []
    for i in motion:
        motion_name.append(label_name_dict[i])

    for i in range(0, len(start)):
        smin, ssec, sdivide_frame = frames_to_timecode2(framerate, start[i])
        emin, esec, edivide_frame = frames_to_timecode2(framerate, end[i])
        detection_res = detection_res.append(pd.DataFrame({'s分': [smin],
                                                           's秒': [ssec],
                                                           's帧': [sdivide_frame],
                                                           'e分': [emin],
                                                           'e秒': [esec],
                                                           'e帧': [edivide_frame],
                                                           '持续时间': [(end[i] - start[i]) / framerate],
                                                           '持续帧数': [end[i] - start[i]],
                                                           '行为': [motion_name[i]],
                                                           '开始帧': [start[i]],
                                                           '结束帧': [end[i]]}))
    detection_res.to_csv(os.path.join(save_path, video_name + '.csv'), index=False)
    print('检测文件已保存至 ' + os.path.join(save_path, video_name + '.csv'))


def generate_check(check, start_check, end_check, motion_check, framerate):
    for i in range(0, len(start_check)):
        smin, ssec, sdivide_frame = frames_to_timecode2(framerate, start_check[i])
        emin, esec, edivide_frame = frames_to_timecode2(framerate, end_check[i])
        check = check.append(pd.DataFrame({'s分': [smin],
                                           's秒': [ssec],
                                           's帧': [sdivide_frame],
                                           'e分': [emin],
                                           'e秒': [esec],
                                           'e帧': [edivide_frame],
                                           '持续时间': [(end_check[i] - start_check[i]) / 25],
                                           '持续帧数': [end_check[i] - start_check[i]],
                                           '行为': [motion_check[i]],
                                           '开始帧': [start_check[i]],
                                           '结束帧': [end_check[i]]}))
    return check


def judge_motion_section(label_list: list, threshold=12):
    """
    给定每一帧的label预测结果，返回判定的行为区间
    :param label_list: 包含每一帧的label预测结果
    :param threshold: 判定是否发生行为的帧数阈值
    :return: 开始时间、结束时间、对应的行为名称
    """
    flag = 0
    start = []
    end = []
    motion = []
    now = 0

    for i, label in enumerate(label_list):
        if i == 0:
            now = label
            start.append(i)
            motion.append(label)
        elif i == len(label_list) - 1:
            end.append(i)
        elif now == label:
            flag = 0
        elif now != label:
            if flag < threshold:
                flag += 1
            elif flag >= threshold:
                flag = 0
                now = label
                motion.append(label)
                end.append(i - threshold)
                start.append(i - threshold)
                if int(now) in [1, 2]:
                    threshold = 12
                elif int(now) in [0, 5, 6]:
                    threshold = 15
                elif int(now) in [3, 4, 7]:
                    threshold = 20
    return start, end, motion


def save_img_rec(img_rec: numpy.ndarray, save_path, frame_index):
    """
    保存图片组
    :param img_rec: Shape: (num_parts, roi_size, roi_size, 3)
    :param frame_index: 帧索引
    :param save_path: ...
    """
    group = list(range(0, img_rec.shape[0]))
    for i in group:
        isExists = os.path.exists(os.path.join(save_path, str(i)))
        if not isExists:
            os.makedirs(os.path.join(save_path, str(i)))
    for i in group:
        img_save_path = os.path.join(save_path, str(i), str(frame_index) + '.png')
        cv2.imencode('.png', img_rec[i, ..., ::-1] * 255)[1].tofile(img_save_path)


if __name__ == '__main__':
    pass
