import sys
import os
import time

import pandas as pd
from video_processer import GenerateStImageFrameByFrame, draw_detect_result_on_frame
from data_processer import generate_csv, judge_motion_section
from pathlib import Path
import torch
import cv2
from TrainModel.utils import predict_image
from torchvision import transforms
from Modules.data_processer import show_st_image, get_display_label, save_img_rec
from colour import Color


key_points_dict = {0: 'front', 1: 'posterior'}
label_name_dict = {0: '0', 1: 'Head', 2: 'ForeLeg', 3: 'ForeMid', 4: 'HindMid',
                   5: 'HindLeg', 6: 'Abdomen', 7: 'Wing'}
start_color = Color("Red")
colors = list(start_color.range_to(Color("Blue"), len(label_name_dict) - 1))
color_map = dict(zip(list(range(1, len(label_name_dict))),
                     map(lambda Color_obj: [int(i * 255) for i in Color.get_rgb(Color_obj)], colors)))

# 根据deeplabcut检测的关键点数据和生成的标签提取信息文件，提取RGB和ST roi

vido_scale = 1  # 视频缩放比率
roi_size: int = 320
time_window: int = 8

if roi_size * vido_scale % 2 != 0:
    print('roi的值需要重新选取，推荐的值为：')
    for roi_i in range(-21, 21):
        for scale_i in range(-5, 5):
            c = (roi_size + roi_i) * (vido_scale + scale_i / 100) / 2
            c = str(c).split('.')
            if c[1] == '0':
                print('roi=%d,scale=%.2f,c=%d' % (roi_size + roi_i, vido_scale + scale_i / 100, int(c[0])))
    sys.exit()

file_folder = r'E:\硕士\柑橘大实蝇梳理行为统计数据\recode\00192.mp4'  # 输入待检测的单个视频或者视频文件夹 F:\昆虫\jxsy\jxsy_recode  E:\硕士\柑橘大实蝇梳理行为统计数据\recode
keypoints_base_folder = r'G:\test'  # 关键点检测信息文件夹 E:\硕士\桔小实蝇数据\桔小实蝇\detect  G:\test
save_path = r'G:'  # 保存行为检测数据的文件夹
start_frame = 0  # 从视频的第几帧开始检测

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.load(r'TrainModel\model\dsy_all16.pth')
model.to(device)
model.eval()

# 将file_folder转化为视频列表，如果file_folder是文件夹就遍历里面所有的视频，包含了视频的绝对路径
video_file_list: list = []
if os.path.isdir(file_folder):
    video_file = os.listdir(file_folder)
    [video_file_list.append(os.path.join(file_folder, file)) for file in video_file]
elif os.path.isfile(file_folder):
    video_file_list.append(file_folder)

for video in video_file_list:
    video_name = str(Path(video).stem)
    print(video_name)
    generator = GenerateStImageFrameByFrame(video, keypoints_base_folder, scale=vido_scale,
                                            roi_size=roi_size, time_window=time_window, start_frame=start_frame)
    cu_idx = 0
    count_time = 0
    frame_rate_count = 0
    predict_label_list = []

    # 是否保存检测视频，不保存就注释掉
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_name + '_detect.avi', fourcc, 25, (generator.width_crop, generator.height_crop))

    while cu_idx < generator.end_frame - 1:

        start = time.time()

        generator.get_roi_rec_frame_by_frame()
        st_rec, cu_idx = generator.generate_st_image()
        # print(cu_idx)

        # 用来保存当前的被裁剪过的帧
        # frame = generator.get_current_frame()
        # cv2.imwrite('1.png', frame)
        # time.sleep(10)

        # 是否显示ST image 和 ROI
        # st_rec_numpy = show_st_image(st_rec, key_points_dict, is_show=True)
        # generator.show_roi()
        # save_img_rec(st_rec_numpy, r'G:\add', cu_idx)

        # 需要将(num_parts, roi_size, roi_size, 3) 转换为 (num_parts, 3, roi_size, roi_size) 以适配模型检测输入
        st_rec_swap = transforms.Resize((224, 224))(torch.permute(st_rec, (0, 3, 1, 2)))

        # 获取roi集合的所有预测置信率与预测label
        confidence, pred_label = predict_image(model, st_rec_swap)
        # print(confidence, pred_label)

        # 经处理后选择显示最终结果（具体规则见论文）
        display_part, display_label, display_conf = get_display_label(confidence, pred_label)
        predict_label_list.append(display_label)
        # print(display_part, display_label, display_conf)

        # 获取当前帧、检测的x，y坐标组
        cur_frame = generator.get_current_frame()
        cur_x, cur_y = generator.get_current_key_points()
        # print(cur_x, cur_y)

        # 在frame上绘制检测结果并显示
        # draw_detect_result_on_frame(video_name, cur_frame, display_part, cur_x, cur_y, color_map,
        #                             display_label, display_conf, cu_idx, label_name_dict[display_label],
        #                             roi_size, vido_scale)
        cv2.imshow('frame', cur_frame)
        # cv2.imwrite('img/' + str(cu_idx) + '.png', cur_frame)
        cv2.waitKey(1)

        # 保存检测视频
        out.write(cur_frame)

        end = time.time()
        count_time += end - start
        frame_rate_count += 1 / (end - start)
        frame_rate = frame_rate_count / (cu_idx + 1)
        # print(frame_rate)
        # print(1 / (end - start))

    print(count_time)
    print(generator.end_frame / count_time)
    print()

    # start, end, motion = judge_motion_section(predict_label_list)
    # generate_csv(start, end, motion, save_path, video_name)

    cv2.destroyAllWindows()
    del generator
