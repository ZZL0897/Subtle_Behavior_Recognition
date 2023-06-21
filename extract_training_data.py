import os
import pandas as pd
from video_processer import GenerateStImageByFrameIndex
from data_processer import save_img_data
from pathlib import Path

"""
使用template.csv以输入提取的label信息
保存后用记事本打开，另存为utf-8编码
"""

# 存视频的文件夹路径
file_folder = r''
# 存关键点检测文件的文件夹路径
keypoints_base_folder = r'template\keypoints'

video_file_list = os.listdir(file_folder)

# 要提帧的表，表的格式为：【视频，行为，帧】
file = pd.read_csv(r'')
video_id_list = file['视频']
# video_id_list = int(video_id_list)
motion_list = file['行为']
frame_id_list = file['帧']

vid = str(0)
for i, video in enumerate(video_id_list, 0):
    if len(str(int(video))) == 2:
        video_name = '000' + str(int(video))
    elif len(str(int(video))) == 3:
        video_name = '00' + str(int(video))

    if int(vid) != int(video_name):
        vid = video_name
        print(vid)
        for video_file in video_file_list:
            if Path(video_file).stem == video_name:
                video_path = os.path.join(file_folder, video_file)
                generator = GenerateStImageByFrameIndex(video_path, keypoints_base_folder, roi_size=200)

    frameInd = frame_id_list[i]
    generator.get_roi_rec_by_frame_index(frameInd)
    st_rec, cu_idx = generator.generate_st_image()
    # cv2.imshow('f', cv2.cvtColor(st_rec[0].cpu().numpy(), cv2.COLOR_RGB2BGR))
    # cv2.imshow('p', cv2.cvtColor(st_rec[1].cpu().numpy(), cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)

    save_img_data(path=r'template\dataset', file_name=str(vid) + '_' + str(frameInd), ST_arr=st_rec.cpu().numpy(),
                  save_size=generator.roi_size, video_id=vid, label_name=str(motion_list[i]))
    print(i)
