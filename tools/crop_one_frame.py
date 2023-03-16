import os

import cv2
import pandas as pd
from video_processer import GenerateStImageByFrameIndex
from data_processer import save_img_data
from pathlib import Path

if __name__ == '__main__':
    g = GenerateStImageByFrameIndex(r'H:\recode\00167.mp4', r'G:\test')
    maxframe = g.maxframe

    frame_index = maxframe
    g.set_frame_index_to_read(frame_index-1)
    frame = g.get_current_frame()

    save_path = r'D:\Users\ZZL\Desktop\关键点轨迹绘制\252'

    cv2.imencode('.png', frame)[1].tofile(os.path.join(save_path, str(frame_index)+'.png'))

