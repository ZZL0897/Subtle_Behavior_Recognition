import sys
import threading
import cv2
import pandas as pd
import tkinter
from tkinter import filedialog
from data_processer import generate_check
from pathlib import Path


class T(threading.Thread):  # 创建一个线程用来检测键盘的输入
    twice = 2
    input = ''
    input_str = ""

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        # while True:
        input_kb = str(sys.stdin.readline()).strip("\n")
        print()
        if input_kb:  # stop
            self.twice = 1
            self.input = input_kb
            # break
        else:
            self.input_str = input_kb


class PlayWithProgressBar:
    def __init__(self, VideoCap, totalFrame, start_frameIdx, end_frameIdx,
                 trackbarName='Progress', windowName='Video'):
        self.VideoCap = VideoCap
        self.trackbarName = trackbarName
        self.windowName = windowName
        self.n_currentframe = 0
        self.totalFrame = totalFrame
        self.start_frameIdx = start_frameIdx
        self.end_frameIdx = end_frameIdx
        self.controlRate = 0
        cv2.createTrackbar(trackbarName, windowName, 0, 100, self.callback)

    def callback(self, controlRate):
        self.n_currentframe = int((controlRate / 100) * self.totalFrame + self.start_frameIdx)
        self.VideoCap.set(cv2.CAP_PROP_POS_FRAMES, self.n_currentframe)

    def fresh_controlRate(self):
        self.controlRate = int((self.n_currentframe / self.totalFrame) * 100)
        cv2.setTrackbarPos(self.trackbarName, self.windowName, self.controlRate)

    def run(self, Fid) -> None:
        self.n_currentframe = int(Fid - self.start_frameIdx)
        self.fresh_controlRate()


def get_file_path(message):
    print(message, end='')
    root = tkinter.Tk()
    root.wm_withdraw()
    path = filedialog.askopenfilename()
    root.destroy()
    root.mainloop()
    return path


video_path = get_file_path('选择视频文件：')
video_name = str(Path(video_path).stem)
print(video_path)
detection_file_path = get_file_path('选择该视频对应的检测文件：')
print(detection_file_path)

cap = cv2.VideoCapture(video_path)
detection_file = pd.read_csv(detection_file_path)

framerate = 25  # 帧率

start = detection_file['开始帧']  # 把csv里的对应列的数据存放到列表中
end = detection_file['结束帧']
motion = detection_file['行为']
times = detection_file['持续时间']

start_check = []  # 创建空列表存储检查之后的结果
end_check = []
motion_check = []

# while 1:
#     cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
#     cv2.createTrackbar('Progress', 'Video', 1, 255, callback)
#     frame = cv2.getTrackbarPos('Progress', 'Video')
#     print(frame)
#     cv2.waitKey(0)

for i in range(0, len(start)):

    s = start[i]
    e = end[i]

    print('检测的行为是：{}，持续时间：{}秒。当前是第{}/{}个区间。'.format(motion[i], times[i], i + 1, len(start)))
    print('行为是否正确？正确输入1，不正确则输入正确行为：', end='')

    my_t = T()
    my_t.start()

    while True:
        is_exit = False
        # cv2.namedWindow('Video', cv2.WINDOW_FREERATIO)
        P = PlayWithProgressBar(cap, e - s, s, e)
        cap.set(cv2.CAP_PROP_POS_FRAMES, s)
        # for frameID in range(s, e):
        frameID = s
        while frameID < e:
            success, frame = cap.read()
            frameID = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if success:
                P.run(frameID)
                frame = cv2.resize(frame, (960, 560))
                cv2.imshow('Video', frame)
                cv2.waitKey(25)
            if my_t.twice == 1:
                flag = my_t.input
                is_exit = True
                break
        if is_exit:
            break

    if i == 0:
        if flag == '1' or flag == '':
            m = motion[i]
            start_check.append(start[i])
            end_check.append(end[i])
            motion_check.append(motion[i])
        elif flag == 'exit':
            print('手动结束！')
            break
        else:
            m = flag
            start_check.append(start[i])
            end_check.append(end[i])
            motion_check.append(m)

    else:
        if flag == '1' or flag == '':
            m = motion[i]
        elif flag == 'exit':
            print('手动结束！')
            break
        else:
            m = flag

        if motion_check[-1] == m:
            end_check[-1] = end[i]
        elif motion_check[-1] != m:
            start_check.append(start[i])
            end_check.append(end[i])
            motion_check.append(m)

    try:
        check = pd.DataFrame(columns=['s分', 's秒', 's帧', 'e分', 'e秒', 'e帧', '持续时间', '持续帧数', '行为', '开始帧', '结束帧'])
        check = generate_check(check, start_check, end_check, motion_check, framerate)

        check.to_csv(video_name + '_check.csv', index=False)  # 检查结果保存路径与文件名称
    except:
        print('保存失败')

cap.release()
cv2.destroyAllWindows()
