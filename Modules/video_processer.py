import sys
import cv2
import os

import numpy
import torch
import numpy as np
from colour import Color
from torch import Tensor
from scipy.signal import savgol_filter
from torchvision.transforms import functional as f
from data_processer import load_keypoints_info

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

green = Color("Lime")
colors = list(green.range_to(Color("red"), 256))


def get_r_g(gray):
    return Color.get_red(colors[int(gray)]), Color.get_green(colors[int(gray)])


map_list = tuple(map(get_r_g, range(0, 256)))


def return_r_g(gray):
    if gray < 24:
        return 0, 0
    return map_list[gray][0], map_list[gray][1]


def frame_crop(frame, cropping, cropping_parameters):
    if cropping:
        [x1, x2, y1, y2] = cropping_parameters
        frame_cropped = frame[y1: y2, x1: x2]
    else:
        frame_cropped = frame

    return frame_cropped


def smooth_2d(M, winSm):
    Msm = savgol_filter(M, window_length=winSm, polyorder=0)
    return Msm


def subtract_average(frameVectRec, dim, device):
    shFrameVectRec = frameVectRec.shape  # 获取shape
    averageSubtFrameVecRec = torch.zeros((shFrameVectRec[0], shFrameVectRec[1])).to(device)  # 生成一个同shape的全0矩阵
    assert dim == 0 or dim == 1
    if dim == 0:
        averageVect = torch.mean(frameVectRec, 0)  # 计算cfr每一列的均值

    if dim == 1:
        averageVect = torch.mean(frameVectRec, 1)

    if dim == 0:
        for i in range(0, shFrameVectRec[0]):
            averageSubtFrameVecRec[i, :] = frameVectRec[i, :] - averageVect  # 将cfr的每一行都减去均值

    if dim == 1:
        for i in range(0, shFrameVectRec[1]):
            averageSubtFrameVecRec[:, i] = frameVectRec[:, i] - averageVect

    return averageSubtFrameVecRec  # 这里返回的矩阵维数与传入矩阵相同，每一行都减去了平均值


def cal_mid_point(x, y, i1=0, i2=1):
    """
    计算两个坐标的中点坐标及它们的距离
    :param x: (num_parts, )
    :param y: (num_parts, )
    :param i1, i2: 根据i1与i2取得x, y中对应位置的坐标进行计算
    :return:
    """
    x1 = x[i1]
    y1 = y[i1]
    x2 = x[i2]
    y2 = y[i2]
    mid_point = [(x1 + x2) / 2, (y1 + y2) / 2]
    # distance = math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))
    return mid_point


def cal_rotation(x, y, i1=0, i2=1):
    """
    计算给定两点形成的直线与x轴的夹角
    :return: 旋转角度，该参数对应torchvision.rotation
    """
    x1 = x[i1]
    y1 = y[i1]
    x2 = x[i2]
    y2 = y[i2]
    if np.absolute(x2 - x1) < 20 or np.absolute(y2 - y1) < 20:
        return 0
    k1 = (y2 - y1) / float((x2 - x1))
    kx = 0
    vx = np.array([1, k1])
    vy = np.array([1, kx])
    Lx = np.sqrt(vx.dot(vx))
    Ly = np.sqrt(vy.dot(vy))
    angle = int((np.arccos(vx.dot(vy) / (float(Lx * Ly))) * 180 / np.pi) + 0.5)
    if angle > 45:
        angle = 90 - angle
        if k1 < 0:
            angle = -angle
    else:
        angle = -angle
    return angle


class GenerateStImageByFrameIndex:
    def __init__(self, video_path, keypoints_base_folder, scale=1, roi_size=200, time_window=8):
        self.scale = scale
        self.roi_size = int(roi_size * scale)
        self.time_window = time_window
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load Video and Keypoints Information
        self.cap = cv2.VideoCapture(video_path)
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) * self.scale)
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * self.scale)
        self.bodyparts, self.df_x, self.df_y, self.df_likelihood, self.cropping, cropping_parameters = \
            load_keypoints_info(keypoints_base_folder,
                                video_path)
        self.num_parts = len(self.bodyparts)
        self.current_frame_index: int = 0
        self.ret = False
        self.cur_frame = None

        # Setting Video Scale
        self.df_x = self.df_x * self.scale
        self.df_y = self.df_y * self.scale
        self.cropping_parameters_s = []
        for p in cropping_parameters:
            self.cropping_parameters_s.append(int(p * self.scale))
        [self.x1, self.x2, self.y1, self.y2] = self.cropping_parameters_s
        self.width_crop = int(self.x2 - self.x1)
        self.height_crop = int(self.y2 - self.y1)

        self.maxframe = int(min(self.cap.get(cv2.CAP_PROP_FRAME_COUNT), self.df_x.shape[1]))

        # self.heat_map = np.frompyfunc(return_r_g, 1, 2)

        # Pre_allocate Data Tensor Array
        # Size: (time_window, H, W)
        self.fr_rec = torch.zeros((self.time_window, self.height_crop, self.width_crop)).to(self.device)
        # Size: (num_parts, roi, roi)
        self.roi_arr = torch.zeros((self.num_parts, self.roi_size, self.roi_size)).to(self.device)
        # Size: (time_window, num_parts, roi, roi)
        self.roi_arr_rec = torch.zeros((self.time_window, self.num_parts, self.roi_size, self.roi_size)).to(self.device)
        # print(self.fr_rec)
        self.st_image_rec = torch.zeros((self.num_parts, self.roi_size, self.roi_size, 3)).to(self.device)

        self.fr_rec.requires_grad = False
        self.roi_arr.requires_grad = False
        self.roi_arr_rec.requires_grad = False
        self.st_image_rec.requires_grad = False

    def set_frame_index_to_read(self, frame_index):
        self.cap.set(1, frame_index)
        self.current_frame_index = frame_index

    def read_frame(self):
        self.ret, full_frame = self.cap.read()
        if self.scale != 1:
            full_frame = cv2.resize(full_frame, (self.video_width, self.video_height))
        else:
            full_frame = full_frame
        self.cur_frame = frame_crop(full_frame, self.cropping, self.cropping_parameters_s)
        self.current_frame_index = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)

    def get_current_frame(self, is_show=False) -> numpy.ndarray:
        frame = self.cur_frame
        if is_show:
            cv2.imshow('frame', frame)
            cv2.waitKey(1)
        return frame

    def show_roi(self, is_show=True):
        roi1 = self.roi_arr_rec[-1, 0, :].cpu().numpy()
        roi2 = self.roi_arr_rec[-1, 1, :].cpu().numpy()
        if is_show:
            cv2.imshow('roi1', roi1)
            cv2.imshow('roi2', roi2)
            cv2.waitKey(1)
        return roi1, roi2

    def get_current_key_points(self):
        return self.df_x[:, self.current_frame_index], self.df_y[:, self.current_frame_index]

    def get_frame_and_preprocess(self) -> Tensor:
        """
        读一帧视频帧，转为Gray，根据对象属性裁剪frame，标准化图像数组，并将该帧转换为Tensor对象
        :return: Type: Tensor, Size: (1, H, W)
        """
        self.read_frame()
        # cv2.imshow('frame', cv2.resize(frame, (960, 540)))
        # cv2.waitKey(0)
        frame = self.cur_frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_tensor = torch.div(torch.from_numpy(frame).to(self.device), 255)
        return frame_tensor

    def select_keypoints_ROI(self,
                             frame_index: int,
                             frame_cropped: Tensor):
        """
        传入单帧，通过frame_index确定裁剪中心坐标，并裁剪出一组指定大小的roi，roi的块数由关键点数量决定
        :param frame_index: 通过frame_index确定裁剪中心坐标
        :param frame_cropped: 需要裁剪的帧或帧集合（已经过一次裁剪）, shape: (..., H, W)
        :return: roi_array: 返回裁剪后每个bodyparts的ROI，size : (number of parts, roi, roi)
        """
        # init param by class local variable
        roi_size = self.roi_size

        # load crop center by class local variable
        x = self.df_x[:, frame_index]
        y = self.df_y[:, frame_index]

        # old
        # bottomOvershot = 0
        # rightOvershot = 0

        for i in range(0, self.num_parts):
            # print(x[i])
            # print(y[i])
            topEdge = int(y[i]) - int(roi_size * 0.5)
            if topEdge < 0:
                topEdge = 0

            # old
            # bottomEdge = int(y[i]) + int(roi_size * 0.5)
            # if bottomEdge > int(self.height_crop):
            #     bottomOvershot = bottomEdge - int(self.height_crop)
            #     bottomEdge = int(self.height_crop)

            leftEdge = int(x[i]) - int(roi_size * 0.5)
            if leftEdge < 0:
                leftEdge = 0

            # old
            # rightEdge = int(x[i]) + int(roi_size * 0.5)
            # if rightEdge > int(self.width_crop):
            #     rightOvershot = rightEdge - int(self.width_crop)
            #     rightEdge = int(self.width_crop)

            # new by torchvision.transforms.functional
            cfr = f.crop(frame_cropped, topEdge, leftEdge, roi_size, roi_size)

            # old method
            # cfr = frame_cropped[topEdge:bottomEdge, leftEdge:rightEdge]
            # shapeCfr = cfr.shape
            #
            # # Correct (adding zeros) to make a square shape in case it is not roixroi due to negative values in above section substractions
            # if topEdge == 0:
            #     rw = torch.zeros((np.absolute(shapeCfr[0] - roi_size), shapeCfr[1]), requires_grad=False).to(
            #         self.device)
            #     cfr = torch.vstack((rw, cfr))
            #     shapeCfr = cfr.shape
            # if bottomOvershot > 0:
            #     rw = torch.zeros((np.absolute(shapeCfr[0] - roi_size), shapeCfr[1]), requires_grad=False).to(
            #         self.device)
            #     cfr = torch.vstack((cfr, rw))
            #     shapeCfr = cfr.shape
            # if leftEdge == 0:
            #     col = torch.zeros((shapeCfr[0], np.absolute(shapeCfr[1] - roi_size)), requires_grad=False).to(
            #         self.device)
            #     cfr = torch.hstack((col, cfr))
            #     shapeCfr = cfr.shape
            # if rightOvershot > 0:
            #     col = torch.zeros((shapeCfr[0], np.absolute(shapeCfr[1] - roi_size)), requires_grad=False).to(
            #         self.device)
            #     cfr = torch.hstack((cfr, col))
            #     shapeCfr = cfr.shape
            # self.roi_arr[i] = cfr

            """
            if对应方法get_roi_rec_by_frame_index，只裁剪一帧，刷新self.roi_arr，
            在get_roi_rec_by_frame_index中才会刷新self.roi_arr_rec
            
            else对应子类方法get_roi_rec_frame_by_frame，裁剪多帧，直接刷新self.roi_arr_rec
            """
            if len(frame_cropped.shape) == 2:
                self.roi_arr[i, ...] = cfr
            else:
                self.roi_arr_rec[:, i, ...] = cfr

    def center_of_gravity(self, cfrRec):
        # Quoted from ABRS
        # Written by Primoz Ravbar
        # Finally modified by ZL Zhang
        # sh = np.shape(cfrVectRec)
        """
        :param cfrRec: roi时间窗口集合，Size: (TimeWindow, roi**2)
        :return: 行为时间特征的列向量
        """
        sh = cfrRec.shape

        F = torch.absolute(torch.fft.fft(cfrRec, dim=0))  # 对cfr进行快速傅立叶变换之后取绝对值

        av = torch.zeros((1, sh[0])).to(self.device)  # 建一个行向量，长度为windowST的值
        av[0, :] = torch.arange(1, sh[0] + 1).to(self.device)  # 给它赋值，1到ST
        A = av.repeat(sh[1], 1)  # 把上面那个矩阵行数扩展到size的平方，每一行的值都等于上面赋值的内容

        FA = F * torch.transpose(A, 0, 1)  # F和A的转置相乘（对应位置的元素直接相乘），F与A的转置维数相同
        # print(FA.shape)
        sF = torch.sum(F, dim=0)  # 把F每一列的元素加起来
        sFA = torch.sum(FA, dim=0)  # 把FA每一列的元素加起来
        # print(sF.shape)
        cG = sFA / sF

        return cG  # 这里返回的是一个列向量

    # 生成时空特征图像
    def create_st_image(self, roi_arr_rec_single):
        # Quoted from ABRS
        # Written by Primoz Ravbar
        # Modified by Augusto Escalante
        # Finally modified by ZL Zhang

        # Size: (TimeWindow, roi, roi)
        cfrVectRec_single = torch.reshape(roi_arr_rec_single, (8, self.roi_size ** 2))
        imVarBin = torch.zeros((self.roi_size ** 2)).to(self.device)
        channel_green = torch.absolute(torch.sub(roi_arr_rec_single[-1], roi_arr_rec_single[0]))
        channel_blue = roi_arr_rec_single[-1]

        cG = self.center_of_gravity(cfrVectRec_single)  # 通过快速傅立叶变换对图片集进行处理
        # print(torch.max(cG))

        averageSubtFrameVecRec = subtract_average(cfrVectRec_single, 0, self.device)  # 得到一个减去均值后的图片集数据

        totVar = torch.sum(torch.absolute(averageSubtFrameVecRec), dim=0)  # averageSubtFrameVecRec取绝对值后再求每一列的和
        # imVar = torch.reshape(totVar, (self.roi_size, self.roi_size))
        imVarNorm = totVar / torch.max(torch.max(totVar))
        imVarBin[imVarNorm > 0.15] = 1

        I = (cG - 1) * imVarBin
        I = torch.nan_to_num(I)

        max_in_I = torch.max(I)

        if max_in_I > 0:
            sMNorm = I / max_in_I
        else:
            sMNorm = I

        sMNorm = torch.reshape(sMNorm, (self.roi_size, self.roi_size))
        # imSTsm = smooth_2d(sMNorm, 3)
        # I_RS[I_RS < 0.5] = 0

        # I_RS = cp.reshape(IN, (roi, roi))
        # # cv2.imshow('I_RS', I_RS)
        # I_RS = cp.asnumpy(I_RS)
        # imSTsm = smooth_2d(I_RS, 3)

        """
        由于帧边界在裁剪ROI时有溢出的可能
        溢出边界像素值均为0
        下面给rgb第三个通道复制时要实现颜色的反转，所以需要将channel_blue中为0的值反转，否则ST图像中溢出区域会变为纯蓝色
        """
        channel_blue[channel_blue == 0] = 1
        # print(torch.max(channel_blue))

        rgbArray = torch.zeros((self.roi_size, self.roi_size, 3)).to(self.device)
        rgbArray[..., 0] = sMNorm
        rgbArray[..., 1] = channel_green
        rgbArray[..., 2] = 1 - channel_blue

        return rgbArray

        # jet = (sMNorm * 255).short().cpu().numpy()
        # jet = jet.astype('uint8')
        # # cv2.imshow('jet_origin', jet)
        # jet1, jet2 = self.heat_map(jet)
        # # jet = jet.astype('uint8')
        # # cv2.imshow('JET1', jet1.astype('float32'))
        # # cv2.imshow('JET2', jet2.astype('float32'))
        # # cv2.waitKey(0)
        #
        # rgbArray = np.zeros((self.roi_size, self.roi_size, 3), dtype='float32')
        # rgbArray[..., 0] = jet1.astype('float32')
        # rgbArray[..., 1] = jet2.astype('float32')
        # rgbArray[..., 2] = (1 - channel_blue).cpu().numpy()
        #
        # return torch.from_numpy(rgbArray)

    def get_roi_rec_by_frame_index(self, frame_index: int):
        """
        根据给定的frame_index提取从 (frame_index-TimeWindow+1) 到 frame_index 帧，长度为 TimeWindow 的一组ROI_Rec
        整个时间窗口的裁剪参数均为 frame_index
        :param frame_index: 给定的帧索引，范围在 0 到 self.maxframe
        :result: 将成员变量 self.roi_arr_rec 的值完全刷新
        """
        if frame_index >= self.maxframe or frame_index < self.time_window - 1 or frame_index > self.df_x.shape[1]:
            print('输入的frame_index值非法，帧窗口索引越界')
            sys.exit(0)
        else:
            start_frame = frame_index + 1 - self.time_window
            self.set_frame_index_to_read(start_frame)
            for t in range(0, self.time_window):
                frame_tensor = self.get_frame_and_preprocess()
                # print(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

                """
                refresh the self.roi_arr: (num_parts, roi, roi)
                由于该类用于给定帧索引来生成ST图像，所以每给定一个新的index，需要依次读取time_window帧
                这里传入的frame_tensor是单帧，所以需要刷新self.roi_arr，再迭代赋值给self.roi_arr_rec
                """
                self.select_keypoints_ROI(frame_index, frame_tensor)

                self.roi_arr_rec[t] = self.roi_arr

                # cv2.imshow('roi', self.roi_arr[0].cpu().numpy())
                # cv2.imshow('roi2', self.roi_arr[1].cpu().numpy())
                # cv2.waitKey(0)
            # for j in range(0, 8):
            #     cv2.imshow('roi', self.roi_arr_rec[j, 0].cpu().numpy())
            #     cv2.waitKey(0)
            # if t == self.time_window - 1:
            #     print(self.roi_arr_rec.size())
            #     cv2.imshow('0', self.roi_arr_rec[0, 0].cpu().numpy())
            #     cv2.imshow('1', self.roi_arr_rec[0, 1].cpu().numpy())
            #     cv2.waitKey(1)
            # self.fr_rec = torch.cat((self.fr_rec[1:], frame_tensor))

    def generate_st_image(self):
        """
        生成时空特征图像的前提是获取到一组时间窗口长度的ROI_Rec: (TimeWindow, nparts, roi, roi)
        即需要先运行get_roi_rec_by_frame_index或子类中的get_roi_rec_frame_by_frame，以获取到一组ROI_Rec
        :return: self.st_image_rec: Tensor 含有num_parts数量的st图像集合，第一维为num_parts
                 current_frame_index: int 当前处理的视频帧索引
        """
        for n in range(0, self.num_parts):
            self.st_image_rec[n] = self.create_st_image(self.roi_arr_rec[:, n])
        return self.st_image_rec, self.current_frame_index


class GenerateStImageFrameByFrame(GenerateStImageByFrameIndex):
    def __init__(self, video_path, keypoints_base_folder, start_frame=0, end_frame=None, scale=1, roi_size=340,
                 time_window=8):
        super(GenerateStImageFrameByFrame, self).__init__(video_path, keypoints_base_folder, scale, roi_size,
                                                          time_window)
        if end_frame is None:
            self.end_frame = self.maxframe
        else:
            self.end_frame = end_frame
        if self.end_frame <= start_frame or self.end_frame > self.maxframe or self.end_frame - start_frame < self.time_window - 1:
            print('输入的 start_frame 或 end_frame值非法')
            sys.exit(0)
        self.start_frame = start_frame
        self.cap.set(1, self.start_frame)

    def get_roi_rec_frame_by_frame(self):
        """
        该方法用于顺序读取帧并生成ST图像
        所以构建了self.fr_rec变量存放整个帧窗口
        每次读取一帧只需要删除self.fr_rec第一帧，再在其后面拼接新读取的帧即可，不需要重复读取time_window长度的帧
        """
        frame_tensor = self.get_frame_and_preprocess()
        self.fr_rec = torch.cat((self.fr_rec[1:], torch.unsqueeze(frame_tensor, 0)))

        # refresh the self.roi_arr_rec: (time_window, num_parts, roi, roi)
        self.select_keypoints_ROI(int(self.cap.get(cv2.CAP_PROP_POS_FRAMES) - 1), self.fr_rec)


def draw_detect_result_on_frame(video_name: str,
                                frame: numpy.ndarray,
                                part: int,
                                x_rec: list,
                                y_rec: list,
                                color_map: dict,
                                display_label: int,
                                display_conf: float,
                                current_frame_index: int,
                                label_name: str,
                                roi_size: int,
                                video_scale: float) -> None:
    # cv2.putText(frame,
    #             str('Video Name: ' + video_name),
    #             (25, 60),
    #             cv2.FONT_HERSHEY_COMPLEX_SMALL, 2.5, (0, 255, 255),
    #             2)
    cv2.putText(frame,
                str('Current Frame: ' + str(current_frame_index)),
                (25, 80),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 3.5, (0, 255, 255),
                3)
    if part >= 0:
        roi_size = roi_size * video_scale
        cv2.rectangle(frame,
                      (int(x_rec[part] - roi_size / 2), int(y_rec[part] - roi_size / 2)),
                      (int(x_rec[part] + roi_size / 2), int(y_rec[part] + roi_size / 2)),
                      color_map[display_label],
                      int(5*video_scale))
        px = int(x_rec[part] - roi_size/2 - 120)
        py = int(y_rec[part] + roi_size/2 + 75)
        px = px if px > 0 else 0
        py = py if py < frame.shape[1] else frame.shape[1]
        cv2.putText(frame,
                    str(label_name + ' ' + str(int(display_conf * 100)) + '%'),
                    (px, py),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 3.5, color_map[display_label],
                    3)


if __name__ == '__main__':
    g = GenerateStImageFrameByFrame(r'G:\new2025\1.mp4', r'G:\new2025', roi_size=500)
    i = 1
    while 1:
        g.get_roi_rec_frame_by_frame()
        st_rec, cu_idx = g.generate_st_image()
        cv2.imshow('f', cv2.cvtColor(st_rec[0].cpu().numpy(), cv2.COLOR_RGB2BGR))
        # cv2.imshow('p', cv2.cvtColor(st_rec[1].cpu().numpy(), cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        # print(i)
        i += 1
