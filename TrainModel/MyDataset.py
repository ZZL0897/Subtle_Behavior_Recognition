import random
from typing import List, Dict, Any
from torch.utils.data import Dataset
import os
import json
import numpy as np
from pathlib import Path
import cv2
from torch.utils.data.dataset import T_co
from torchvision import transforms
from torch.utils.data import DataLoader


def load_path_and_label(image_file_name, image_file_path, label_file_name, label_file_path) -> List[Dict]:
    #  载入训练图片的绝对路径和label，返回一个包含了所有数据的List
    info_list = []
    dictlabel = {'img_path': None,
                 'label_name': None,
                 'label': None}
    for i in range(0, len(image_file_name)):
        if str(image_file_name[i][-2]) == '0':
            img = image_file_name[i][:-1]
        else:
            img = image_file_name[i]
        if img in label_file_name:
            # print(label_file.index(RGB_img))
            with open(label_file_path[label_file_name.index(img)], 'r') as f:
                data = json.load(f)
            #  每个字典内存放了三个key的数据
            dictlabel['img_path'] = image_file_path[i]
            dictlabel['label_name'] = data['label_name']
            dictlabel['label'] = data['label']
            info_list.append(dictlabel.copy())
        else:
            print('Can not find ', img, 'label')

    # y_train = np.array(y_label)
    # state = np.random.get_state()
    # np.random.shuffle(x_path)
    # np.random.set_state(state)
    # np.random.shuffle(y_train)

    return info_list


class LoadInfoIntoOneJson:
    #  调用该类，将所有图片的绝对路径及对应的label全部保存到一个json文件中，该文件保存于数据集的根目录下
    #  后续训练将读取保存的这个json文件以获取所有训练图片的路径及对应的label
    def __init__(self, train_img_folder, train_label_folder, test_img_folder=None, test_label_folder=None):
        self.train_img_name, self.train_img_path = self.show_files(train_img_folder, [], [])
        self.train_label_name, self.train_label_path = self.show_files(train_label_folder, [], [])
        train_info = load_path_and_label(self.train_img_name,
                                         self.train_img_path,
                                         self.train_label_name,
                                         self.train_label_path)
        os.chdir(train_label_folder)
        OutputPath = os.path.abspath('..')
        with open(os.path.join(OutputPath, 'train_info') + '.json', 'w+') as f:
            json.dump(train_info, f)
        if test_img_folder and test_label_folder:
            self.test_img_name, self.test_img_path = self.show_files(test_img_folder, [], [])
            self.test_label_name, self.test_label_path = self.show_files(test_label_folder, [], [])
            test_info = load_path_and_label(self.test_img_name,
                                            self.test_img_path,
                                            self.test_label_name,
                                            self.test_label_path)
            os.chdir(test_label_folder)
            OutputPath = os.path.abspath('..')
            with open(os.path.join(OutputPath, 'test_info') + '.json', 'w+') as f:
                json.dump(test_info, f)
        else:
            print('没有输入测试集或验证集的路径信息')

    def show_files(self, path, all_files, all_files_path):
        # 首先遍历当前目录所有文件及文件夹
        file_list = os.listdir(path)
        # 准备循环判断每个元素是否是文件夹还是文件，是文件的话，把名称传入list，是文件夹的话，递归
        for file in file_list:
            # 利用os.path.join()方法取得路径全名，并存入cur_path变量，否则每次只能遍历一层目录
            cur_path = os.path.join(path, file)
            # 判断是否是文件夹
            if os.path.isdir(cur_path):
                self.show_files(cur_path, all_files, all_files_path)
            else:
                all_files.append(str(Path(file).stem))
                all_files_path.append(cur_path)

        return all_files, all_files_path


def load_image(image_path, image_size):
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)  # BGR，可读取中文路径
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size)
    return image


class MyDataset(Dataset):
    def __init__(self, dataset_info_path, transform=None, img_size=(224, 224), drop_label0=False):
        super(MyDataset, self).__init__()
        dataset_info = []
        dataset_info_0 = []
        with open(dataset_info_path, "r") as f:
            for idx, line in enumerate(f):
                ...
            d = json.loads(line)
            for i, info_dict in enumerate(d):
                img_path = info_dict['img_path']
                label = info_dict['label']
                if int(label) != 0:
                    dataset_info.append((img_path, int(label)))
                else:
                    dataset_info_0.append((img_path, int(label)))

        random.seed(666)
        random.shuffle(dataset_info_0)

        if drop_label0:
            dataset_info_0 = dataset_info_0[:int(len(dataset_info_0) / 2)]

        dataset_info.extend(dataset_info_0)

        random.shuffle(dataset_info)
        self.dataset_info = dataset_info
        self.image_size = img_size
        if transform is None:
            self.transforms = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Resize(self.image_size)])
        else:
            self.transforms = transform

    def __getitem__(self, index: Any) -> T_co:
        img_path, label = self.dataset_info[index]
        img = load_image(img_path, self.image_size)
        img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.dataset_info)


if __name__ == '__main__':
    LoadInfoIntoOneJson(train_img_folder=r'G:\all_jxsy\ST',
                        train_label_folder=r'G:\all_jxsy\label',
                        test_img_folder=r'',
                        test_label_folder=r'')

    # train_data = MyDataset(r'G:\all_jxsy\train_info.json', transform=transforms.ToTensor())
    # test_data = MyDataset(r'G:\val_dsy_two\test_info.json', transform=transforms.ToTensor(), drop_label0=True)
    # train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True, num_workers=4, drop_last=True)
    # test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, num_workers=4, drop_last=True)
    # print(len(train_loader))
    # for i in test_loader:
    #     print(i)
