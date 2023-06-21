# Subtle Behavior Recognition System

## 使用前
### 准备
本系统的运行依赖于DeepLabCut的关键点检测结果，选取头部和尾部两个位置的关键点进行跟踪。  
首先需要完成DeepLabCut模型的训练，然后使用模型检测视频，会得到两个检测文件，以.meta结尾的检测文件放进`template\keypoints\metadata`中，另一个文件放进`template\keypoints\analyze`中，视频检测时会读取这两个文件。

### 安装
run `pip install -r requirement.txt`

### 注意事项
所有的`.csv`文件都应确保编码为utf-8，否则无法正常读取

## 数据集制作
### 数据标注
首先标注视频，记录视频中的行为、开始时间、结束时间，保存在`template\video_label.csv`中  
基于视频标注结果，指定需要从视频中提取数据的帧序号，保存在`template\template.csv`中  
您需要在该步骤中完成对训练集和验证集的划分，在下一步中分别提取

例如：

`video_label.csv`  

| video_name | behavior | start_frame | end_frame |
|------------|----------|-------------|-----------|
| 100        | foreleg  | 0           | 50        |
| 100        | head     | 51          | 120       |

从`video_label.csv`中可以随机选取视频帧并在稍后提取，要确保帧序号在时间范围内

`template.csv`

| video_name | behavior | frame |
|------------|----------|-------|
| 100        | foreleg  | 10    |
| 100        | foreleg  | 25    |
| 100        | foreleg  | 40    |
| 100        | head     | 75    |
| 100        | head     | 100   |

### 训练数据提取
将所有视频放在一个文件夹中

un `extract_training_data.py`

alter `file_folder` = The folder path for storing videos

alter `keypoints_base_folder` = `template\keypoints`

函数`save_img_data`设置`path`参数为`template\tarin_dataset` or `template\val_dataset`

在函数`save_img_data`中可以修改标签字典以适配新的行为识别场景

时空特征图像会分类保存在`template\train_dataset`和`template\val_dataset`中

### 训练数据校对

在训练之前，应该对数据集中的时空特征图像进行检查，剔除明显有错误的样本

## 训练行为识别模型
### 准备工作
run `TrainModel\MyDataset.py`

alter `main()` function

指定训练集和验证集的路径  
运行结束后，会在数据集根目录下生成一个`info.json`文件

### 训练
run `TrainModel\train.py`

需要指定训练集和验证集`info.json`文件的路径，自行设置模型名称与保存路径

## 检测视频
run `detect_video.py`  
代码 41-47行 指定各类参数的路径，检测完毕后会生成行为统计结果

run `tools\check.py`  
指定视频和对应的统计结果文件，可以对统计结果进行校对并更正可能的错误


