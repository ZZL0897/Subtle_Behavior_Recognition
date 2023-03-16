import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib
from data_processer import load_keypoints_info
from colour import Color
from mpl_toolkits.mplot3d import proj3d
import os


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
matplotlib.use('QT4Agg')


#  重写axes3d.Axes3D，使得z轴位置可以更改
class MyAxes3D(axes3d.Axes3D):

    def __init__(self, baseObject, sides_to_draw):
        self.__class__ = type(baseObject.__class__.__name__,
                              (self.__class__, baseObject.__class__),
                              {})
        self.__dict__ = baseObject.__dict__
        self.sides_to_draw = list(sides_to_draw)
        self.mouse_init()

    def set_some_features_visibility(self, visible):
        for t in self.w_zaxis.get_ticklines() + self.w_zaxis.get_ticklabels():
            t.set_visible(visible)
        self.w_zaxis.line.set_visible(visible)
        self.w_zaxis.pane.set_visible(visible)
        self.w_zaxis.label.set_visible(visible)

    def draw(self, renderer):
        # set visibility of some features False
        self.set_some_features_visibility(False)
        # draw the axes
        super(MyAxes3D, self).draw(renderer)
        # set visibility of some features True.
        # This could be adapted to set your features to desired visibility,
        # e.g. storing the previous values and restoring the values
        self.set_some_features_visibility(True)

        zaxis = self.zaxis
        draw_grid_old = zaxis.axes._draw_grid
        # disable draw grid
        zaxis.axes._draw_grid = False

        tmp_planes = zaxis._PLANES

        if 'l' in self.sides_to_draw:
            # draw zaxis on the left side
            zaxis._PLANES = (tmp_planes[2], tmp_planes[3],
                             tmp_planes[0], tmp_planes[1],
                             tmp_planes[4], tmp_planes[5])
            zaxis.draw(renderer)
        if 'r' in self.sides_to_draw:
            # draw zaxis on the right side
            zaxis._PLANES = (tmp_planes[3], tmp_planes[2],
                             tmp_planes[1], tmp_planes[0],
                             tmp_planes[4], tmp_planes[5])
            zaxis.draw(renderer)

        zaxis._PLANES = tmp_planes

        # disable draw grid
        zaxis.axes._draw_grid = draw_grid_old


def plot_3d(x_list: np.ndarray, y_list: np.ndarray, proj=(0.5, 2, 1, 1.5), elev=30, azim=-45):
    """
    本函数xyz并不是与plot_3D的xyz一一对应的，对应关系为：plot_3D: x, y, z 对应输入顺序为 x, z, y，即z轴数据在底层实际是生成在y轴上的
    这样才模拟出了时间轴的效果
    :param x_list: Shape: (n, frame_length)，n代表有多少组点需要画轨迹
    :param y_list: 同x_list
    :param proj: 通过设置4个参数来调整生成3D图像x，y，z轴的比例。顺序为：x，z，y，总大小
    :param elev: 初始视角
    :param azim: 初始视角
    """
    assert x_list.shape == y_list.shape

    fig = plt.figure(figsize=(15, 10), dpi=150)
    ax = plt.axes(projection='3d')
    ax.set_xticks([])
    ax.set_zticks([])
    ax.set_xlabel('x')
    # ax.set_ylabel('Time')
    ax.set_zlabel('y')
    ax = fig.add_axes(MyAxes3D(ax, 'l'))
    ax.grid(None)

    ax.w_xaxis.set_pane_color((0.95, 0.95, 0.95, 0.01))
    ax.w_yaxis.set_pane_color((0.95, 0.95, 0.95, 0.01))
    ax.w_zaxis.set_pane_color((0.95, 0.95, 0.95, 0.01))

    # 设置每个轴的范围，不设置的话，如果输入数据的min与max差距太小，会导致生成的图像在屏幕之外的情况发生
    ax.set_xlim3d([0, np.max(x_list)])
    ax.set_ylim3d([0, x_list.shape[1]])
    ax.set_zlim3d([0, np.max(y_list)])

    # print(np.max(x_list))
    # print(np.max(y_list))
    # print(np.max(x_list)/np.max(y_list))

    # 设置3D视角为正交透视，这个设置十分重要，默认为persp，默认设置的透视角度并不好看
    ax.set_proj_type('ortho')

    # 通过设置下面的4个参数来调整生成3D图像x，y，z轴的比例
    ax.get_proj = lambda: np.dot(axes3d.Axes3D.get_proj(ax), np.diag(proj))
    # 设置3D图像的初始视角
    ax.view_init(elev=elev, azim=azim)

    # 设置多条轨迹线颜色渐变
    purple = Color("LightSkyBlue")
    colors = list(purple.range_to(Color("OrangeRed"), x_list.shape[0]))
    z = range(0, x_list.shape[1])

    # 画图代码
    for x, y, color in zip(x_list, y_list, colors):
        # print(x.shape)
        # print(y.shape)
        # print(color.get_rgb())
        ax.plot3D(x, z, y*0.75, color=color.get_rgb(), linewidth=1.2)

    # 显示图片
    plt.show()

    return fig


if __name__ == '__main__':
    bodyparts, df_x, df_y, df_likelihood, cropping, cropping_parameters = load_keypoints_info(r'E:\硕士\桔小实蝇数据\桔小实蝇\detect', '00440')
    # print(bodyparts)
    print(df_x.shape)
    print(df_y.shape)
    fig = plot_3d(df_x, df_y)
    fig.savefig('11.png', transparent=True)
    # x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # a = x[range(0, 10, 2)]
    # print(range(0, 10, 2))
    # print(a)
