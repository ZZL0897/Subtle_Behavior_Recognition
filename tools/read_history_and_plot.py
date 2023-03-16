import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

# 弹窗显示图片
matplotlib.use('QT4Agg')


def read_tensorboard_data(tensorboard_path):
    """读取tensorboard数据，
    tensorboard_path是tensorboard数据地址"""
    ea = event_accumulator.EventAccumulator(tensorboard_path)
    ea.Reload()
    print(ea.scalars.Keys())
    return ea.scalars


def get_acc_and_loss(tensorboard_data,
                     train_names=('train/acc_per_epoch', 'train/loss_per_epoch'),
                     val_names=('val/acc_per_epoch', 'val/loss_per_epoch')):
    """
    读取tensorboard_data中各项数据，共4项，需要指定标签名称
    """
    Train_acc = [acc.value for acc in tensorboard_data.Items(train_names[0])]
    Train_loss = [loss.value for loss in tensorboard_data.Items(train_names[1])]
    Val_acc = [acc.value for acc in tensorboard_data.Items(val_names[0])]
    Val_loss = [loss.value for loss in tensorboard_data.Items(val_names[1])]
    return Train_acc, Train_loss, Val_acc, Val_loss


def plot_training_fig(title, mode='all', save=None, Train_acc=None, Train_loss=None, Val_acc=None, Val_loss=None):
    if mode == 'all':
        assert (Train_acc and Train_loss and Val_acc and Val_loss) is not None
        assert len(Train_acc) == len(Train_loss) == len(Val_acc) == len(Val_loss)
    elif mode == 'train':
        assert (Train_acc and Train_loss) is not None
        assert len(Train_acc) == len(Train_loss)
    elif mode == 'val':
        assert (Val_acc and Val_loss) is not None
        assert len(Val_acc) == len(Val_loss)
    else:
        print('Value mode is invalid, optional: all, train, val')
        return -1

    length = len(Train_acc)
    epochs = range(1, length + 1)

    font = {
        'size': 15,
    }

    plt.title(title, fontsize=15)
    plt.ylabel('Loss and Accuracy', fontsize=13)
    plt.xlabel('Epoch', fontsize=13)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 1.0)
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, length + 1, step=5))
    if mode == 'all':
        plt.plot(epochs, Train_acc, 'fuchsia', label='Training acc')
        plt.plot(epochs, Train_loss, 'cyan', label='Training loss')
        plt.plot(epochs, Val_acc, 'lime', label='Val acc', linestyle="--")
        plt.plot(epochs, Val_loss, 'coral', label='Val loss', linestyle="--")
    elif mode == 'train':
        plt.plot(epochs, Train_acc, 'fuchsia', label='Training acc')
        plt.plot(epochs, Train_loss, 'cyan', label='Training loss')
    elif mode == 'val':
        plt.plot(epochs, Val_acc, 'lime', label='Val acc')
        plt.plot(epochs, Val_loss, 'coral', label='Val loss')
    plt.legend(prop=font)
    if save:
        plt.savefig(save + '.png', dpi=300, bbox_inches='tight')
        print('图片已保存至根目录下')
    # plt.grid(linestyle="--", color="gray")
    plt.show()


if __name__ == '__main__':
    training_data = read_tensorboard_data(r'../TrainModel/log/dsy_two_aug_res50v1_sgd_bs16')
    train_acc, train_loss, val_acc, val_loss = get_acc_and_loss(training_data)

    # print(train_acc)
    # print(train_loss)
    # print(val_acc)
    # print(val_loss)

    plot_training_fig('Training Results for Bactrocera minax',
                      Train_acc=train_acc,
                      Train_loss=train_loss,
                      Val_acc=val_acc,
                      Val_loss=val_loss)
