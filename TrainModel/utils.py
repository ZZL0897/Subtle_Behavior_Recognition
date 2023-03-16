import torch
import os
import random
import numpy as np
from tqdm import tqdm
from time import sleep
import cv2


def train(model, device, train_loader, criterion, optimizer, epoch, board_writer=None):
    model.train()
    sum_loss = 0
    correct = 0
    for step, (input_tensor, labels) in tqdm(enumerate(train_loader),
                                             desc='Training', total=len(train_loader), mininterval=1, unit=' step'):

        input_tensor, labels = input_tensor.to(device), labels.to(device)
        # cv2.imshow('aaaa', cv2.cvtColor(torch.permute(input_tensor[0], (1, 2, 0)).cpu().numpy(), cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)
        pred = model(input_tensor)
        _, pred_label = torch.max(pred.data, 1)
        correct_step = torch.sum(pred_label == labels)
        correct += correct_step
        loss = criterion(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print_loss = loss.data.item()
        sum_loss += print_loss
        # if (step + 1) % 100 == 0:
        #     print('Train Epoch： {} [{}/{} ({:.0f}%)]\t Loss:{:.6f}\t Acc:{:.4f}'.format(
        #         epoch, (step + 1) * len(input_tensor), len(train_loader.dataset),
        #                100. * (step + 1) / len(train_loader), sum_loss / (step+1), correct/((step+1)*len(input_tensor))
        #     ))
        if board_writer:
            board_writer.add_scalar('train/loss_per_step', print_loss, (epoch-1)*len(train_loader) + (step+1))
            board_writer.add_scalar('train/acc_per_step', correct_step/len(input_tensor), (epoch-1) * len(train_loader) + (step+1))
    correct = correct.data.item()
    acc = correct / len(train_loader.dataset)
    avg_loss = sum_loss / len(train_loader)
    if board_writer:
        board_writer.add_scalar('learning_rate', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
        board_writer.add_scalar('train/loss_per_epoch', avg_loss, epoch)
        board_writer.add_scalar('train/acc_per_epoch', acc, epoch)
    print('Epoch: {}, Train set: Average loss:{}, Accuracy::{}/{} ({:.4f}%)'.format(
        epoch, avg_loss, correct, len(train_loader.dataset), 100 * acc))
    sleep(1)
    return acc


def val(model, device, val_loader, criterion, epoch, board_writer=None):
    model.eval()
    val_loss = 0
    correct = 0
    total_num = len(val_loader.dataset)
    # print('Number of validate dataset:{}'.format(total_num))
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc='Validating', total=len(val_loader), mininterval=1, unit=' step'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            _, pred = torch.max(output.data, 1)
            correct += torch.sum(pred == target)
            print_loss = loss.data.item()
            val_loss += print_loss
        correct = correct.data.item()
        acc = correct / total_num
        avg_loss = val_loss / len(val_loader)
        print('Epoch: {}, Val set: Average loss:{:.4f}, Accuracy:{}/{} ({:.4f}%)'.format(
            epoch, avg_loss, correct, len(val_loader.dataset), 100 * acc
        ))
        if board_writer:
            board_writer.add_scalar('val/loss_per_epoch', avg_loss, epoch)
            board_writer.add_scalar('val/acc_per_epoch', acc, epoch)
        sleep(1)
    return acc


def evaluate(model, device, evaluate_loader, criterion):
    list1 = [0, 0, 0, 0, 0, 0, 0, 0]
    list2 = [0, 0, 0, 0, 0, 0, 0, 0]
    model.eval()
    with torch.no_grad():
        first_iter = True
        total_num = len(evaluate_loader.dataset)
        for data, true_label in tqdm(evaluate_loader, desc='Evaluating',
                                     total=len(evaluate_loader), mininterval=1, unit=' step'):
            data, true_label = data.to(device), true_label.to(device)
            output = model(data)
            loss = criterion(output, true_label)
            _, pred_label = torch.max(output.data, 1)
            if first_iter:
                true_list = true_label
                pred_list = pred_label
                first_iter = False
            else:
                true_list = torch.cat((true_list, true_label))
                pred_list = torch.cat((pred_list, pred_label))
        correct = torch.sum(pred_list == true_list)
        correct = correct.data.item()
        acc = correct / total_num

        for t, p in zip(true_list, pred_list):
            t_i, p_i = t.item(), p.item()
            list1[t_i] += 1
            if t_i == p_i:
                list2[t_i] += 1

        for i in range(0, 8):
            print(str(i) + ': ' + str(list2[i] / list1[i]))

        print(correct)
        print(acc)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


@torch.no_grad()
def predict_image(model, image_tensor):
    """
    载入模型，预测一组batch的图片
    :param model: model load by pytorch
    :param image_tensor: Size: (num_images, num_channels, height, width)
    :return: 预测的置信率与标签值, Size: (num_images)
    """
    output = model(image_tensor)
    pro_softmax = torch.nn.functional.softmax(output, 1)
    confidence, pred_label = torch.max(pro_softmax.data, 1)
    # print(confidence, pred_label)
    return confidence, pred_label
