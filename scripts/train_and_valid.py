import os.path
import time

import pandas as pd
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from torch import nn
from tqdm import tqdm

from scripts.dataloader import build_gabor_image_dataloader
from scripts.models import perceptual_net
from scripts.utils import concatenate_transform_steps, noise_func, determine_training_stops

# 设置种子，保证随机数一致
torch.manual_seed(20010509)
np.random.seed(20010509)

# 打印运行机器类型：cpu、gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'workiing on {device}')


def fit_one_cycle(network,
                  dataloader,
                  optimizer,
                  loss_func,
                  device=torch.device("cpu"),
                  idx_epoch=0,
                  train=True,
                  verbose=0,
                  n_noise=2,
                  low_noise=1e-3,
                  high_noise=2e-1,
                  noise_probability=0.5,
                  image_channel=1,
                  sequence_length=100, ):
    """
    一次训练或者验证过程

    :param noise_probability: 每张图片是否加噪音的概率
    :param low_noise: 图片噪音程度下限
    :param high_noise: 图片噪音程度上限
    :param network: 模型网络
    :param dataloader: 数据集合
    :param optimizer: 优化器
    :param loss_func: 损失函数
    :param device: cpu、gpu
    :param idx_epoch: 当前第几轮
    :param train: 是否是训练过程
    :param verbose: 输出信息的详细程度：大于0更详细
    :param n_noise: 是否插入纯噪音trail
    :return:
    """

    # 设置模型模式
    if train:
        network.train(True).to(device)
    else:
        network.eval().to(device)

    loss = 0
    # rule_true_list, action_true_list, angle_true_list = [], [], []
    # rule_pre_list, action_pre_list, angle_pre_list = [], [], []
    # 初始化epoch级别的计数器
    correct_predictions_rule = 0
    correct_predictions_action = 0
    correct_predictions_angle = 0
    total_predictions_rule = 0
    total_predictions_action = 0
    total_predictions_angle = 0

    # 初始化batch级别的计数器
    batch_correct_predictions_rule = 0
    batch_correct_predictions_action = 0
    batch_correct_predictions_angle = 0
    batch_total_predictions_rule = 0
    batch_total_predictions_action = 0
    batch_total_predictions_angle = 0

    iterator = tqdm(enumerate(dataloader))
    for idx_batch, (first_image_list, second_image_list, angle_true_label, rule_true_label, action_true_label) in iterator:

        new_rule_label = torch.zeros_like(rule_true_label)

        # 遍历每个batch
        for i in range(rule_true_label.shape[0]):
            # 设置每个batch的第一个序列元素为 [0, 1]
            new_rule_label[i, 0, :] = torch.tensor([0, 1])

            # 移动剩余的元素，原始序列的第 n 个元素移动到新序列的第 n-1 个位置
            new_rule_label[i, 1:, :] = rule_true_label[i, :-1, :]

        if train and np.random.randn() < noise_probability:  # 50%的概率给图片加噪音
            # TODO
            pass

        if n_noise > 0:
            # TODO
            pass

        first_image_list = first_image_list.squeeze(2)
        second_image_list = second_image_list.squeeze(2)
        first_image_list = first_image_list.reshape(batch_size, sequence_length, image_resize * image_resize)  # first_image_list: (batch_size, sequence_length, image_resize * image_resize)
        second_image_list = second_image_list.reshape(batch_size, sequence_length, image_resize * image_resize)

        # 处理图片输入
        rnn_input = torch.cat((first_image_list, second_image_list), dim=2)  # combined: (batch_size, sequence_length, 2*(image_size*image_size))

        # 重置优化器
        optimizer.zero_grad()

        # 模型向前传播
        rule_pre, action_pre, angle_pre, out = network(rnn_input.to(device), new_rule_label.to(device))

        # 计算损失
        rule_loss = loss_func(rule_pre.float().to(device),
                              rule_true_label.float().to(device), )
        action_loss = loss_func(action_pre.float().to(device),
                                action_true_label.float().to(device), )
        angle_loss = loss_func(angle_pre.float().to(device),
                               angle_true_label.float().to(device), )
        batch_all_loss = action_loss + rule_loss
        loss += batch_all_loss.item()

        # if n_noise > 0:
        #     # TODO
        #     pass
        # else:
        #     rule_true_list.append(rule_true_label.detach().cpu().numpy())
        #     rule_pre_list.append(rule_pre.detach().cpu().numpy())
        #
        #     action_true_list.append(action_true_label.detach().cpu().numpy())
        #     action_pre_list.append(action_pre.detach().cpu().numpy())
        #
        #     angle_true_list.append(angle_true_label.detach().cpu().numpy())
        #     angle_pre_list.append(angle_pre.detach().cpu().numpy())

        if train:
            # 反向传播
            batch_all_loss.backward()
            # 修改权重
            optimizer.step()

        if verbose > 0:
            try:
                # 确保真实标签也在正确的设备上
                rule_true_label = rule_true_label.to(device)
                action_true_label = action_true_label.to(device)
                angle_true_label = angle_true_label.to(device)

                # 计算准确率
                _, predicted_rule = torch.max(rule_pre, 2)  # [batch_size, sequence_length]
                _, predicted_action = torch.max(action_pre, 2)  # [batch_size, sequence_length]
                _, predicted_angle = torch.max(angle_pre, 2)  # [batch_size, sequence_length]

                # 比较预测和真实标签
                batch_correct_predictions_rule += (predicted_rule == rule_true_label.argmax(dim=2)).sum().item()
                batch_correct_predictions_action += (predicted_action == action_true_label.argmax(dim=2)).sum().item()
                batch_correct_predictions_angle += (predicted_angle == angle_true_label.argmax(dim=2)).sum().item()

                batch_total_predictions_rule += rule_pre.numel() // rule_pre.size(2)
                batch_total_predictions_action += action_pre.numel() // action_pre.size(2)
                batch_total_predictions_angle += angle_pre.numel() // angle_pre.size(2)

                # 更新epoch级别的计数器
                correct_predictions_rule += batch_correct_predictions_rule
                correct_predictions_action += batch_correct_predictions_action
                correct_predictions_angle += batch_correct_predictions_angle
                total_predictions_rule += batch_total_predictions_rule
                total_predictions_action += batch_total_predictions_action
                total_predictions_angle += batch_total_predictions_angle

                # 计算并打印每个批次的准确率
                batch_accuracy_rule = batch_correct_predictions_rule / batch_total_predictions_rule
                batch_accuracy_action = batch_correct_predictions_action / batch_total_predictions_action
                batch_accuracy_angle = batch_correct_predictions_angle / batch_total_predictions_angle
                epoch_accuracy_rule = correct_predictions_rule / total_predictions_rule
                epoch_accuracy_action = correct_predictions_action / total_predictions_action
                epoch_accuracy_angle = correct_predictions_angle / total_predictions_angle
            except Exception as e:
                print(e)

            message = f"""epoch {idx_epoch + 1}-{idx_batch + 1:3.0f}-{len(dataloader):4.0f}/{100 * (idx_batch + 1) / len(dataloader):2.3f}%, loss = {loss / (idx_batch + 1):2.4f}, S(rule) = {epoch_accuracy_rule:.4f}, S(action) = {epoch_accuracy_action:.4f}, S(angle) = {epoch_accuracy_angle:.4f}""".replace(
                '\n', '')
            iterator.set_description(message)

            # 重置batch级别的计数器
            batch_correct_predictions_rule = 0
            batch_correct_predictions_action = 0
            batch_correct_predictions_angle = 0
            batch_total_predictions_rule = 0
            batch_total_predictions_action = 0
            batch_total_predictions_angle = 0
        # if idx_batch == 200:
        #     torch.save(network.state_dict(), saving_name)  # why do i need state_dict()?
        #     print(f'save {saving_name}')

    if verbose > 0:
        epoch_accuracy_rule = correct_predictions_rule / total_predictions_rule
        epoch_accuracy_action = correct_predictions_action / total_predictions_action
        epoch_accuracy_angle = correct_predictions_angle / total_predictions_angle

    return network, loss / (idx_batch + 1)


def train_valid_loop(network,
                     dataloader_train,
                     dataloader_valid,
                     optimizer,
                     loss_func,
                     device=torch.device('cpu'),
                     n_epochs=1000,
                     verbose=0,
                     patience=5,
                     warmup_epochs=5,
                     tol=1e-4,
                     n_noise=2,
                     saving_name="perceptual_network.h5",
                     low_noise=1e-3,
                     high_noise=2e-1,
                     noise_probability=0.5,
                     image_channel=1,
                     sequence_length=100, ):
    """
    总体训练以及验证循环函数

    :param noise_probability: 每张图片是否加噪音的概率
    :param low_noise: 图片噪音程度下限
    :param high_noise: 图片噪音程度上限
    :param network: 需要训练的模型
    :param dataloader_train: 训练集
    :param dataloader_valid: 验证集
    :param optimizer: 优化器
    :param loss_func: 损失函数
    :param device: cpu、gpu
    :param n_epochs: 循环次数
    :param verbose: 输出信息的详细程度：大于0更详细
    :param patience: patience轮次内效果不改善则早停
    :param warmup_epochs: 在训练初期的warmup_epochs轮次内逐步提高学习率
    :param tol: 改善值低于tol值则认定模型未改善
    :param n_noise: 是否插入纯噪音trail
    :param saving_name: 模型参数保存名
    :return:
    """

    best_valid_loss = np.inf  # 记录最佳验证损失
    losses = []
    counts = 0
    for idx_epoch in range(n_epochs):
        print("Training ...")
        network, loss_train = fit_one_cycle(network,
                                            dataloader=dataloader_train,
                                            optimizer=optimizer,
                                            loss_func=loss_func,
                                            device=device,
                                            idx_epoch=idx_epoch,
                                            train=True,
                                            verbose=verbose,
                                            n_noise=n_noise,
                                            low_noise=low_noise,
                                            high_noise=high_noise,
                                            noise_probability=noise_probability,
                                            image_channel=image_channel,
                                            sequence_length=sequence_length, )
        print("Validating ...")
        with torch.no_grad():
            _, loss_valid = fit_one_cycle(network,
                                          dataloader=dataloader_valid,
                                          optimizer=optimizer,
                                          loss_func=loss_func,
                                          device=device,
                                          idx_epoch=idx_epoch,
                                          train=False,
                                          verbose=verbose,
                                          n_noise=0,
                                          low_noise=low_noise,
                                          high_noise=high_noise,
                                          noise_probability=0,
                                          image_channel=image_channel,
                                          sequence_length=sequence_length, )
        losses.append([loss_train, loss_valid])

        best_valid_loss, counts = determine_training_stops(nets=[network, ],
                                                           idx_epoch=idx_epoch,
                                                           warmup_epochs=warmup_epochs,
                                                           valid_loss=loss_valid,
                                                           counts=counts,
                                                           best_valid_loss=best_valid_loss,
                                                           tol=tol,
                                                           saving_names={saving_name: True, },
                                                           )
        if counts > patience:
            break
        else:
            message = f"""epoch {idx_epoch + 1}, best validation loss = {best_valid_loss:.4f}, count = {counts}""".replace(
                '\n', '')
            print(message)
    return network, losses


if __name__ == "__main__":
    # 图形参数
    image_resize = 128
    lamda = 8
    batch_size = 1
    shuffle = False  # 注意设置！！！
    num_workers = 2  # 指定用于数据加载的子进程数
    image_channel = 1
    transform_steps = concatenate_transform_steps(image_resize=image_resize,
                                                  noise_level=0,
                                                  flip=False,
                                                  rotate=0,
                                                  num_output_channels=image_channel, )
    low_noise = 1e-3
    high_noise = 2e-1
    noise_probability = 0
    n_noise = 0
    sequence_length = 100

    # 模型参数
    model_name = "rnn"
    activation_func_name = "relu"
    model_dir = '../models'
    if not os.path.exists(model_dir): os.mkdir(model_dir)
    right_now = time.localtime()

    learning_rate = 1e-4
    l2_decay = 1e-4
    n_epochs = 1000
    patience = 5
    warmup_epochs = 2
    tol = 1e-4
    saving_name = os.path.join(model_dir,
                               f'{right_now.tm_year}_{right_now.tm_mon}_{right_now.tm_mday}_{right_now.tm_hour}_{right_now.tm_min}_{model_name}.h5')
    input_size = 2 * image_resize * image_resize + 1024
    hidden_size = 64
    num_layers = 1
    verbose = 1

    # load dataframes
    data_dir = "../data/type1_stimuli"
    df_train = pd.read_csv(os.path.join(data_dir, "df_train_320.csv"))
    df_valid = pd.read_csv(os.path.join(data_dir, 'df_valid_320.csv'))
    df_test = pd.read_csv(os.path.join(data_dir, 'df_test_5000.csv'))

    # build dataloaders
    arg_dataloader = dict(image_resize=image_resize,
                          transform_steps=transform_steps,
                          lamda=lamda,
                          shuffle=shuffle,
                          num_workers=num_workers, )
    dataloader_train = build_gabor_image_dataloader(df_train,
                                                    batch_size=batch_size,
                                                    sequence_length=sequence_length,
                                                    **arg_dataloader)
    dataloader_valid = build_gabor_image_dataloader(df_valid,
                                                    batch_size=batch_size,
                                                    sequence_length=sequence_length,
                                                    **arg_dataloader)
    dataloader_test = build_gabor_image_dataloader(df_test,
                                                   batch_size=1,
                                                   sequence_length=sequence_length,
                                                   **arg_dataloader)

    # 建立模型
    network = perceptual_net(device=device,
                             input_size=input_size,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             model_name=model_name,
                             activation_func_name=activation_func_name,
                             batch_first=True,
                             image_channel=image_channel, )
    # network.load_state_dict(torch.load('../models/2024_9_27_17_51_rnn.h5'))
    # network.eval()

    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(params=network.parameters(),
                                 lr=learning_rate,
                                 weight_decay=l2_decay, )
    loss_func = nn.BCELoss()

    # 训练函数
    network, losses = train_valid_loop(network=network,
                                       dataloader_train=dataloader_train,
                                       dataloader_valid=dataloader_valid,
                                       optimizer=optimizer,
                                       loss_func=loss_func,
                                       device=device,
                                       n_epochs=n_epochs,
                                       verbose=verbose,
                                       patience=patience,
                                       warmup_epochs=warmup_epochs,
                                       tol=tol,
                                       n_noise=n_noise,
                                       saving_name=saving_name,
                                       low_noise=low_noise,
                                       high_noise=high_noise,
                                       noise_probability=noise_probability,
                                       image_channel=image_channel,
                                       sequence_length=sequence_length, )
    del network
