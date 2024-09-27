# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 12:42:26 2023

@author: ning
"""

import os, torch
from torchvision import transforms

import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import colors
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader

np.random.seed(12345)
torch.manual_seed(12345)

from utils import concatenate_transform_steps, generator, invTrans


def deg2rad(deg):
    return (deg * np.pi) / 180


def generate_gabor_patch(envelope, frequency, orientation, phase, size, std):
    im_range = np.arange(size)
    x, y = np.meshgrid(im_range, im_range)
    dx = x - size // 2
    dy = y - size // 2
    t = np.arctan2(dy, dx) - deg2rad(orientation)
    r = np.sqrt(dx ** 2 + dy ** 2)
    x = r * np.cos(t)
    y = r * np.sin(t)
    # The amplitude without envelope (from 0 to 1)
    amp = 0.5 + 0.5 * np.cos(2 * np.pi * (x * frequency + phase))
    if envelope == "gaussian":
        f = np.exp(-0.5 * (std / size) * ((x ** 2) + (y ** 2)))
    elif envelope == "linear":
        f = np.max(0, (size // 2 - r) / (size // 2))
    elif envelope == "sine":
        f = np.cos((np.pi * (r + size // 2)) / (size - 1) - np.pi / 2)
        f[r > size // 2] = 0
    elif envelope == "circle":
        f = np.ones_like(r)
        f[r > size // 2] = 0
    else:
        raise ValueError("Envelope type is incorrect!")
    return amp, f


def make_gabor_patch(envelope='sine',
                     frequency=1 / 20,
                     orientation=45,
                     phase=1 / 8,
                     size=244,
                     std=1 / 8,
                     c1='black',
                     c2='white',
                     bg_color='white',
                     image_type='grey',
                     ):
    amp, f = generate_gabor_patch(envelope, frequency, orientation, phase, size, std)
    c1 = np.array(colors.to_rgb(c1))
    c2 = np.array(colors.to_rgb(c2))
    bg_color = np.array(colors.to_rgb(bg_color))

    im = None
    if image_type == 'grey':
        gray_vals = (c1[0] * amp + c2[0] * (1 - amp)) * f + bg_color[0] * (1 - f)
        im = Image.fromarray((gray_vals * 255).astype('uint8'), 'L')  # 'L'模式用于灰度图像
    elif image_type == 'rgb':
        im_rgb_vals = (c1 * amp[:, :, None]) + (c2 * (1 - amp[:, :, None]))
        im_rgb_vals = (im_rgb_vals * f[:, :, None]) + (bg_color * (1 - f[:, :, None]))
        im = Image.fromarray((im_rgb_vals * 255).astype('uint8'), 'RGB')
    else:
        print("无此图片类型!")

    return im


# build dataset
class gabor_image_dataset(Dataset):
    """
    generate gabor images x 2


    """

    def __init__(self,
                 dataframe,
                 image_resize,
                 transform_steps,
                 lamda=4,
                 fill_empty_space=255,
                 sequence_length=100,
                 ):
        # specify the augmentation steps
        if transform_steps == None:
            self.transform_steps = concatenate_transform_steps(image_resize=image_resize,
                                                               fill_empty_space=fill_empty_space,
                                                               grayscale=True,
                                                               )
        else:
            self.transform_steps = transform_steps
        # preallocate
        self.dataframe = dataframe
        self.image_resize = image_resize
        self.lamda = lamda
        self.sequence_length = sequence_length

        self.first_image = self.dataframe['first_image'].values.astype(float)
        self.second_image = self.dataframe['second_image'].values.astype(float)
        self.correct_answer = self.dataframe['correct_answer'].values.astype(float)
        self.rule = self.dataframe['rule'].values.astype(float)
        self.action = self.dataframe['action'].values.astype(float)

    def __len__(self, ):
        # 有重叠
        return len(self.dataframe)

    def __getitem__(self, index):
        rule = self.rule[index: index + self.sequence_length]
        first_image = self.first_image[index: index + self.sequence_length]
        second_image = self.second_image[index: index + self.sequence_length]
        correct_answer = self.correct_answer[index: index + self.sequence_length]
        action = self.action[index: index + self.sequence_length]

        first_image_list = []
        second_image_list = []
        for fi in first_image:
            fi = make_gabor_patch(orientation=fi,
                                  size=self.image_resize, )
            fi = self.transform_steps(fi)
            first_image_list.append(fi)
        for si in second_image:
            si = make_gabor_patch(orientation=si,
                                  size=self.image_resize, )
            si = self.transform_steps(si)
            second_image_list.append(si)
        first_image_list = torch.stack(first_image_list)
        second_image_list = torch.stack(second_image_list)

        # 将 rule 转换为相应的标签
        rule_true_label = []
        for r in rule:
            if r == 0:
                rule_true_label.append([1, 0])  # 0 转换为 [1, 0]
            elif r == 1:
                rule_true_label.append([0, 1])  # 1 转换为 [0, 1]
        rule_true_label = torch.FloatTensor(rule_true_label)

        angle_true_label = []
        for ans in correct_answer:
            if ans == 0:
                angle_true_label.append([1, 0])
            elif ans == 1:
                angle_true_label.append(([0, 1]))
        angle_true_label = torch.FloatTensor(angle_true_label)

        action_true_label = []
        for a in action:
            if a == 0:
                action_true_label.append([1, 0])
            elif a == 1:
                action_true_label.append([0, 1])
        action_true_label = torch.FloatTensor(action_true_label)

        return first_image_list, second_image_list, angle_true_label, rule_true_label, action_true_label


def build_gabor_image_dataloader(df: pd.core.frame.DataFrame,
                                 image_resize: int,
                                 transform_steps,
                                 lamda: int = 16,
                                 fill_empty_space: int = 255,
                                 batch_size: int = 8,
                                 shuffle: bool = False,
                                 num_workers: int = 2,
                                 sequence_length: int = 100
                                 ) -> torch.utils.data.dataloader.DataLoader:
    """_summary_

    Args:
        df (pd.core.frame.DataFrame): _description_
        image_resize (int): _description_
        transform_steps (_type_): _description_
        lamda (int, optional): 控制光栅的频率，越小，越密. Defaults to 16.
        fill_empty_space (int, optional): _description_. Defaults to 255.
        batch_size (int, optional): _description_. Defaults to 8.
        shuffle (bool, optional): _description_. Defaults to False.
        num_workers (int, optional): _description_. Defaults to 2.

    Returns:
        torch.utils.data.dataloader.DataLoader: _description_
    """

    if df is not None:
        dataset = gabor_image_dataset(df,
                                      image_resize=image_resize,
                                      transform_steps=transform_steps,
                                      lamda=lamda,
                                      fill_empty_space=fill_empty_space,
                                      sequence_length=sequence_length,
                                      )
    else:
        df_temp = generate_dataframe(most_left=-20,
                                     most_right=20,
                                     image_B_range=45,
                                     n_trials=int((20 * 2 + 1) * 200),
                                     break_point=None,
                                     testset=False,
                                     )
        df = pd.DataFrame(np.zeros((df_temp.shape[0] * batch_size, 6)), columns=['first_image',
                                                                                 'second_image',
                                                                                 'answers',
                                                                                 'correct_answer',
                                                                                 'rule',
                                                                                 'action'])
        for idx_batch in range(batch_size):
            df.iloc[idx_batch::batch_size] = generate_dataframe(
                most_left=-20,
                most_right=20,
                image_B_range=45,
                n_trials=int((20 * 2 + 1) * 200),
                break_point=None,
                testset=False,
            )
            dataset = gabor_image_dataset(df,
                                          image_resize=image_resize,
                                          transform_steps=transform_steps,
                                          lamda=lamda,
                                          fill_empty_space=fill_empty_space,
                                          sequence_length=sequence_length,
                                          )

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers, )
    return dataloader


def generate_dataframe(most_left: int = -45,
                       most_right: int = 45,
                       image_B_range: int = 45,
                       n_trials: int = 1000,
                       break_point: int = 20,
                       p_flip: float = 0.1,
                       testset: bool = False,
                       ) -> pd.core.frame.DataFrame:
    """_summary_

    Args:
        most_left (int, optional): 正确答案逆时针能够转的最大角度差. Defaults to -30.
        most_right (int, optional): 正确答案顺时针能够转的最大角度差. Defaults to 30.
        image_B_range (int, optional): 第二张光栅的角度范围. Defaults to 45.
        n_trials (int, optional): 总的trial数目. Defaults to 1000.
        break_point (int, optional): 角度差的范围如果不是连续的，例如使用最悬殊的角度差来训练模型，但是我们可以用比较难的角度差去测试模型. 
                                    Defaults to 20.
        p_flip (float, optional): probability to flip the rule. Defaults to 0.1.
        testset (bool,optional): 因为测试集含有0旋转的答案，所以它不是对称的，所以最后的对称检查不适合它
    Returns:
        pd.core.frame.DataFrame: _description_
    """
    if break_point is not None:
        _all_angles = np.concatenate([
            np.arange(most_left, -break_point, ),
            np.arange(break_point, most_right)])
        n_trials_per_angle = int(n_trials / _all_angles.shape[0])
        all_answers = shuffle(np.array([[item] * n_trials_per_angle for item in _all_angles]).flatten())
    else:
        _all_angles = np.arange(most_left, most_right + 1, )
        n_trials_per_angle = int(n_trials / _all_angles.shape[0])
        all_answers = shuffle(np.array([[item] * n_trials_per_angle for item in _all_angles]).flatten())
    # check if the number of answers matches the number of desired trials
    if len(all_answers) < n_trials:
        print(len(all_answers))
        all_answers = np.concatenate([all_answers,
                                      np.random.choice(_all_angles, size=n_trials - len(all_answers), replace=True)])
        print(len(all_answers))
    image_B = np.random.randint(-image_B_range,
                                image_B_range,
                                size=n_trials,
                                )

    image_A = image_B + all_answers

    df = pd.DataFrame(dict(first_image=image_A,
                           second_image=image_B,
                           answers=all_answers))
    df["correct_answer"] = df['answers'].apply(lambda x: x > 0).astype(int)
    # if not testset:
    #     # 保证两种决策答案的比例相同，不然训练出来的模型可能有一定的偏见
    #     df_condition0 = df[df['correct_answer'] == 0]
    #     df_condition1 = df[df['correct_answer'] == 1]
    #     n_trials_for_sample = min(df_condition0.shape[0], df_condition1.shape[0])
    #     #
    #     df_condition0 = df_condition0.sample(int(n_trials_for_sample), replace=False)
    #     df_condition1 = df_condition1.sample(int(n_trials_for_sample), replace=False)
    #     df = pd.concat([df_condition0, df_condition1])
    #     df = df.sample(df.shape[0], replace=False).reset_index(drop=True)
    # 随机生成一些“规则”序列
    initial_rule = np.random.choice([0, 1], size=1, )[0]
    rules = []
    counter = 0
    for ii in range(df.shape[0]):
        if ii == 0:
            k = initial_rule
        else:
            if counter >= 7 and np.random.rand() < p_flip:  # p的概率发生规则转换
                k = 1 - k
                counter = 0
            elif counter >= 20:  # change rule if more than 20 consecutive trials
                k = 1 - k
                counter = 0
            else:
                counter += 1
        rules.append(k)
    rules = np.array(rules)
    df['rule'] = rules
    # xor判断，其实也可以用np.logical_xor
    df['action'] = df['correct_answer'] - df['rule']
    df['action'] = df['action'].apply(np.abs)
    return df


if __name__ == "__main__":
    # np.random.seed(12345)
    # torch.manual_seed(12345)
    # # generate sequence of trials
    # n_trials = int(3e3)
    #
    # # training set
    # df_train = generate_dataframe(n_trials=3200,
    #                               most_left=-45,
    #                               most_right=45,
    #                               break_point=None,
    #                               image_B_range=45,
    #                               )
    # # validation set - determine when to stop training
    # df_valid = generate_dataframe(n_trials=3200,
    #                               most_left=-45,
    #                               most_right=45,
    #                               break_point=None,
    #                               image_B_range=45,
    #                               )
    # # testing set - should the same/similar to the experiment
    # df_test = generate_dataframe(most_left=-20,
    #                              most_right=20,
    #                              image_B_range=45,
    #                              n_trials=int(6400),
    #                              break_point=None,
    #                              testset=True,
    #                              )
    data_dir = '../data/type1_stimuli'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    # df_train.to_csv(os.path.join(data_dir, 'df_train_32.csv'), index=False)
    # df_valid.to_csv(os.path.join(data_dir, 'df_valid_32.csv'), index=False)
    # df_test.to_csv(os.path.join(data_dir, 'df_test_32.csv'), index=False)

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

    df_train = pd.read_csv(os.path.join(data_dir, "df_train_32.csv"))

    dataloader_train = build_gabor_image_dataloader(df=df_train,
                                                    image_resize=image_resize,
                                                    transform_steps=transform_steps,
                                                    shuffle=False,
                                                    batch_size=batch_size
                                                    )

    count = 0
    for first_image_list, second_image_list, angle_true_label, rule_true_label, action_true_label in dataloader_train:
        count += 1
        print("count: ", count)

