# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 12:42:26 2023

@author: ning
"""

import os, torch
import numpy as np

torch.manual_seed(12345)
np.random.seed(12345)

import torch.nn as nn
from torchvision import transforms

from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt

from sklearn.metrics import roc_auc_score

from typing import List, Callable, Union, Any, TypeVar, Tuple

###############################################################################
Tensor = TypeVar('torch.tensor')
###############################################################################
invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                               transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                    std=[1., 1., 1.]),
                               ])
normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])


###############################################################################
def noise_func(x, noise_level):
    """
    add guassian noise to the images during agumentation procedures
    Inputs
    --------------------
    x: torch.tensor, batch_size x 3 x height x width
    noise_level: float, level of noise, between 0 and 1
    """

    generator = torch.distributions.Normal(x.mean(), x.std())
    noise = generator.sample(x.shape)
    new_x = x * (1 - noise_level) + noise * noise_level
    new_x = torch.clamp(new_x, x.min(), x.max(), )
    return new_x


def concatenate_transform_steps(image_resize: int = 128,
                                num_output_channels: int = 3,
                                noise_level: float = 0.,
                                flip: bool = False,
                                rotate: float = 0.,
                                fill_empty_space: int = 255,
                                grayscale: bool = True,
                                center_crop: bool = False,
                                center_crop_size: tuple = (1200, 1200),
                                ):
    """
    from image to tensors

    Parameters
    ----------
    image_resize : int, optional
        DESCRIPTION. The default is 128.
    num_output_channels : int, optional
        DESCRIPTION. The default is 3.
    noise_level : float, optional
        DESCRIPTION. The default is 0..
    flip : bool, optional
        DESCRIPTION. The default is False.
    rotate : float, optional
        DESCRIPTION. The default is 0.,
    fill_empty_space : int, optional
        DESCRIPTION. The defaultis 130.
    grayscale: bool, optional
        DESCRIPTION. The default is True.
    center_crop : bool, optional
        DESCRIPTION. The default is False.
    center_crop_size : Tuple, optional
        DESCRIPTION. The default is (1200, 1200)

    Returns
    -------
    transformer_steps : TYPE
        DESCRIPTION.

    """
    transformer_steps = []
    # crop the image - for grid like layout
    if center_crop:
        transformer_steps.append(transforms.CenterCrop(center_crop_size))
    # resize
    transformer_steps.append(transforms.Resize((image_resize, image_resize)))
    # flip
    if flip:
        transformer_steps.append(transforms.RandomHorizontalFlip(p=.5))
        transformer_steps.append(transforms.RandomVerticalFlip(p=.5))
    # rotation
    if rotate > 0.:
        transformer_steps.append(transforms.RandomRotation(degrees=rotate,
                                                           fill=fill_empty_space,
                                                           ))
    # grayscale
    if grayscale:
        transformer_steps.append(  # it needs to be 3 if we want to use pretrained CV models
            transforms.Grayscale(num_output_channels=num_output_channels)
        )
    # rescale to [0,1] from int8
    transformer_steps.append(transforms.ToTensor())
    # add noise
    if noise_level > 0:
        transformer_steps.append(transforms.Lambda(lambda x: noise_func(x, noise_level)))
    # normalization
    if num_output_channels == 3:
        transformer_steps.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])
                                 )
    elif num_output_channels == 1:
        transformer_steps.append(transforms.Normalize(mean=[0.5], std=[0.5]))
        transformer_steps = transforms.Compose(transformer_steps)
    return transformer_steps


invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                               transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                    std=[1., 1., 1.]),
                               ])


def generator(image_size=224, lamda=4, thetaRad_base=45, ):
    """
    Inputs
    -------------
    image_size: int, image size
    lamda: float, better be in range between 4 and 32
    thetaRad_base:float, base value of theta, in degrees
    """

    # convert degree to pi-based
    thetaRad = thetaRad_base * np.pi / 180
    # Sanjeev's algorithm
    X = np.arange(image_size)
    X0 = (X / image_size) - .5
    freq = image_size / lamda
    Xf = X0 * freq * 2 * np.pi
    sinX = np.sin(Xf)
    Xm, Ym = np.meshgrid(X0, X0)
    Xt = Xm * np.cos(thetaRad)
    Yt = Ym * np.sin(thetaRad)
    XYt = Xt + Yt
    XYf = XYt * freq * 2 * np.pi

    grating = np.sin(XYf)

    s = 0.075
    w = np.exp(-(0.3 * ((Xm ** 2) + (Ym ** 2)) / (2 * s ** 2))) * 2
    w[w > 1] = 1
    gabor = ((grating - 0.5) * w) + 0.5

    plt.close('all')
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(gabor, cmap=plt.cm.gray)
    ax.axis('off')
    return fig2img(fig)


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def determine_training_stops(nets: List[nn.Module],
                             idx_epoch: int,
                             warmup_epochs: int,
                             valid_loss: Tensor,
                             counts: int = 0,
                             best_valid_loss=np.inf,
                             tol: float = 1e-4,
                             saving_names: dict = {'model_saving_name': True, },
                             ) -> Tuple[Tensor, int]:
    """
   

    Parameters
    ----------
    nets : List[nn.Module]
        DESCRIPTION. 
    idx_epoch : int
        DESCRIPTION.
    warmup_epochs : int
        DESCRIPTION.
    valid_loss : Tensor
        DESCRIPTION.
    counts : int, optional
        DESCRIPTION. The default is 0.
    best_valid_loss : TYPE, optional
        DESCRIPTION. The default is np.inf.
    tol : float, optional
        DESCRIPTION. The default is 1e-4.
    saving_names : dict, optional
        DESCRIPTION. The default is {'model_saving_name':True,}.

    Returns
    -------
    Tuple[Tensor,int]
        DESCRIPTION.

    """
    if idx_epoch >= warmup_epochs:  # warming up
        temp = valid_loss
        if np.logical_and(temp < best_valid_loss, np.abs(best_valid_loss - temp) >= tol):
            best_valid_loss = valid_loss
            for net, (saving_name, save_model) in zip(nets, saving_names.items()):
                if save_model:
                    torch.save(net.state_dict(), saving_name)  # why do i need state_dict()?
                    print(f'save {saving_name}')
            counts = 0
        else:
            counts += 1

    return best_valid_loss, counts


def add_noise_instance_for_training(batch_features: Tensor,
                                    n_noise: int = 1,
                                    clip_output: bool = False,
                                    ) -> Tensor:
    """
    

    Parameters
    ----------
    batch_features : Tensor
        DESCRIPTION.
    n_noise : int, optional
        DESCRIPTION. The default is 1.
    clip_output : bool, optional
        DESCRIPTION. The default is False.
    Returns
    -------
    batch_features : Tensor
        DESCRIPTION.

    """
    if n_noise > 0:
        noise_generator = torch.distributions.normal.Normal(batch_features.mean(),
                                                            batch_features.std(), )
        noise_features = noise_generator.sample(batch_features.shape)[:n_noise]
        if clip_output:
            temp = invTrans(batch_features[:n_noise])
            idx_pixels = torch.where(temp == 1)
            temp = invTrans(noise_features)
            temp[idx_pixels] = 1
            noise_features = normalizer(temp)
        batch_features = torch.cat([batch_features, noise_features])
    else:
        pass
    return batch_features


def fit_one_cycle(
        perceptual_network,
        action_network,
        dataloader,
        optimizer,
        loss_func,
        device='cpu',
        idx_epoch=0,
        train=True,
        verbose=0,
        n_noise=2,
):
    """这个方程会训练和检验模型，这个训练和检验是一个循环。
    """
    if train:
        perceptual_network.train(True).to(device)
        action_network.train(True).to(device)
    else:
        perceptual_network.eval().to(device)
        action_network.eval().to(device)
    loss = 0
    yp_true, yd_true = [], []
    yp_pred, yd_pred = [], []
    iterator = tqdm(enumerate(dataloader))
    for idx_batch, (image1, image2, batch_label, angle1, angle2, rule, action) in iterator:
        rule = torch.vstack([1 - rule, rule]).T.float()
        action = torch.vstack([1 - action, action]).T.float()
        if train and np.random.randn() > 0.5:  # 50% chance we add noise to the images
            image1 = noise_func(image1, noise_level=np.random.uniform(1e-3, 2e-1, size=1)[0])
            image2 = noise_func(image2, noise_level=np.random.uniform(1e-3, 2e-1, size=1)[0])

        if n_noise > 0:
            image1 = add_noise_instance_for_training(image1, n_noise, clip_output=True, )
            image2 = add_noise_instance_for_training(image2, n_noise, clip_output=True, )
            noisy_labels = torch.ones(batch_label.shape) * (1 / 2)
            noisy_labels = noisy_labels[:n_noise]
            batch_label = torch.cat([batch_label.to(device), noisy_labels.to(device)])
            rule = torch.cat([rule.to(device), noisy_labels.to(device)])
            action = torch.cat([action.to(device), noisy_labels.to(device)])
        # zero grad
        optimizer.zero_grad()

        # forward pass
        _, perceptual_prediction = perceptual_network([image1.to(device), image2.to(device)], )
        decision = action_network(perceptual_prediction, rule.to(device))

        # compute loss
        # log_predictions = torch.log(perceptual_prediction)
        batch_perceptual_loss = loss_func(perceptual_prediction.float().to(device),
                                          batch_label.float().to(device), )
        # log_decision = torch.log(decision)
        batch_decision_loss = loss_func(decision.float().to(device),
                                        action.float().to(device), )
        batch_loss = batch_perceptual_loss + batch_decision_loss
        loss += batch_loss.item()  # use this function to save memory

        # append labels and predictions
        if n_noise > 0:
            ## perceptual
            yp_true.append(batch_label.detach().cpu().numpy()[:-n_noise])
            yp_pred.append(perceptual_prediction.detach().cpu().numpy()[:-n_noise])
            ## action
            yd_true.append(action.detach().cpu().numpy()[:-n_noise])
            yd_pred.append(decision.detach().cpu().numpy()[:-n_noise])
        else:
            ## perceptual
            yp_true.append(batch_label.detach().cpu().numpy())
            yp_pred.append(perceptual_prediction.detach().cpu().numpy())
            ## action
            yd_true.append(action.detach().cpu().numpy())
            yd_pred.append(decision.detach().cpu().numpy())

        if train:
            # bachpropagation
            batch_loss.backward()
            # modify weights
            optimizer.step()

        # print message
        if verbose > 0:
            try:
                score_p = roc_auc_score(np.concatenate(yp_true),
                                        np.concatenate(yp_pred))
                score_d = roc_auc_score(np.concatenate(yd_true),
                                        np.concatenate(yd_pred))
            except Exception as e:
                print(e)
                score_p = np.nan
                score_d = np.nan
            message = f"""epoch {idx_epoch + 1}-
{idx_batch + 1:3.0f}-{len(dataloader):4.0f}/{100 * (idx_batch + 1) / len(dataloader):2.3f}%, 
loss = {loss / (idx_batch + 1):2.4f}, S(perceptual) = {score_p:.4f}, S(decision) = {score_d:.4f}
""".replace('\n', '')
            iterator.set_description(message)
    return perceptual_network, action_network, loss / (idx_batch + 1)


def train_valid_loop(
        perceptual_network,
        action_network,
        dataloader_train,
        dataloader_valid,
        optimizer,
        loss_func,
        device='cpu',
        n_epochs=int(1e3),
        verbose=0,
        patience=5,
        warmup_epochs=5,
        tol=1e-4,
        n_noise=2,
        saving_names={'model_saving_name': True, },
):
    """这个方程利用一个循环的训练和检验方程，多循环地训练和检验模型，直达模型的valid loss在连续多次检验后都
    没有变得更好。

    """
    best_valid_loss = np.inf
    losses = []
    counts = 0
    for idx_epoch in range(n_epochs):
        print('Training ...')
        perceptual_network, action_network, loss_train = fit_one_cycle(
            perceptual_network,
            action_network,
            dataloader_train,
            optimizer,
            loss_func,
            device=device,
            idx_epoch=idx_epoch,
            train=True,
            verbose=verbose,
            n_noise=n_noise,
        )
        print('Validating ...')
        with torch.no_grad():
            _, _, loss_valid = fit_one_cycle(
                perceptual_network,
                action_network,
                dataloader_valid,
                optimizer,
                loss_func,
                device=device,
                idx_epoch=idx_epoch,
                train=False,
                verbose=verbose,
                n_noise=0,
            )

        losses.append([loss_train, loss_valid])

        best_valid_loss, counts = determine_training_stops([perceptual_network, action_network, ],
                                                           idx_epoch=idx_epoch,
                                                           warmup_epochs=warmup_epochs,
                                                           valid_loss=loss_valid,
                                                           counts=counts,
                                                           best_valid_loss=best_valid_loss,
                                                           tol=tol,
                                                           saving_names=saving_names,
                                                           )
        if counts > patience:
            break
        else:
            message = f"""epoch {idx_epoch + 1}, 
best validation loss = {best_valid_loss:.4f},count = {counts}
""".replace('\n', '')
            print(message)
    return perceptual_network, action_network, losses


def fit_rule_one_cycle(
        full_model,
        dataloader,
        optimizer,
        loss_func,
        batch_size=1,
        device='cpu',
        idx_epoch=0,
        train=True,
        verbose=0, ):
    perceptual_network, action_network, rule_network = full_model
    if train:
        rule_network.train(True).to(device)
        # action_network.train(True).to(device)
    else:
        rule_network.eval().to(device)
    action_network.eval().to(device)
    perceptual_network.eval()
    loss = 0
    yp_true, yd_true, yr_true = [], [], []
    yp_pred, yd_pred, yr_pred = [], [], []
    iterator = tqdm(enumerate(dataloader))
    for idx_trial, (image1, image2, batch_label, angle1, angle2, rule, action) in iterator:
        current_rule = torch.vstack([1 - rule, rule]).T.float()
        current_decision = torch.vstack([1 - action, action]).T.float()
        if train and np.random.randn() > 0.5:  # 50% chance we add noise to the images
            image1 = noise_func(image1, noise_level=np.random.uniform(0.1, 0.2, size=1)[0])
            image2 = noise_func(image2, noise_level=np.random.uniform(0.1, 0.2, size=1)[0])
        # zero grad
        optimizer.zero_grad()
        if idx_trial == 0:
            # initialize for the first trial
            hidden_state = torch.rand((rule_network.n_lstm, batch_size,
                                       rule_network.lstm_hidden)).to(device)
            cell_state = torch.rand((rule_network.n_lstm, batch_size,
                                     rule_network.lstm_hidden)).to(device)
            last_predicted_perceptual_label = torch.ones(batch_label.shape).float().detach() * 0.5
            last_true_rule = torch.vstack([1 - rule, rule]).T.float()
            last_predicted_decision = torch.ones(batch_label.shape).float().detach() * 0.5
            last_correctness = torch.ones(batch_label.shape).float().detach() * 0.5
            last_predicted_rule = torch.vstack([1 - rule, rule]).T.float()

        # forward pass
        with torch.no_grad():
            outputs, current_predicted_perceptual_label = perceptual_network([image1.to(device),
                                                                              image2.to(device)])
        rule_network_input = [
            last_predicted_perceptual_label.view(batch_size, 1, last_predicted_perceptual_label.shape[-1]).float().to(
                device),
            last_true_rule.view(batch_size, 1, last_predicted_decision.shape[-1]).float().to(device),
            last_predicted_decision.view(batch_size, 1, last_predicted_decision.shape[-1]).float().to(device),
            last_predicted_rule.view(batch_size, 1, last_predicted_rule.shape[-1]).float().to(device),
            last_correctness.view(batch_size, 1, last_correctness.shape[-1]).float().to(device),
        ]
        current_predicted_rule, (hidden_state, cell_state) = rule_network(rule_network_input,
                                                                          hidden_state=hidden_state.detach(),
                                                                          cell_state=cell_state.detach(), )
        current_predicted_decision = action_network(current_predicted_perceptual_label,
                                                    current_predicted_rule,
                                                    )
        # update for the next trial
        last_predicted_perceptual_label = current_predicted_perceptual_label.detach()
        last_predicted_rule = current_predicted_rule.to(device).detach()
        last_predicted_decision = current_predicted_decision.detach()
        last_true_rule = torch.vstack([1 - rule, rule]).T.float()
        correctness = last_predicted_decision.argmax(1) == action[0]
        correctness = correctness.long()
        last_correctness = torch.vstack([1 - correctness, correctness]).T.detach()

        # recording
        ## true perceptual label
        yp_true.append(batch_label.detach().cpu().numpy())
        ## true action/key label
        yd_true.append(action.detach().cpu().numpy())
        ## true rule label
        yr_true.append(rule.detach().cpu().numpy())
        ## predicted perceptual label
        yp_pred.append(current_predicted_perceptual_label.detach().cpu().numpy())
        ## predicted action/key label
        yd_pred.append(current_predicted_decision.detach().cpu().numpy())
        ## predicted rule label
        yr_pred.append(current_predicted_rule.detach().cpu().numpy())

        # compute loss
        batch_perceptual_loss = loss_func(current_predicted_perceptual_label.float().to(device),
                                          batch_label.float().to(device), )
        batch_rule_loss = loss_func(current_predicted_rule.to(device), current_rule.to(device))
        batch_decision_loss = loss_func(current_predicted_decision.to(device), current_decision.to(device))
        batch_loss = batch_perceptual_loss + batch_rule_loss + batch_decision_loss
        loss += batch_loss.item()

        if train and idx_trial > 0:
            # bachpropagation
            batch_loss.backward()
            # modify weights
            optimizer.step()

        # print message
        if verbose > 0:
            try:
                score_p = roc_auc_score(np.concatenate(yp_true),
                                        np.concatenate(yp_pred))
                score_d = roc_auc_score(np.concatenate(yd_true),
                                        np.concatenate(yd_pred).argmax(1))  # [:,-1])
                score_r = roc_auc_score(np.concatenate(yr_true),
                                        np.concatenate(yr_pred).argmax(1))  # [:,-1])
            except Exception as e:
                # print(e)
                score_r = np.nan
                score_p = np.nan
                score_d = np.nan
            message = f"""epoch {idx_epoch}-
{idx_trial + 1:4.0f}-{len(dataloader):4.0f}/{100 * (idx_trial + 1) / len(dataloader):2.3f}%, 
loss = {loss / (idx_trial + 1):2.4f}, S(perctual) = {score_p:.4f},S(decision) = {score_d:.4f},S(rule) = {score_r:.4f}
""".replace('\n', '')
            iterator.set_description(message)
    return (perceptual_network, action_network, rule_network), loss / (idx_trial + 1)


def rule_train_valid_loop(
        full_model,
        dataloader_train,
        dataloader_valid,
        optimizer,
        loss_func,
        device='cpu',
        n_epochs=int(1e3),
        verbose=0,
        batch_size=8,
        patience=5,
        warmup_epochs=5,
        tol=1e-4,
        saving_names={'model_saving_name': True, },
):
    """这个方程利用一个循环的训练和检验方程，多循环地训练和检验模型，直达模型的valid loss在连续多次检验后都
    没有变得更好。
    """

    best_valid_loss = np.inf
    losses = []
    counts = 0
    for idx_epoch in range(n_epochs):
        print('Training ...')
        (perceptual_network, action_network, rule_network), loss_train = fit_rule_one_cycle(
            full_model,
            dataloader_train,
            optimizer,
            loss_func,
            batch_size=batch_size,
            device=device,
            idx_epoch=idx_epoch,
            train=True,
            verbose=verbose,
        )
        print('Validating ...')
        with torch.no_grad():
            _, loss_valid = fit_rule_one_cycle(
                full_model,
                dataloader_valid,
                optimizer,
                loss_func,
                batch_size=batch_size,
                device=device,
                idx_epoch=idx_epoch,
                train=False,
                verbose=verbose,
            )

        losses.append([loss_train, loss_valid])

        best_valid_loss, counts = determine_training_stops([perceptual_network, action_network, rule_network],
                                                           idx_epoch=idx_epoch,
                                                           warmup_epochs=warmup_epochs,
                                                           valid_loss=loss_valid,
                                                           counts=counts,
                                                           best_valid_loss=best_valid_loss,
                                                           tol=tol,
                                                           saving_names=saving_names,
                                                           )
        if counts > patience:
            break
        else:
            message = f"""epoch {idx_epoch + 1}, 
best validation loss = {best_valid_loss:.4f},count = {counts}
""".replace('\n', '')
            print(message)
    return (perceptual_network, action_network, rule_network), losses
