from collections import Counter
import json
import random
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import pickle
import torch
import re
from torch.utils.data import DataLoader
from .dataset import elmo_Dataset
from .elmo_xc import Elmo_Net
import numpy as np
from torch.autograd import Variable
import logging
import time

use_gpu = torch.cuda.is_available()

learning_rate = 0.002
num_epochs = 50
batch_size = 8


net = Elmo_Net()
if use_gpu:
    net.cuda()

net.train()
# different learning rate
params = []
params_dict = dict(net.named_parameters())
for key, value in params_dict.items():
    if key.startswith('features'):
        params += [{'params': [value], 'lr': learning_rate * 1}]
    else:
        params += [{'params': [value], 'lr': learning_rate}]
optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)

train_dataset = elmo_Dataset(train= True)#S=7, B=2, C=16,
train_loader = DataLoader(train_dataset, batch_size=batch_size)# shuffle=True
# test_dataset = yoloDataset(root=file_root, img_size=448, transforms=[transforms.ToTensor()],train=False)#S=7, B=2, C=16,
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
##
print('the dataset has %d images' % (len(train_dataset)))
print('the batch_size is %d' % (batch_size))
logfile = open('log.txt', 'w')

num_iter = 0
# vis = Visualizer(env='xiong')
best_test_loss = np.inf







def train_model(epoch, opt, model, optimizer,
                train, valid, test, best_train, best_valid, test_result):
    """
    Training model for one epoch
    :param epoch:
    :param opt:
    :param model:
    :param optimizer:
    :param train:
    :param best_train:
    :param valid:
    :param best_valid:
    :param test:
    :param test_result:
    :return:
    """
    model.train()

    total_loss, total_tag = 0.0, 0
    cnt = 0
    start_time = time.time()

    train_w, train_c, train_lens, train_masks = train

    lst = list(range(len(train_w)))
    random.shuffle(lst)

    train_w = [train_w[l] for l in lst]
    train_c = [train_c[l] for l in lst]
    train_lens = [train_lens[l] for l in lst]
    train_masks = [train_masks[l] for l in lst]

    for w, c, lens, masks in zip(train_w, train_c, train_lens, train_masks):
        cnt += 1
        model.zero_grad()
        loss_forward, loss_backward = model.forward(w, c, masks)

        loss = (loss_forward + loss_backward) / 2.0
        total_loss += loss_forward.data[0]
        n_tags = sum(lens)
        total_tag += n_tags
        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), opt.clip_grad)
        optimizer.step()
        if cnt * opt.batch_size % 1024 == 0:
            logging.info("Epoch={} iter={} lr={:.6f} train_ppl={:.6f} time={:.2f}s".format(
                epoch, cnt, optimizer.param_groups[0]['lr'],
                np.exp(total_loss / total_tag), time.time() - start_time
            ))
            start_time = time.time()

        if cnt % opt.eval_steps == 0 or cnt % len(train_w) == 0:
            if valid is None:
                train_ppl = np.exp(total_loss / total_tag)
                logging.info("Epoch={} iter={} lr={:.6f} train_ppl={:.6f}".format(
                    epoch, cnt, optimizer.param_groups[0]['lr'], train_ppl))
                if train_ppl < best_train:
                    best_train = train_ppl
                    logging.info("New record achieved on training dataset!")
                    model.save_model(opt.model, opt.save_classify_layer)
            else:
                valid_ppl = eval_model(model, valid)
                logging.info("Epoch={} iter={} lr={:.6f} valid_ppl={:.6f}".format(
                    epoch, cnt, optimizer.param_groups[0]['lr'], valid_ppl))

                if valid_ppl < best_valid:
                    model.save_model(opt.model, opt.save_classify_layer)
                    best_valid = valid_ppl
                    logging.info("New record achieved!")

                    if test is not None:
                        test_result = eval_model(model, test)
                        logging.info("Epoch={} iter={} lr={:.6f} test_ppl={:.6f}".format(
                            epoch, cnt, optimizer.param_groups[0]['lr'], test_result))
    return best_train, best_valid, test_result


