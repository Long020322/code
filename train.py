import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math  # 确保导入 math 模块

from sklearn.metrics import silhouette_score, calinski_harabasz_score
from config import Config
from models import CP
from utils import accuracy, record, EarlyStopping
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.autograd.set_detect_anomaly(True)

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--gpuid', type=int, default=0)
parser.add_argument('--mm', type=int, default=0)
parser.add_argument('--dd', type=int, default=0)
parser.add_argument('--patience', type=int, default=100)
args = parser.parse_args()

if torch.cuda.is_available():
    torch.cuda.set_device(args.gpuid)
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cuda')

if __name__ == "__main__":
    Models = ['CP']
    Datasets = ['Abilene']
    config = Config('./data/' + Datasets[args.dd] + '.ini')
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    train_ratio = 0.1
    error_list = []
    checkpoint_dir = ('./results/checkpoint/{}_{}_{}.pt').format(Models[args.mm], Datasets[args.dd], train_ratio)

    # 加载采样后的数据
    tr_idxs, tr_vals, va_idxs, va_vals, te_idxs, te_vals = config.Sampling(train_ratio)

    # 初始化3个CP模型（对应三个标签）
    cp_models = [CP(config.num_dim, config.num_emb).cuda() for _ in range(3)]
    optimizers = [optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay) for model in cp_models]
    criterion = nn.MSELoss()

    # 为每个标签创建独立的 EarlyStopping 对象
    early_stopping = [EarlyStopping(patience=args.patience, verbose=True, save_path=f'{checkpoint_dir}_label{label}.pt') for label in range(3)]
    stop_flags = [False] * 3  # 初始时，所有模型的 stop_flags 都是 False

    # 在 for epoch 循环前初始化最佳分数
    for label in range(3):
        if early_stopping[label].best_score is None:
            early_stopping[label].best_score = float('inf')  # 初始化为无穷大

    for epoch in range(config.epochs):
        print(f'Epoch {epoch + 1}/{config.epochs}')

        for label in range(3):
            if stop_flags[label]:
                continue

            cp_models[label].train()
            train_loss = 0

            # 选取对应标签的数据
            selected_data = tr_idxs[tr_idxs[:, 4] == label]
            selected_vals = tr_vals[tr_idxs[:, 4] == label]

            if len(selected_data) == 0:
                continue

            # 随机打乱数据索引
            random_perm_idx = torch.randperm(len(selected_data))
            selected_data = selected_data[random_perm_idx]
            selected_vals = selected_vals[random_perm_idx]

            num_batches = math.ceil(len(selected_data) / config.batch_size)

            for n in range(num_batches):
                batch_data = selected_data[n * config.batch_size : (n + 1) * config.batch_size]
                batch_vals = selected_vals[n * config.batch_size : (n + 1) * config.batch_size]

                i = batch_data[:, 0]
                j = batch_data[:, 1]
                k = batch_data[:, 2]

                optimizers[label].zero_grad()
                predict = cp_models[label](i, j, k)
                loss = criterion(predict, batch_vals)
                loss.backward()
                optimizers[label].step()

                train_loss += loss.item()

            train_loss /= num_batches
            print(f"Label {label} - Train Loss: {train_loss}")


            # 验证模型
            cp_models[label].eval()
            valid_loss = 0
            selected_data = va_idxs[va_idxs[:, 4] == label]
            selected_vals = va_vals[va_idxs[:, 4] == label]

            if len(selected_data) > 0:
                i = selected_data[:, 0]
                j = selected_data[:, 1]
                k = selected_data[:, 2]
                with torch.no_grad():
                    valid_output = cp_models[label](i, j, k)
                    valid_loss = criterion(valid_output, selected_vals).item()

            # 早停检查
            early_stopping[label](valid_loss, cp_models[label])

            if early_stopping[label].early_stop:
                print(f"Early stopping for label {label}")
                stop_flags[label] = True  # 标记该模型不再继续训练

        # 如果所有模型都已经早停，则提前结束整个训练过程
        if all(stop_flags):
            print("All models have early stopped. Ending training.")
            break


    print("Testing...")

    # 在测试前加载早停保存的最佳模型参数
    for label in range(3):
        checkpoint_path = f'{checkpoint_dir}_label{label}.pt'
        if os.path.exists(checkpoint_path):
            cp_models[label].load_state_dict(torch.load(checkpoint_path))
    
    cp_models = [model.eval() for model in cp_models]  # 确保所有模型都处于评估模式

    # 遍历整个训练集
    with torch.no_grad():
        Estimated_train = []
        for n in range(config.num_batch):
            tr_idxs_batch = tr_idxs[n * config.batch_size_eval: (n + 1) * config.batch_size_eval]
            labels = tr_idxs_batch[:, 4]

            for i in range(len(tr_idxs_batch)):
                label = labels[i].item()
                i_idx = tr_idxs_batch[i, 0].unsqueeze(0)
                j_idx = tr_idxs_batch[i, 1].unsqueeze(0)
                k_idx = tr_idxs_batch[i, 2].unsqueeze(0)

                predict = cp_models[label](i_idx, j_idx, k_idx)
                Estimated_train += predict.cpu().numpy().tolist()

    Estimated_train = np.asarray(Estimated_train)
    train_nmae, train_nrmse = accuracy(Estimated_train, tr_vals.cpu().detach().numpy())
    print(f'Train NMAE: {train_nmae:.4f}, Train NRMSE: {train_nrmse:.4f}')

    # 遍历整个测试集
    with torch.no_grad():
        Estimated = []
        for n in range(config.num_batch_test):
            te_idxs_batch = te_idxs[n * config.batch_size_eval: (n + 1) * config.batch_size_eval]
            labels = te_idxs_batch[:, 4]

            for i in range(len(te_idxs_batch)):
                label = labels[i].item()
                i_idx = te_idxs_batch[i, 0].unsqueeze(0)
                j_idx = te_idxs_batch[i, 1].unsqueeze(0)
                k_idx = te_idxs_batch[i, 2].unsqueeze(0)

                predict = cp_models[label](i_idx, j_idx, k_idx)
                Estimated += predict.cpu().numpy().tolist()

    Estimated = np.asarray(Estimated)
    test_nmae, test_nrmse = accuracy(Estimated, te_vals)
    print(f'Test NMAE: {test_nmae:.4f}, Test NRMSE: {test_nrmse:.4f}')
