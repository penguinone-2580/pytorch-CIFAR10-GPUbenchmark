import torch
import torch.nn as nn
import os
import math
import time
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import Construct_Model
from dataloader import DataLoader
from parameter_loader import read_parameters, str_to_bool, get_gpu_info
from logger import Logger

class Trainer:
    def __init__(self, setting_csv_path, index):
        self.parameters_dict = read_parameters(setting_csv_path, index)                                         #csvファイルからパラメータを読み込み
        self.gpu_info = get_gpu_info()                                                                          #nvidia-smi --query-gpuからGPU情報を取得
        self.model_name = self.parameters_dict["model_name"]                                                    #モデル名
        self.gpu_name = self.gpu_info["name"]                                                                   #GPU名
        self.log_path = os.path.join(self.parameters_dict["base_log_path"], self.gpu_name, self.model_name)     #ログの保存先
        self.batch_size = int(self.parameters_dict["batch_size"])                                               #バッチサイズ
        self.epochs = int(self.parameters_dict["epochs"])                                                       #エポック数
        self.learning_rate = float(self.parameters_dict["learning_rate"])                                       #学習率
        self.momentum = float(self.parameters_dict["momentum"])                                                 #慣性項
        self.weight_decay = float(self.parameters_dict["weight_decay"])                                         #重み減衰
        self.logger = Logger(self.log_path)                                                                     #ログ書き込みを行うLoggerクラスの宣言
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                              #GPUが利用可能であればGPUを利用
        self.model = Construct_Model(self.model_name).model.to(self.device)                                     #モデル
        self.loss_func = nn.CrossEntropyLoss()                                                                  #損失関数
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)                        #最適化手法
        self.data_loader = DataLoader(dataset_path=self.parameters_dict["data_path"],                           #データローダ
                                      batch_size=int(self.parameters_dict["batch_size"]),
                                      num_workers=int(self.parameters_dict["num_workers"]),
                                      pin_memory=str_to_bool(self.parameters_dict["pin_memory"]))


    def train(self):
        print("")
        print("-----------------------------------------")
        print("GPU name:", self.gpu_name)
        print("Start train")
        print("Current time:", datetime.datetime.now())
        print("Start measurement")
        print("")
        torch.backends.cudnn.benchmark = True
        scaler = torch.cuda.amp.GradScaler()
        start_time = time.time()

        with tqdm(range(self.epochs)) as progress_bar:
            for epoch in enumerate(progress_bar):
                i = epoch[0]
                progress_bar.set_description("[Epoch %d]" % (i+1))
                loss_result = 0.0
                acc = 0.0
                val_loss_result = 0.0
                val_acc = 0.0

                self.model.train()
                for inputs, labels in self.data_loader.train_loader:
                    inputs = inputs.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    self.optimizer.zero_grad()

                    with torch.cuda.amp.autocast():
                        output = self.model(inputs)
                        loss = self.loss_func(output, labels)

                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()

                    _, preds = torch.max(output, 1)
                    loss_result += loss.item()
                    acc += torch.sum(preds==labels.data)

                else:
                    with torch.no_grad():
                        self.model.eval()
                        for val_inputs, val_labels in self.data_loader.val_loader:
                            val_inputs = val_inputs.to(self.device, non_blocking=True)
                            val_labels = val_labels.to(self.device, non_blocking=True)

                            with torch.cuda.amp.autocast():
                                val_output = self.model(val_inputs)
                                val_loss = self.loss_func(val_output, val_labels)

                            _, val_preds = torch.max(val_output, 1)
                            val_loss_result += val_loss.item()
                            val_acc += torch.sum(val_preds==val_labels.data)

                    epoch_loss = loss_result / len(self.data_loader.train_loader.dataset)
                    epoch_acc = acc.float() / len(self.data_loader.train_loader.dataset)
                    val_epoch_loss = val_loss_result / len(self.data_loader.val_loader.dataset)
                    val_epoch_acc = val_acc.float() / len(self.data_loader.val_loader.dataset)
                    self.logger.collect_history(loss=epoch_loss, accuracy=epoch_acc, val_loss=val_epoch_loss, val_accuracy=val_epoch_acc)

                progress_bar.set_postfix({"loss":epoch_loss, "accuracy": epoch_acc.item(), "val_loss":val_epoch_loss, "val_accuracy": val_epoch_acc.item()})

        end_time = time.time() - start_time
        print("")
        print("Train ended")
        print("Erapsed time: %ds" %(end_time))
        print("-----------------------------------------")
        print("Recording logs...")

        txt_name = "log.txt"
        with open(os.path.join(self.log_path,txt_name), mode="w") as f:
            f.write(self.gpu_name + "\n" + str(end_time))

        torch.save(self.model.state_dict(), os.path.join(self.log_path,self.model_name))
        self.logger.draw_graph()
