from typing import overload
import torch
from numpy.lib import save
from util import Logger, accuracy, TotalMeter
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from util.prepossess import mixup_criterion, mixup_data
from util.loss import mixup_cluster_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BasicTrain:

    def __init__(self, train_config, model, optimizers, dataloaders, log_folder) -> None:
        self.logger = Logger()
        self.model = model.to(device)
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders
        self.epochs = train_config['epochs']
        self.optimizers = optimizers
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

        self.group_loss = train_config['group_loss']
      

        self.sparsity_loss = train_config['sparsity_loss']
        self.sparsity_loss_weight = train_config['sparsity_loss_weight']
        self.save_path = log_folder

        self.save_learnable_graph = True

        self.init_meters()

    def init_meters(self):
        self.train_loss, self.val_loss, self.test_loss, self.train_accuracy,\
            self.val_accuracy, self.test_accuracy, self.edges_num = [
                TotalMeter() for _ in range(7)]

        self.loss1, self.loss2, self.loss3 = [TotalMeter() for _ in range(3)]

    def reset_meters(self):
        for meter in [self.train_accuracy, self.val_accuracy, self.test_accuracy,
                      self.train_loss, self.val_loss, self.test_loss, self.edges_num,
                      self.loss1, self.loss2, self.loss3]:
            meter.reset()

    def train_per_epoch(self, optimizer):

        self.model.train()

        for data_in, pearson, label in self.train_dataloader:
            label = label.long()

            data_in, pearson, label = data_in.to(
                device), pearson.to(device), label.to(device)

            inputs, nodes, targets_a, targets_b, lam = mixup_data(
                data_in, pearson, label, 1, device)

            output, learnable_matrix, edge_variance = self.model(inputs, nodes)

            loss = 2 * mixup_criterion(
                self.loss_fn, output, targets_a, targets_b, lam)

            if self.group_loss:
                loss += mixup_cluster_loss(learnable_matrix,
                                           targets_a, targets_b, lam)

            if self.sparsity_loss:
                sparsity_loss = self.sparsity_loss_weight * \
                    torch.norm(learnable_matrix, p=1)
                loss += sparsity_loss

            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            top1 = accuracy(output, label)[0]
            self.train_accuracy.update_with_weight(top1, label.shape[0])
            self.edges_num.update_with_weight(edge_variance, label.shape[0])

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []

        self.model.eval()

        for data_in, pearson, label in dataloader:
            label = label.long()
            data_in, pearson, label = data_in.to(
                device), pearson.to(device), label.to(device)
            output, _, _ = self.model(data_in, pearson)

            loss = self.loss_fn(output, label)
            loss_meter.update_with_weight(
                loss.item(), label.shape[0])
            top1 = accuracy(output, label)[0]
            acc_meter.update_with_weight(top1, label.shape[0])
            result += F.softmax(output, dim=1)[:, 1].tolist()
            labels += label.tolist()

        auc = roc_auc_score(labels, result)
        result = np.array(result)
        result[result > 0.5] = 1
        result[result <= 0.5] = 0
        metric = precision_recall_fscore_support(
            labels, result, average='micro')
        return [auc] + list(metric)

    def generate_save_learnable_matrix(self):
        learable_matrixs = []

        labels = []

        for data_in, nodes, label in self.test_dataloader:
            label = label.long()
            data_in, nodes, label = data_in.to(
                device), nodes.to(device), label.to(device)
            _, learable_matrix, _ = self.model(data_in, nodes)

            learable_matrixs.append(learable_matrix.cpu().detach().numpy())
            labels += label.tolist()

        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"learnable_matrix.npy", {'matrix': np.vstack(
            learable_matrixs), "label": np.array(labels)}, allow_pickle=True)

    def save_result(self, results):
        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"training_process.npy",
                results, allow_pickle=True)

        torch.save(self.model.state_dict(), self.save_path/"model.pt")

    def train(self):
        training_process = []
        for epoch in range(self.epochs):
            self.reset_meters()
            self.train_per_epoch(self.optimizers[0])
            val_result = self.test_per_epoch(self.val_dataloader,
                                             self.val_loss, self.val_accuracy)

            test_result = self.test_per_epoch(self.test_dataloader,
                                              self.test_loss, self.test_accuracy)

            self.logger.info(" | ".join([
                f'Epoch[{epoch}/{self.epochs}]',
                f'Train Loss:{self.train_loss.avg: .3f}',
                f'Train Accuracy:{self.train_accuracy.avg: .3f}%',
                f'Edges:{self.edges_num.avg: .3f}',
                f'Test Loss:{self.test_loss.avg: .3f}',
                f'Test Accuracy:{self.test_accuracy.avg: .3f}%',
                f'Val AUC:{val_result[0]:.2f}',
                f'Test AUC:{test_result[0]:.2f}'
            ]))
            training_process.append([self.train_accuracy.avg, self.train_loss.avg,
                                     self.val_loss.avg, self.test_loss.avg]
                                    + val_result + test_result)
        if self.save_learnable_graph:
            self.generate_save_learnable_matrix()
        self.save_result(training_process)


class BiLevelTrain(BasicTrain):

    def __init__(self, train_config, model, optimizers, dataloaders, log_folder) -> None:
        super().__init__(train_config, model, optimizers, dataloaders, log_folder)

    def train(self):
        training_process = []
        matrix_epoch = 5

        for epoch in range(self.epochs):
            self.reset_meters()
            if epoch % 10 < matrix_epoch:
                self.train_per_epoch(self.optimizers[0])
            else:
                self.train_per_epoch(self.optimizers[1])
            val_result = self.test_per_epoch(self.val_dataloader,
                                             self.val_loss, self.val_accuracy)

            test_result = self.test_per_epoch(self.test_dataloader,
                                              self.test_loss, self.test_accuracy)

            self.logger.info(" | ".join([
                f'Epoch[{epoch}/{self.epochs}]',
                f'Train Loss:{self.train_loss.avg: .3f}',
                f'Train Accuracy:{self.train_accuracy.avg: .3f}%',
                f'Edges:{self.edges_num.avg: .3f}',
                f'Test Loss:{self.test_loss.avg: .3f}',
                f'Test Accuracy:{self.test_accuracy.avg: .3f}%',
                f'Val AUC:{val_result[0]:.2f}',
                f'Test AUC:{test_result[0]:.2f}'
            ]))
            training_process.append([self.train_accuracy.avg, self.train_loss.avg,
                                     self.val_loss.avg, self.test_loss.avg]
                                    + val_result + test_result)
        if self.save_learnable_graph:
            self.generate_save_learnable_matrix()
        self.save_result(training_process)


class GNNTrain(BasicTrain):

    def __init__(self, train_config, model, optimizers, dataloaders, log_folder) -> None:
        super().__init__(train_config, model, optimizers, dataloaders, log_folder)
        self.pure_gnn_graph = train_config['pure_gnn_graph']
        self.save_learnable_graph = False

    def train_per_epoch(self, optimizer):

        self.model.train()

        for _, pearson, label in self.train_dataloader:
            label = label.long()

            pearson, label = pearson.to(device), label.to(device)

            bz, module_num, _ = pearson.shape

            if self.pure_gnn_graph == "uniform":
                graph = torch.ones(
                    (bz, module_num, module_num)).float().to(device)
            elif self.pure_gnn_graph == "pearson":
                graph = torch.abs(pearson)

            graph, nodes, targets_a, targets_b, lam = mixup_data(
                graph, pearson, label, 1, device)

            output = self.model(graph, nodes)

            loss = mixup_criterion(
                self.loss_fn, output, targets_a, targets_b, lam)

            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            top1 = accuracy(output, label)[0]
            self.train_accuracy.update_with_weight(top1, label.shape[0])

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []

        self.model.eval()

        for _, pearson, label in dataloader:
            label = label.long()

            pearson, label = pearson.to(device), label.to(device)

            bz, module_num, _ = pearson.shape

            if self.pure_gnn_graph == "uniform":
                graph = torch.ones(
                    (bz, module_num, module_num)).float().to(device)
            elif self.pure_gnn_graph == "pearson":
                graph = torch.abs(pearson)

            output = self.model(graph, pearson)

            loss = self.loss_fn(output, label)
            loss_meter.update_with_weight(
                loss.item(), label.shape[0])
            top1 = accuracy(output, label)[0]
            acc_meter.update_with_weight(top1, label.shape[0])
            result += F.softmax(output, dim=1)[:, 1].tolist()
            labels += label.tolist()

        auc = roc_auc_score(labels, result)
        result = np.array(result)
        result[result > 0.5] = 1
        result[result <= 0.5] = 0
        metric = precision_recall_fscore_support(
            labels, result, average='micro')
        return [auc] + list(metric)


class SeqTrain(BasicTrain):
    def __init__(self, train_config, model, optimizers, dataloaders, log_folder) -> None:
        super().__init__(train_config, model, optimizers, dataloaders, log_folder)
        self.save_learnable_graph = False

    def train_per_epoch(self, optimizer):

        self.model.train()

        for seq_group, _, label in self.train_dataloader:
            label = label.long()

            seq_group, label = seq_group.to(device), label.to(device)

            seq_group, _, targets_a, targets_b, lam = mixup_data(
                seq_group, seq_group, label, 1, device)

            output = self.model(seq_group)

            loss = mixup_criterion(
                self.loss_fn, output, targets_a, targets_b, lam)

            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            top1 = accuracy(output, label)[0]
            self.train_accuracy.update_with_weight(top1, label.shape[0])

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []

        self.model.eval()

        for seq_group, _, label in dataloader:
            label = label.long()

            seq_group, label = seq_group.to(device), label.to(device)

            output = self.model(seq_group)

            loss = self.loss_fn(output, label)
            loss_meter.update_with_weight(
                loss.item(), label.shape[0])
            top1 = accuracy(output, label)[0]
            acc_meter.update_with_weight(top1, label.shape[0])
            result += F.softmax(output, dim=1)[:, 1].tolist()
            labels += label.tolist()

        auc = roc_auc_score(labels, result)
        result = np.array(result)
        result[result > 0.5] = 1
        result[result <= 0.5] = 0
        metric = precision_recall_fscore_support(
            labels, result, average='micro')
        return [auc] + list(metric)


class BrainCNNTrain(BasicTrain):

    def __init__(self, train_config, model, optimizers, dataloaders, log_folder) -> None:
        super().__init__(train_config, model, optimizers, dataloaders, log_folder)
        self.save_learnable_graph = False

    def train_per_epoch(self, optimizer):

        self.model.train()

        for _, pearson, label in self.train_dataloader:
            label = label.long()

            pearson, label = pearson.to(device), label.to(device)

            _, nodes, targets_a, targets_b, lam = mixup_data(
                pearson, pearson, label, 1, device)

            output = self.model(nodes)

            loss = mixup_criterion(
                self.loss_fn, output, targets_a, targets_b, lam)

            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            top1 = accuracy(output, label)[0]
            self.train_accuracy.update_with_weight(top1, label.shape[0])

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []

        self.model.eval()

        for _, pearson, label in dataloader:
            label = label.long()

            pearson, label = pearson.to(device), label.to(device)

            output = self.model(pearson)

            loss = self.loss_fn(output, label)
            loss_meter.update_with_weight(
                loss.item(), label.shape[0])
            top1 = accuracy(output, label)[0]
            acc_meter.update_with_weight(top1, label.shape[0])
            result += F.softmax(output, dim=1)[:, 1].tolist()
            labels += label.tolist()

        auc = roc_auc_score(labels, result)
        result = np.array(result)
        result[result > 0.5] = 1
        result[result <= 0.5] = 0
        metric = precision_recall_fscore_support(
            labels, result, average='micro')
        return [auc] + list(metric)


class FCNetTrain(BasicTrain):

    def __init__(self, train_config, model, optimizers, dataloaders, log_folder):
        super().__init__(train_config, model, optimizers, dataloaders, log_folder)
        self.generated_graph = []

    def train_per_epoch(self, optimizer):

        self.model.train()

        for seq_group, label in self.train_dataloader:
            label = label.long()

            seq_group, label = seq_group.to(device), label.to(device)

            output = self.model(seq_group)

            loss = self.loss_fn(output, label)

            self.train_loss.update_with_weight(loss.item(), label.shape[0])
 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    def test_per_epoch(self, dataloader, loss_meter, acc_meter, save_graph=False):

        self.model.eval()

        self.generated_graph = []

        for seq_group, label in dataloader:
            label = label.long()

            seq_group, label = seq_group.to(device), label.to(device)

            output = self.model(seq_group)

            loss = self.loss_fn(output, label)
         
            loss_meter.update_with_weight(
                loss.item(), label.shape[0])
  
        return None


    
    def train(self):
        training_process = []
        for epoch in range(self.epochs):
            self.reset_meters()
            self.train_per_epoch(self.optimizers[0])
            self.test_per_epoch(self.val_dataloader,
                                             self.val_loss, self.val_accuracy)

            self.test_per_epoch(self.test_dataloader,
                                              self.test_loss, self.test_accuracy, save_graph=True)

            self.logger.info(" | ".join([
                f'Epoch[{epoch}/{self.epochs}]',
                f'Train Loss:{self.train_loss.avg: .3f}',
                f'Train Accuracy:{self.train_accuracy.avg: .3f}%',
                f'Edges:{self.edges_num.avg: .3f}',
                f'Test Loss:{self.test_loss.avg: .3f}',
                f'Test Accuracy:{self.test_accuracy.avg: .3f}%'
            ]))
            training_process.append([self.train_accuracy.avg, self.train_loss.avg,
                                     self.val_loss.avg, self.test_loss.avg])
            
        self.save_result(training_process)

