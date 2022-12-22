import os
import matplotlib.pyplot as plt
import csv
import mlflow
import logging

import torch
from torch.utils.data import DataLoader

class Logger():
    
    def __init__(self, log_path):
        self.log_path = log_path
        self.train_acc = []; self.train_loss = []
        self.test_acc = [];  self.test_binary_acc = []
        self.precision = []; self.recall = []
        self.f_score = []
        self.rd = 0

    def get_hist(self, trainset, classes):
        self.train_points = len(trainset)
        self.classes = classes
        hist = [0]*len(classes)
        os.makedirs(os.path.join(self.log_path, 'hist'), exist_ok=True)
        train_loader = DataLoader(trainset, batch_size=1, shuffle=True, pin_memory=True)
        for _, (_, y, _) in enumerate(train_loader):
            for i in y:
                hist[i] += 1
        fig = plt.figure()
        plt.bar(classes, hist, width=0.9)
        plt.xlabel('classes')
        plt.ylabel('number of queried data')
        fig.savefig(os.path.join(self.log_path, 'hist', str(self.rd)+'.png'))
        plt.clf()
        plt.close()
    
    def save_weight(self, clf):
        if 'module' in [i for i,_ in clf.named_parameters()][0].split('.'):
            torch.save(clf.module.state_dict(), self.log_path+'/weight'+str(self.rd)+'.pth')
        else:
            torch.save(clf.state_dict(), self.log_path+'/weight'+str(self.rd)+'.pth')
    
    def write_train_log(self):
        for i in range(len(self.train_acc)):
            mlflow.log_metric('acc', self.train_acc[i])
            mlflow.log_metric('loss', self.train_loss[i])
            logging.info('Epoch:' + str(i) + '- training accuracy:'+str(self.train_acc[i])+'- training loss:'+str(self.train_loss[i]))
        self.train_acc = []; self.train_loss = []

    def write_test_log(self, class_correct, class_total, binary_correct, binary_total):
        self.test_acc.append(sum(class_correct)/sum(class_total))
        if sum(binary_total) > 1:
            self.test_binary_acc.append(sum(binary_correct)/sum(binary_total))
            binary_classes = ('NG' 'OK')
            for i in range(2):
                mlflow.log_metric(binary_classes[i], (binary_correct[i]/binary_total[i]), step=self.train_points)
                logging.info(f'{binary_classes[i]} class accuracy: {binary_correct[i]/binary_total[i]}')
            true_positive = binary_correct[0]/binary_total[0]; false_negative = 1-true_positive
            true_negative = binary_correct[1]/binary_total[1]; false_positive = 1-true_negative
            self.precision.append(true_positive/(true_positive+false_positive))
            self.recall.append(true_positive/(true_positive+false_negative))
            self.f_score.append(2*(self.precision[-1]*self.recall[-1])/(self.precision[-1]+self.recall[-1]))
            mlflow.log_metric('precision', self.precision[-1], step=self.train_points)
            mlflow.log_metric('recall', self.recall[-1], step=self.train_points)
            mlflow.log_metric('f_score', self.f_score[-1], step=self.train_points)
        else: self.test_binary_acc.append(0)
            
        mlflow.log_metric('test_acc', self.test_acc[-1], step=self.train_points)
        mlflow.log_metric('binary_test_acc', self.test_binary_acc[-1], step=self.train_points)
        logging.info(f'training points: {self.train_points}')
        logging.info(f'test accuracy: {round(self.test_acc[-1]*100, 2)}')
        logging.info(f'binary test accuracy: {round(self.test_binary_acc[-1]*100, 2)}')
        for i in range(len(self.classes)):
            mlflow.log_metric(self.classes[i], (class_correct[i]/class_total[i]), step=self.train_points)
            logging.info(f'{self.classes[i]} class accuracy: {100*class_correct[i]/class_total[i]}')

    def show_result(self, seed):
        with open(self.log_path+'/../../test.csv', mode='a') as f:
            writer = csv.writer(f)
            writer.writerow([seed])
            writer.writerow(self.test_acc)
            f.close()
        if len(self.f_score)>1:
            with open(self.log_path+'/../../f_score.csv', mode='a') as f:
                writer = csv.writer(f)
                writer.writerow([seed])
                writer.writerow(self.f_score)
                f.close()
            with open(self.log_path+'/../../precision.csv', mode='a') as f:
                writer = csv.writer(f)
                writer.writerow([seed])
                writer.writerow(self.precision)
                f.close()
            with open(self.log_path+'/../../recall.csv', mode='a') as f:
                writer = csv.writer(f)
                writer.writerow([seed])
                writer.writerow(self.recall)
                f.close()
            with open(self.log_path+'/../../binary_result.csv', mode='a') as f:
                writer = csv.writer(f)
                writer.writerow([seed])
                writer.writerow(self.test_binary_acc)
                f.close()