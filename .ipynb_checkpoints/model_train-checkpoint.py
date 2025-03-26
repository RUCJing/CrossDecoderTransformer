# -*- coding: utf-8 -*-
import torch
from tqdm import tqdm
import time
from torch.optim import RMSprop, Adam, AdamW
from transformers import get_cosine_schedule_with_warmup
from torch.nn import CrossEntropyLoss, Softmax
from torch.utils.data import DataLoader
from numpy import vstack, argmax
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from model import CrossDecoderTransformer
from text_featuring import CSVDataset
from params import TRAIN_BATCH_SIZE, TEST_BATCH_SIZE, LEARNING_RATE, EPOCHS, TRAIN_FILE_PATH, TEST_FILE_PATH, MODEL_FOLDER, START_NO

def print_log(epoch, train_time, train_loss, train_acc, val_loss, val_acc):
    print(f"Epoch [{epoch}], time: {train_time:.2f}s, "
          f"loss: {train_loss:.4f}, acc: {train_acc:.4f}, "
          f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")
    
def accuracy(outputs, labels):
    preds = torch.max(outputs, dim=1)[1]
    return (preds == labels).sum().item() / len(labels)

# model train
class ModelTrainer(object):
    def __init__(self):
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        model = CrossDecoderTransformer(nhead=10,             
                           dim_feedforward=600,  
                           num_layers=1,
                           att_dropout = 0.2,
                           ffn_dropout = 0.2,
                           num_class = 2)
        self.model = model.to(self.device)
        num_params = sum(param.numel() for param in model.parameters())
        print(num_params)
        
    def train_step(self, model, trainloader, optimizer, epoch, scheduler):
        model.train()
        train_loss, train_acc, idx = 0, 0, 0
        for texts, X_cat, X_num, labels, speakers in tqdm(trainloader):
            idx += 1
            texts, X_cat, X_num, labels, speakers = texts.to(self.device), X_cat.to(self.device), X_num.to(self.device), labels.to(self.device), speakers.to(self.device)
            optimizer.zero_grad()
            outputs = model(texts, X_cat, X_num, speakers)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            if idx % 10 == 0: print(f'epoch: {epoch}, loss:{train_loss / idx}')
            train_acc += accuracy(outputs, labels)
        train_loss /= len(trainloader)
        train_acc /= len(trainloader)
        return train_loss, train_acc

    # 定义验证过程
    def validate_step(self, model, testloader, epoch):
        model.eval()
        val_loss, val_acc = 0, 0
        with torch.no_grad():
            for texts, X_cat, X_num, labels, speakers in testloader:
                texts, X_cat, X_num, labels, speakers = texts.to(self.device), X_cat.to(self.device), X_num.to(self.device), labels.to(self.device), speakers.to(self.device)
                outputs = model(texts, X_cat, X_num, speakers)
                loss = F.cross_entropy(outputs, labels)
                val_loss += loss.item()
                val_acc += accuracy(outputs, labels)
        val_loss /= len(testloader)
        val_acc /= len(testloader)
        torch.save(model, MODEL_FOLDER + f'model_{epoch}.pth')
        return val_loss, val_acc

    # Model Training, evaluation and metrics calculation
    def train(self):
        model = self.model
        # calculate split
        train, test = CSVDataset(TRAIN_FILE_PATH), CSVDataset(TEST_FILE_PATH)
        # prepare data loaders
        train_dl = DataLoader(train, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        test_dl = DataLoader(test, batch_size=TEST_BATCH_SIZE)

        # Define optimizer
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
        len_dataset, warm_up_ratio = len(train), 0.1
        total_steps = (len_dataset // TRAIN_BATCH_SIZE) * EPOCHS
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = warm_up_ratio * total_steps, num_training_steps = total_steps)
        # Starts training phase
        train_losses, train_accs, val_losses, val_accs = [], [], [], []
        for epoch in range(EPOCHS):
            start_time = time.time()
            train_loss, train_acc = self.train_step(model, train_dl, optimizer, epoch, scheduler)
            val_loss, val_acc = self.validate_step(model, test_dl, epoch)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            print_log(epoch + 1, time.time() - start_time, train_loss, train_acc, val_loss, val_acc)
        print(f'train_accs:{train_accs}')
        print(f'val_accs:{val_accs}')


if __name__ == '__main__':
    trainer = ModelTrainer()
    trainer.train()
