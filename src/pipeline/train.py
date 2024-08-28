import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from src.constants.model import Model
from src.constants.dataloader import DataLoader
from src.config.config import TrainingConfig
import uuid


training_config = TrainingConfig()

lr = training_config.learning_rate
batch_size = training_config.batch_size
epochs = training_config.epochs
device = training_config.device
weight_decay = training_config.weight_decay
momentum = training_config.momentum
betas = training_config.betas


model = Model()
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, betas=betas)
criterion = nn.CrossEntropyLoss(reduction='mean')
writer = SummaryWriter(f'./artifacts/runs/experiment_{uuid.uuid4()}')


data_loader = DataLoader()
train_loader, val_loader = data_loader.load(batch_size=batch_size)

class Trainer():
    
    def __init__(self,
                 epochs: int,
                 batch_size: int,
                 model: nn.Module,
                 optimizer: optim.Optimizer,
                 criterion: nn.Module,
                 writer: SummaryWriter,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: str):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.writer = writer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

    def train_one_epoch(self, epoch: int):
        self.model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(device), target.to(device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch_idx % 100 == 99:
                print(f'Epoch [{epoch+1}], Step [{batch_idx+1}/{len(self.train_loader)}], Loss: {loss.item():.4f}')
                writer.add_scalar('training_loss', running_loss / 100, epoch * len(self.train_loader) + batch_idx)
                running_loss = 0.0

    def validate(self, model):
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(self.val_loader.dataset)
        val_accuracy = 100. * correct / len(self.val_loader.dataset)
        print(f'Validation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(self.    val_loader.dataset)} ({val_accuracy:.2f}%)\n')
        return val_loss, val_accuracy
    
    def train(self):
        for epoch in range(self.epochs):
            self.train_one_epoch(epoch)
            val_loss, val_accuracy = self.validate(self.model)
            writer.add_scalar('validation_loss', val_loss, epoch)
            writer.add_scalar('validation_accuracy', val_accuracy, epoch)


if __name__ == '__main__':
    trainer = Trainer(epochs, batch_size, model, optimizer, criterion, writer, train_loader, val_loader, device)
    trainer.train()