from typing import Any
import torch
import numpy as np
import abc
import lightning as L
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error
from sklearn.linear_model import LinearRegression, LogisticRegression
from skorch import NeuralNetClassifier, NeuralNetRegressor


class LinearRegressor(torch.nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super(LinearRegressor, self).__init__()
        assert out_dim == 1, 'Output dim should be 1 for regression, use LogisticRegressor if tast is classification'
        self.linear = torch.nn.Linear(in_dim, out_dim)
        
    def forward(self, x):
        output = self.linear(x).flatten()
        return output
  
    
class MLPRegressor(torch.nn.Module):   
    def __init__(self, in_dim, out_dim, hidden_dim) -> None:
        super(MLPRegressor, self).__init__()
        assert out_dim == 1, 'Output dim should be 1 for regression, use MLPClassifier if tast is classification'
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(hidden_dim, out_dim)
            )
        
    def forward(self, x):
        output = self.mlp(x).flatten()
        return output
    

class LogisticRegressor(torch.nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super(LogisticRegressor, self).__init__()
        self.logits = torch.nn.Sequential(
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.Sigmoid()
            )
        
    def forward(self, x):
        output = self.logits(x)
        return output
    

class MLPClassifier(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim) -> None:
        super(MLPClassifier, self).__init__()
        self.logits = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(hidden_dim, out_dim)
            )
        
    def forward(self, x):
        output = self.logits(x)
        return output


class LitRegressor(L.LightningModule):
    
    def __init__(self, model_class, in_dim, out_dim, hidden_dim=None, loss_func=torch.nn.MSELoss) -> None:
        super().__init__()
        assert out_dim == 1, 'For regressor out_dim must be 1, use LitClassifier for output_dim > 1.'
        self.loss_func = loss_func()
        self.regressor = model_class
        #self.save_hyperparameters(model_class, in_dim, out_dim)

    
    def training_step(self, batch, batch_idx):
        
        x, y = batch
        outputs = self.regressor(x)
        loss = self.loss_func(outputs, y)
        
        self.log("train_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        x = x.view(x.size(0), -1)
        outputs = self.regresspr(x)
        test_loss = self.loss_func(outputs, y)
        MSE = mean_squared_error(y, outputs, squared=True)
        values = {"test_loss": test_loss, "test_f1": MSE}
        self.log(values, prog_bar=True)
        self.log_dict(values)
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        x = x.view(x.size(0), -1)
        outputs = self.regressor(x)
        val_loss = self.loss_func(outputs, y)
        self.log("val_loss", val_loss)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        return self(x)
    
    def forward(self, x):
        return self.regressor(x) 
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        return optimizer
    

class LitClassifier(L.LightningModule):
    
    def __init__(self, model_class, in_dim, out_dim, hidden_dim=None, loss_func=torch.nn.CrossEntropyLoss) -> None:
        super().__init__()
        self.loss_func = loss_func()
        self.classifier = model_class
        #self.save_hyperparameters(model_class, in_dim, out_dim)
        
    def training_step(self, batch, batch_idx):
        
        x, y = batch
        x = x.view(x.size(0), -1)
        outputs = self.classifier(x)
        loss = self.loss_func(outputs, y)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        x = x.view(x.size(0), -1)
        outputs = self.classifier(x)
        test_loss = self.loss_func(outputs, y)
        #f1 = f1_score(y, outputs)
        #accuracy = accuracy_score(y, outputs)
        values = {"test_loss": test_loss}#, "test_f1": f1, "acc": accuracy}
        self.log("test_loss", test_loss, prog_bar=True)
        #self.log_dict(values)
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        x = x.view(x.size(0), -1)
        outputs = self.classifier(x)
        val_loss = self.loss_func(outputs, y)
        self.log("val_loss", val_loss)
        
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        return self(x)
    
    def forward(self, x):
        return self.classifier(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        return optimizer 
    

