# PyTorch dependencies
import torch
from torch import cuda
from transformers import BertTokenizer

class NeuralNetwork():
    def __init__(self):
        # Very crucial parameter. But, due to limited GPU mememory can't push this value further upto 512!!!!!
        self.max_len = 300   
        self.train_batch_size = 32
        self.valid_batch_size = 24  # Default can be set to 8 (safe with basic GPU and low RAM)
        self.epochs = 1
        self.learning_rate = 1e-05
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.device = 'CUDA' if cuda.is_available() else 'CPU'
        self.train_params = {'batch_size': self.train_batch_size, 'shuffle': True, 'num_workers': 0}
        self.test_params = {'batch_size': self.valid_batch_size, 'shuffle': False, 'num_workers': 0}
        self.validation_params = {'batch_size': self.valid_batch_size,'shuffle': False,'num_workers': 0} 

    def loss_fn(self, outputs, targets):
        return torch.nn.BCEWithLogitsLoss()(outputs, targets)

    def validation(self, validation_loader, model):
        model.eval()
        val_epoch_loss = []
        with torch.no_grad():
            for _,data in enumerate(validation_loader, 0):
                ids = data['ids'].to(self.device, dtype = torch.long)
                mask = data['mask'].to(self.device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(self.device, dtype = torch.long)
                targets = data['targets'].to(self.device, dtype = torch.float)

                outputs = model(ids, mask, token_type_ids)
                #optimizer.zero_grad()

                loss = self.loss_fn(outputs, targets)
                val_epoch_loss.append(loss.item())

        # print("\n-------------------------------------------------------------------")
        return val_epoch_loss

    def train(self, epoch, training_loader, validation_loader, model, optimizer):
        epoch_loss = []
        validation_epoch_loss = []
        #print("Training the model:\n")
        for _,data in enumerate(training_loader, 0):
            model.train()
            ids = data['ids'].to(self.device, dtype = torch.long)
            mask = data['mask'].to(self.device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(self.device, dtype = torch.long)
            targets = data['targets'].to(self.device, dtype = torch.float)

            outputs = model(ids, mask, token_type_ids)

            optimizer.zero_grad()
            loss = self.loss_fn(outputs, targets)

            loss.backward()
            optimizer.step()

            # Performing validation with each step/epoch of training
            val_epoch_loss = self.validation(validation_loader, model)
            validation_loss = sum(val_epoch_loss)/len(val_epoch_loss)

            if _ % 150 == 0:
                print(f'\nEpoch: {epoch}, Step: {_}, Training Loss:  {loss.item()}, Validation Loss:  {validation_loss}')
        
            epoch_loss.append(loss.item())
            validation_epoch_loss.append(validation_loss)

        print("\nTraining Complete!")
        return epoch_loss, validation_epoch_loss

    def testing(self, testing_loader, model):
        model.eval()
        fin_targets=[]
        fin_outputs=[]
        with torch.no_grad():
            for _, data in enumerate(testing_loader, 0):
                ids = data['ids'].to(self.device, dtype = torch.long)
                mask = data['mask'].to(self.device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(self.device, dtype = torch.long)
                targets = data['targets'].to(self.device, dtype = torch.float)
                outputs = model(ids, mask, token_type_ids)
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        return fin_outputs, fin_targets