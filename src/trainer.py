import os
import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter

class Trainer():
    def __init__(self, 
        model, 
        optimizer, 
        train_dataloader,
        outputs_dir, 
        num_epochs,
        device,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.outputs_dir = outputs_dir
        self.num_epochs = num_epochs
        self.device = device
        self.writer = SummaryWriter(outputs_dir)
    
    def train(self):
        model = self.model
        optimizer = self.optimizer
        train_dataloader = self.train_dataloader
        total_loss = 0

        for epoch in tqdm(range(self.num_epochs), total=self.num_epochs):
            epoch_loss = 0
            for idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=False, desc=f"Epoch {epoch + 1}"):
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                
                loss = model(inputs)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                total_loss += loss.item()

                global_step = epoch * len(train_dataloader) + idx + 1
                avg_loss = total_loss / global_step
                self.writer.add_scalar("Train-Step-Loss", avg_loss, global_step=global_step)
            
            epoch_loss /= len(train_dataloader)
            self.writer.add_scalar("Train-Epoch-Loss", epoch_loss, global_step=epoch)
            for name, params in model.named_parameters():
                self.writer.add_histogram(name, params, global_step=epoch)

            save_name = f"model_{epoch}.pth"
            save_path = os.path.join(self.outputs_dir, save_name)
            torch.save(model.state_dict(), save_path)


            
            


