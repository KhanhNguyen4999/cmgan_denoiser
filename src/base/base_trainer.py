# pytorch dependeces
import torch
import torch.distributed as dist

# 
import os

class BaseTrainer:

    def __init__(
            self,
            dist,
            rank,
            resume,
            model,
            train_ds,
            test_ds,
            epochs,
            use_amp,
            interval_eval,
            max_clip_grad_norm,
            save_model_dir
        ):
        self.dist = dist
        self.rank = rank
        self.resume = resume
        self.model = model
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.epochs = epochs
        self.use_amp = use_amp
        self.interval_eval = interval_eval
        self.max_clip_grad_norm = max_clip_grad_norm
        self.save_model_dir = save_model_dir

    def _count_trainable_parameters(self) -> None:
        print("Number of trainable params: ", sum(p.numel() for p in self.model.parameters() if p.requires_grad)/1e6)
    
    def _count_parameters(self) -> None:
        params_of_network =0
        for param in self.model.parameters():
            params_of_network += param.numel()
        print(f"The amount of parameters in the project is {params_of_network / 1e6} million.")

    def _reset(self):
        '''
        function help to reload the pretrained model
        '''
        raise NotImplementedError

    def _serialize(self, epoch) -> None:
        '''
        function help to save new general checkpoint
        '''
        raise NotImplementedError 


    def _train_epoch(self, epoch) -> None:
        raise NotImplementedError

    def _valid_epoch(self, epoch) -> None:
        raise NotImplementedError

    def train(self) -> None:
        for epoch in range(self.epoch_start, self.epochs):
            self.model.train()
            self.model_discriminator.train()
            self._train_epoch(epoch)
        
        if self.rank == 0:
            print("Training Process Done")
            