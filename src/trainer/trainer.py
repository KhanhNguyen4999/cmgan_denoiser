from base.base_trainer import BaseTrainer
from torch.cuda.amp import autocast
import torch.nn.functional as F
import logging 
import time
from utils import *
from typing import Any
import numpy as np
import os
import tqdm
from evaluation import evaluation_model
from tools.compute_metrics import stoi
from augment import Remix
from models import discriminator


class Trainer(BaseTrainer):
    def __init__ (
                self,
                dist,
                rank,
                resume,
                n_gpus,
                epochs,
                batch_size,
                model,
                discriminator,
                train_ds,
                test_ds,
                scheduler_D,
                scheduler_G,
                optimizer,
                optimizer_disc,
                loss_weights,
                hop,
                n_fft,
                scaler,
                use_amp,
                interval_eval,
                max_clip_grad_norm,
                gradient_accumulation_steps,
                remix,
                save_model_dir,
                data_test_dir,
                tsb_writer,
                num_prints,
                logger
            ):

        super(Trainer, self).__init__(
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
                            )

        self.model_discriminator = discriminator                            
        self.optimizer = optimizer
        self.optimizer_disc = optimizer_disc
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        self.scheduler_D = scheduler_D
        self.scheduler_G = scheduler_G

        self.loss_weights = loss_weights
        self.tsb_writer = tsb_writer
        self.n_gpus = n_gpus

        self.n_fft = n_fft
        self.hop = hop
        self.scaler = scaler

        self.best_loss = float('inf')
        self.best_state = None
        self.epoch_start = 0
        self.save_enhanced_dir = self.save_model_dir + "/enhanced_sample"
        self.data_test_dir = data_test_dir
        self.num_prints = num_prints
        self.logger = logger
        
        # data augment
        augments = []
        if remix:
            augments.append(Remix())
        self.augment = torch.nn.Sequential(*augments)

        if not os.path.exists(self.save_enhanced_dir):
            os.makedirs(self.save_enhanced_dir)

        if self.resume:
            self._reset()
            

    def gather(self, value: torch.tensor) -> Any:
        # gather value across devices - https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather
        if value.ndim == 0:
            value = value.clone()[None]

        output_tensors = [value.clone() for _ in range(self.dist.get_world_size())]
        self.dist.all_gather(output_tensors, value)
        return torch.cat(output_tensors, dim=0)

    def _serialize(self, epoch):
        '''
        function help to save new general checkpoint
        '''
        package = {}
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            package["model"] = self.model.module.state_dict()
            package['discriminator'] = self.model_discriminator.module.state_dict()
        else:
            package["model"] = self.model.state_dict()
            package['discriminator'] = self.model_discriminator.state_dict()
        
        package['optimizer'] = self.optimizer.state_dict()
        package['optimizer_disc'] = self.optimizer_disc.state_dict()
        package['best_state'] = self.best_state
        package['loss'] = self.best_loss
        package['epoch'] = epoch
        package['scaler'] = self.scaler
        tmp_path = os.path.join(self.save_model_dir, "checkpoint.tar")
        torch.save(package, tmp_path)

        # Save only the best model, pay attention that don't need to save best discriminator
        # because when infer time, you only need model to predict, and if better the discriminator
        # the worse the model ((:
        model = package['best_state']
        tmp_path = os.path.join(self.save_model_dir, "best.th")
        torch.save(model, tmp_path)

    def _reset(self):
        # self.dist.barrier()
        if os.path.exists(self.save_model_dir) and os.path.isfile(self.save_model_dir + "/checkpoint.tar"):
            if self.rank == 0:
                self.logger.info("<<<<<<<<<<<<<<<<<< Load pretrain >>>>>>>>>>>>>>>>>>")
                self.logger.info("Loading last state for resuming training")

            map_location='cuda:{}'.format(self.rank)
            map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
            package = torch.load(self.save_model_dir + "/checkpoint.tar", map_location = map_location)
            
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                self.model.module.load_state_dict(package['model'])
                self.model_discriminator.module.load_state_dict(package['discriminator'])
            else:
                self.model.load_state_dict(package['model'])
                self.model_discriminator.load_state_dict(package['discriminator'])
                
            self.optimizer.load_state_dict(package['optimizer'])
            self.optimizer_disc.load_state_dict(package['optimizer_disc'])
            self.epoch_start = package['epoch'] + 1
            self.best_loss = package['loss']
            self.best_state = package['best_state']
            self.scaler = package['scaler']
            if self.rank == 0:
                self.logger.info(f"Model checkpoint loaded. Training will begin at {self.epoch_start} epoch.")
                self.logger.info(f"Load pretrained info: ")
                self.logger.info(f"Best loss: {self.best_loss}")


    def _train_step(self, batch):
        clean = batch[0].cuda()
        noisy = batch[1].cuda()
        one_labels = torch.ones(min(self.batch_size, clean.size(0))).cuda()

        # Normalization
        c = torch.sqrt(noisy.size(-1) / torch.sum((noisy ** 2.0), dim=-1))
        noisy, clean = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1)
        noisy, clean = torch.transpose(noisy * c, 0, 1), torch.transpose(clean * c, 0, 1)

        if len(self.augment) > 0:
            sources = torch.stack([noisy - clean, clean])
            sources = self.augment(sources)
            noise, clean = sources
            noisy = noise + clean

        noisy_spec = torch.stft(noisy, self.n_fft, self.hop, window=torch.hamming_window(self.n_fft).cuda(),
                                onesided=True)
        clean_spec = torch.stft(clean, self.n_fft, self.hop, window=torch.hamming_window(self.n_fft).cuda(),
                                onesided=True)
        noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
        clean_spec = power_compress(clean_spec)
        clean_real = clean_spec[:, 0, :, :].unsqueeze(1)
        clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)

        # Runs the forward pass under autocast.
        with autocast(enabled = self.use_amp):
            est_real, est_imag = self.model(noisy_spec)
            if self.use_amp:
                # output is float16 because linear layers autocast to float16.
                # assert est_real.dtype is torch.float16, f"est_real's dtype is not torch.float16 but {est_real.dtype}"
                # assert est_imag.dtype is torch.float16, f"est_imag's dtype is not torch.float16 but {est_imag.dtype}"
                None

            est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
            est_mag = torch.sqrt(est_real**2 + est_imag**2)
            clean_mag = torch.sqrt(clean_real**2 + clean_imag**2)

            predict_fake_metric = self.model_discriminator(clean_mag, est_mag)
            gen_loss_GAN = (predict_fake_metric.flatten() - one_labels.float())**2

            loss_mag = torch.mean(((est_mag - clean_mag)**2).reshape(est_mag.shape[0], -1), 1)
            loss_ri = torch.mean(((est_real - clean_real)**2).reshape(est_mag.shape[0], -1), 1) + \
                        torch.mean(((est_imag - clean_imag)**2).reshape(est_mag.shape[0], -1), 1)

            est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
            est_audio = torch.istft(est_spec_uncompress, self.n_fft, self.hop,
                                    window=torch.hamming_window(self.n_fft).cuda(), onesided=True)

            time_loss = torch.mean(torch.abs(est_audio - clean))
            length = est_audio.size(-1)
        
        est_audio_list = list(est_audio.detach().cpu().numpy())
        clean_audio_list = list(clean.cpu().numpy()[:, :length])
        pesq_score = discriminator.batch_pesq(clean_audio_list, est_audio_list, self.n_gpus)
        pesq_score_weight = torch.nn.functional.softmax(1 - pesq_score)

        loss_ri = (loss_ri * pesq_score_weight).sum()
        loss_mag = (loss_mag * pesq_score_weight).sum()
        gen_loss_GAN = (gen_loss_GAN * pesq_score_weight).sum()

        loss = self.loss_weights[0] * loss_ri + \
                    self.loss_weights[1] * loss_mag + \
                    self.loss_weights[2] * time_loss + \
                    self.loss_weights[3] * gen_loss_GAN
        

        # loss is float32 because mse_loss layers autocast to float32.
        assert loss.dtype is torch.float32, f"loss's dtype is not torch.float32 but {loss.dtype}"

        self.scaler.scale(loss).backward(retain_graph=True)

        # The calculation of PESQ can be None due to silent part
        if pesq_score is not None:
            predict_enhance_metric = self.model_discriminator(clean_mag, est_mag.detach())
            (clean_mag, est_mag.detach())
            predict_max_metric = self.model_discriminator(clean_mag, clean_mag)
            discrim_loss_metric = F.mse_loss(predict_max_metric.flatten(), one_labels) + \
                                  F.mse_loss(predict_enhance_metric.flatten(), pesq_score)
            discrim_loss_metric.backward()
        else:
            discrim_loss_metric = torch.tensor([0.], requires_grad=True).cuda()

        # Logging
        # average over devices in ddp
        if self.n_gpus > 1:
            loss = self.gather(loss).mean()
            discrim_loss_metric = self.gather(discrim_loss_metric).mean()

        return loss.item(), discrim_loss_metric.item()


    def _train_epoch(self, epoch) -> None:
        gen_loss_train, disc_loss_train = [], []

        self.logger.info('\n <Epoch>: {} -- Start training '.format(epoch))
        name = f"Train | Epoch {epoch}"
        logprog = LogProgress(self.logger, self.train_ds, updates=self.num_prints, name=name)

        for idx, batch in enumerate(logprog):
            loss, disc_loss = self._train_step(batch)
            gen_loss_train.append(loss )
            disc_loss_train.append(disc_loss)
            if self.rank  == 0:
                logprog.update(gen_loss=format(loss, ".5f"), disc_loss=format(disc_loss, ".5f"))
            
            # Optimize step
            if (idx + 1) % self.gradient_accumulation_steps == 0 or idx == len(self.train_ds) - 1:
                
                #gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_clip_grad_norm)
                self.scaler.step(self.optimizer)
                self.optimizer.zero_grad()

                self.scaler.update()

                self.optimizer_disc.step()
                self.optimizer_disc.zero_grad()
                

        gen_loss_train = np.mean(gen_loss_train)
        disc_loss_train = np.mean(disc_loss_train)

        template = 'Generator loss: {}, Discriminator loss: {}'
        info = template.format(gen_loss_train, disc_loss_train)
        if self.rank == 0:
            # print("Done epoch {} - {}".format(epoch, info))
            self.logger.info('-' * 70)
            self.logger.info(bold(f"     Epoch {epoch} - Overall Summary Training | {info}"))
            self.tsb_writer.add_scalar("Loss_gen/train", gen_loss_train, epoch)
            self.tsb_writer.add_scalar("Loss_disc/train", disc_loss_train, epoch)

        gen_loss_valid, disc_loss_valid = self._valid_epoch(epoch)

         # save best checkpoint
        if self.rank == 0:
            template = 'Generator loss: {}, Discriminator loss: {}'
            info = template.format(gen_loss_valid, disc_loss_valid)
            self.logger.info(bold(f"             - Overall Summary Validation | {info}"))
            self.logger.info('-' * 70)

            self.tsb_writer.add_scalar("Loss_gen/valid", gen_loss_valid, epoch)
            self.tsb_writer.add_scalar("Loss_disc/valid", disc_loss_valid, epoch)

            self.best_loss = min(self.best_loss, gen_loss_valid)
            if gen_loss_valid == self.best_loss:
                self.best_state = copy_state(self.model.state_dict())

            if epoch % self.interval_eval == 0:
                metrics_avg = evaluation_model(self.model, 
                                self.data_test_dir + "/noisy", 
                                self.data_test_dir + "/clean",
                                True, 
                                self.save_enhanced_dir)

                for metric_type, value in metrics_avg.items():
                    self.tsb_writer.add_scalar(f"metric/{metric_type}", value, epoch)

                info = " | ".join(f"{k.capitalize()} {v:.5f}" for k, v in metrics_avg.items())
                # print("Evaluation epoch {} -- {}".format(epoch, info))
                self.logger.info(bold(f"     Evaluation Summary:  | Epoch {epoch} | {info}"))

            # Save checkpoint
            self._serialize(epoch)

        self.dist.barrier() # see https://stackoverflow.com/questions/59760328/how-does-torch-distributed-barrier-work
        self.scheduler_G.step()
        self.scheduler_D.step()


    @torch.no_grad()
    def test_step(self, batch):
        clean = batch[0].cuda()
        noisy = batch[1].cuda()
        one_labels = torch.ones(clean.size(0)).cuda()

        # Normalization
        c = torch.sqrt(noisy.size(-1) / torch.sum((noisy ** 2.0), dim=-1))
        noisy, clean = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1)
        noisy, clean = torch.transpose(noisy * c, 0, 1), torch.transpose(clean * c, 0, 1)

        noisy_spec = torch.stft(noisy, self.n_fft, self.hop, window=torch.hamming_window(self.n_fft).cuda(),
                                onesided=True)
        clean_spec = torch.stft(clean, self.n_fft, self.hop, window=torch.hamming_window(self.n_fft).cuda(),
                                onesided=True)

        noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
        clean_spec = power_compress(clean_spec)
        clean_real = clean_spec[:, 0, :, :].unsqueeze(1)
        clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)

        est_real, est_imag = self.model(noisy_spec)
        est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
        est_mag = torch.sqrt(est_real ** 2 + est_imag ** 2)
        clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2)

        predict_fake_metric = self.model_discriminator(clean_mag, est_mag)
        gen_loss_GAN = (predict_fake_metric.flatten() - one_labels.float())**2

        loss_mag = torch.mean(((est_mag - clean_mag)**2).reshape(est_mag.shape[0], -1), 1)
        loss_ri = torch.mean(((est_real - clean_real)**2).reshape(est_mag.shape[0], -1), 1) + \
                    torch.mean(((est_imag - clean_imag)**2).reshape(est_mag.shape[0], -1), 1)

        est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
        est_audio = torch.istft(est_spec_uncompress, self.n_fft, self.hop,
                                window=torch.hamming_window(self.n_fft).cuda(), onesided=True)

        time_loss = torch.mean(torch.abs(est_audio - clean))
        length = est_audio.size(-1)
        
        est_audio_list = list(est_audio.detach().cpu().numpy())
        clean_audio_list = list(clean.cpu().numpy()[:, :length])
        pesq_score = discriminator.batch_pesq(clean_audio_list, est_audio_list, self.n_gpus)
        pesq_score_weight = torch.nn.functional.softmax(1 - pesq_score)

        loss_ri = (loss_ri * pesq_score_weight).sum()
        loss_mag = (loss_mag * pesq_score_weight).sum()
        gen_loss_GAN = (gen_loss_GAN * pesq_score_weight).sum()

        loss = self.loss_weights[0] * loss_ri + \
                    self.loss_weights[1] * loss_mag + \
                    self.loss_weights[2] * time_loss + \
                    self.loss_weights[3] * gen_loss_GAN

        if pesq_score is not None:
            predict_enhance_metric = self.model_discriminator(clean_mag, est_mag.detach())
            predict_max_metric = self.model_discriminator(clean_mag, clean_mag)
            discrim_loss_metric = F.mse_loss(predict_max_metric.flatten(), one_labels) + \
                                  F.mse_loss(predict_enhance_metric.flatten(), pesq_score)
        else:
            discrim_loss_metric = torch.tensor([0.]).cuda()
        
        # Logging
        # average over devices in ddp
        if self.n_gpus > 1:
            loss = self.gather(loss).mean()
            discrim_loss_metric = self.gather(discrim_loss_metric).mean()

        return loss.item(), discrim_loss_metric.item()

    def _valid_epoch(self, epoch):

        self.model.eval()
        self.model_discriminator.eval()

        gen_loss_total = 0.
        disc_loss_total = 0.
        for idx, batch in enumerate(self.test_ds):
            step = idx + 1
            loss, disc_loss = self.test_step(batch)
            gen_loss_total += loss
            disc_loss_total += disc_loss

        gen_loss_avg = gen_loss_total / step
        disc_loss_avg = disc_loss_total / step

        return gen_loss_avg, disc_loss_avg


    