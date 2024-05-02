import torch
import torch.nn as nn
import torch.nn.functional as F

class KDLoss(nn.Module):
    def __init__(self, weight_ri, weight_mag, weight_time, weight_gan):
        super(KDLoss, self).__init__()
        self.weight_ri = weight_ri
        self.weight_mag = weight_mag
        self.weight_time = weight_time
        self.weight_gan = weight_gan

    def forward(self, student_generator_outputs, teacher_generator_outputs):
        
        loss_mag = F.mse_loss(
            student_generator_outputs["est_mag"], teacher_generator_outputs["est_mag"]
        )
        loss_ri = F.mse_loss(
            student_generator_outputs["est_real"], teacher_generator_outputs["est_real"]
        ) + F.mse_loss(student_generator_outputs["est_imag"], teacher_generator_outputs["est_imag"])

        time_loss = torch.mean(
            torch.abs(student_generator_outputs["est_audio"] - teacher_generator_outputs["est_audio"])
        )

        loss = (
            self.weight_ri * loss_ri
            + self.weight_mag * loss_mag
            + self.weight_time * time_loss
        )
        
        return loss