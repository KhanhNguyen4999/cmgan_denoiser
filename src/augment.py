import torch
from torch import nn


class Remix(nn.Module):
    """Remix.
    Mixes different noises with clean speech within a given batch
    """

    def forward(self, sources):
        noise, clean = sources
        bs, *other = noise.shape
        device = noise.device
        perm = torch.argsort(torch.rand(bs, device=device), dim=0)
        return torch.stack([noise[perm], clean])
    
class SNRScale(nn.Module):
    """Remix.
    Mixes different noises with clean speech within a given batch
    """

    def calculate_power(self, signal):
        return torch.mean(signal ** 2, dim=-1)

    def calculate_rms(self, signal):
        return torch.sqrt(torch.mean(signal ** 2, dim=-1))

    def forward(self, sources, snr_scale = 1.2):
        noisy, clean = sources
        noise = noisy - clean

        # Calculate the RMS of clean and noise
        rmsclean = self.calculate_rms(clean)
        rmsnoise = self.calculate_rms(noise)
        
        # Define the SNR reduction
        current_snr_db = 20 * torch.log10(rmsclean / (rmsnoise + 1e-8))
        #print("---Current SNR:", current_snr_db)
        # snr_scale = 0.8  # Reduce SNR to 80%
        desired_snr_db = current_snr_db * snr_scale
        #print("---Desired SNR:", desired_snr_db)
        # Calculate the new scaling factor for noise
        desired_snr_linear = 20 ** (desired_snr_db / 10.0)
        noisescalar = rmsclean / (rmsnoise * desired_snr_linear)

        # Scale the noise with the new scaling factor
        scaled_noise = noise * noisescalar.unsqueeze(-1)

        return scaled_noise, clean
