import torchaudio
import numpy as np
import os
import random
from scipy import signal
from torch.nn import functional as F
import torch
import subprocess
import argparse

EPS = np.finfo(float).eps

def is_clipped(audio, clipping_threshold=0.99):
    return torch.any(abs(audio) > clipping_threshold)

def normalize(audio, target_level=-25):
    '''Normalize the signal to the target level'''
    rms = (audio ** 2).mean() ** 0.5
    scalar = 10 ** (target_level / 20) / (rms+EPS)
    audio = audio * scalar
    return audio

def snr_mixer(clean, 
            noise, 
            snr, 
            target_level=-25, 
            clipping_threshold=0.99, 
            target_level_lower=-35, 
            target_level_upper=-15):
    '''Function to mix clean speech and noise at various SNR levels'''

    clean = clean.squeeze()
    noise = noise.squeeze()
    if clean.size(0) > noise.size(0):
        rest = clean.size(0) - noise.size(0)
        half_p = rest//2
        noise = F.pad(noise, (half_p, rest-half_p))   
    else:
        noise = noise[:clean.size(0)]

    # Normalizing to -25 dB FS
    clean_power = clean.norm(2)
    noise_power = noise.norm(2)+EPS

    # Set the noise level for a given SNR
    # noisescalar = clean_power / (10**(snr/20)) / (noise_power+EPS)
    # noisenewlevel = noise * noisescalar
    bck_noisescalar = torch.sqrt(10**(-snr / 10) * clean_power/noise_power)
    noisenewlevel = noise * bck_noisescalar

    # Mix noise and clean speech
    noisyspeech = clean + noisenewlevel
    
    #Randomly select RMS value between -15 dBFS and -35 dBFS and normalize noisyspeech with that value
    #There is a chance of clipping that might happen with very less probability, which is not a major issue. 
    # noisy_rms_level = np.random.randint(target_level_lower, target_level_upper)
    # rmsnoisy = (noisyspeech**2).mean()**0.5
    # scalarnoisy = 10 ** (noisy_rms_level / 20) / (rmsnoisy+EPS)
    # noisyspeech = noisyspeech * scalarnoisy
    # clean = clean * scalarnoisy

    # Final check to see if there are any amplitudes exceeding +/- 1. If so, normalize all the signals accordingly
    if is_clipped(noisyspeech):
        noisyspeech_maxamplevel = max(abs(noisyspeech))/(clipping_threshold-EPS)
        noisyspeech = noisyspeech/noisyspeech_maxamplevel

    clean = clean.unsqueeze(0)
    noisyspeech = noisyspeech.unsqueeze(0)
    return clean, noisyspeech

def snr_mixer_bcknoise_and_infnoise(clean, 
                                    bcknoise, 
                                    bck_snr, 
                                    infnoise, 
                                    inf_snr, 
                                    target_level=-25, 
                                    clipping_threshold=0.99, 
                                    target_level_lower=-35, 
                                    target_level_upper=-15):
    '''Function to mix clean speech and noise at various SNR levels'''
    base_scaler = 20
    clean = clean.squeeze()
    bcknoise = bcknoise.squeeze()
    infnoise = infnoise.squeeze()
    
    if clean.size(0) > infnoise.size(0):
        rest = clean.size(0) - infnoise.size(0)
        index = np.random.randint(rest)
        infnoise = F.pad(infnoise, (index, rest-index))   
    else:
        infnoise = infnoise[:clean.size(0)]
    
    if clean.size(0) > bcknoise.size(0):
        bcknoise = F.pad(bcknoise, (0, clean.size(0) - bcknoise.size(0)))   
    else:
        bcknoise = bcknoise[:clean.size(0)]

    # Normalizing to -25 dB FS
    clean_power = clean.norm(2)
    bcknoise_power = bcknoise.norm(2)+EPS
    infnoise_power = infnoise.norm(2)+EPS

    # Set the noise level for a given SNR
    # bck_noisescalar = clean_power / (10**(bck_snr/base_scaler)) / (bcknoise_power+EPS)
    bck_noisescalar = torch.sqrt(10**(-bck_snr / 10) * clean_power/bcknoise_power)
    bck_noisenewlevel = bcknoise * bck_noisescalar

    # inf_noisescalar = clean_power / (10**(inf_snr/base_scaler)) / (infnoise_power+EPS)
    inf_noisescalar = torch.sqrt(10**(-inf_snr / 10) * clean_power/infnoise_power)
    inf_noisenewlevel = infnoise * inf_noisescalar

    # Mix noise, interfering noise with clean speech
    noisyspeech = clean + bck_noisenewlevel +  inf_noisenewlevel
    
    #Randomly select RMS value between -15 dBFS and -35 dBFS and normalize noisyspeech with that value
    #There is a chance of clipping that might happen with very less probability, which is not a major issue. 
    # noisy_rms_level = np.random.randint(target_level_lower, target_level_upper)
    # rmsnoisy = (noisyspeech**2).mean()**0.5
    # scalarnoisy = 10 ** (noisy_rms_level / base_scaler) / (rmsnoisy+EPS)
    # noisyspeech = noisyspeech * scalarnoisy
    # clean = clean * scalarnoisy

    # Final check to see if there are any amplitudes exceeding +/- 1. If so, normalize all the signals accordingly
    if is_clipped(noisyspeech):
        noisyspeech_maxamplevel = max(abs(noisyspeech))/(clipping_threshold-EPS)
        noisyspeech = noisyspeech/noisyspeech_maxamplevel

    clean = clean.unsqueeze(0)
    noisyspeech = noisyspeech.unsqueeze(0)
    return clean, noisyspeech


def mixer_2(clean, noise, snr):
    if clean.size(1) > noise.size(1):
        noise = F.pad(noise, (0, clean.size(1)- noise.size(1)))
    else:
        noise = noise[:, :clean.size(1)]
    SNR_dB=snr
    clean_power = clean.norm(2)
    noise_power = noise.norm(2)+1.19209290e-7
    scale = torch.sqrt(10**(-SNR_dB / 10) * clean_power/noise_power)
    mixer = (scale*noise + clean)
    return mixer

def add_pyreverb(clean_speech, rir):
    reverb_speech = signal.fftconvolve(clean_speech, rir, mode="full")
    # make reverb_speech same length as clean_speech
    reverb_speech = reverb_speech[:, : clean_speech.size(1)]
    reverb_speech = torch.from_numpy(reverb_speech)
    return reverb_speech

def gen_noisy_audio(use_inf, 
                    use_reverb,
                    clean_audio_path, 
                    noise_audio_path, 
                    snr_noise,
                    inf_audio_path=None, 
                    rir_audio_path=None,
                    snr_noise_inf=None):
    clean, sr = torchaudio.load(clean_audio_path)
    try:
        noise, sr = torchaudio.load(noise_audio_path)
    except:
        print("Error loading file path: ", noise_audio_path)
    

    if use_inf:
        noise_inter, sr = torchaudio.load(inf_audio_path)
        if use_reverb:
            rir, sr = torchaudio.load(rir_audio_path)
            if rir.size(0)>1:
                rir = rir.sum(axis=0) / rir.size(0)
                rir = rir.unsqueeze(0)
            clean = add_pyreverb(clean, rir)
            noise_inter = add_pyreverb(noise_inter, rir)

        clean, noisy = snr_mixer_bcknoise_and_infnoise(clean, noise, snr_noise, noise_inter, snr_noise_inf)
    else:
        if use_reverb:
            rir, sr = torchaudio.load(rir_audio_path)
            if rir.size(0)>1:
                rir = rir.sum(axis=0) / rir.size(0)
                rir = rir.unsqueeze(0)
            clean = add_pyreverb(clean, rir)
    
        clean, noisy = snr_mixer(clean, noise, snr_noise)
    
    return clean, noisy

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="make pair clean-noisy data")
    parser.add_argument("--rir_file", 
                        type=str, 
                        default="/data1/speech/khanhnnm/database/dns_db/rir/valid.txt", 
                        help="file txt contain path of reverb file")

    parser.add_argument("--noise_file",
                        type=str,
                        default="/data1/speech/khanhnnm/database/dns_db/valid_noise.txt",
                        help="file txt contain path of noise audio: noise set need to contain vary type of noise")

    parser.add_argument("--clean_file",
                        type=str,
                        default="/data1/speech/khanhnnm/database/denoiser_db/kaldi_db/reading_150h/test.txt",
                        help="file txt contain path of clean audio")

    parser.add_argument("--dir",
                        type=str,
                        default="/data1/speech/khanhnnm/database/denoiser_db/kaldi_db/reading_150h/mix_testset/",
                        help="workspace contain testset generation")

    parser.add_argument("--min_snr_bcknoise", 
                        type=int, 
                        default=0, 
                        help="Minimum SNR background noise")

    parser.add_argument("--max_snr_bcknoise", 
                        type=int, 
                        default=15, 
                        help="Maximum SNR background noise")

    parser.add_argument("--min_snr_infnoise", 
                        type=int, 
                        default=0, 
                        help="Minimum SNR interference noise")

    parser.add_argument("--max_snr_infnoise", 
                        type=int, 
                        default=15, 
                        help="Maximum SNR interference noise")
    
    parser.add_argument("--ratio_reverb", 
                        type=float, 
                        default=0.0, 
                        help="the ratio of selected audios to mix reverb")
    
    parser.add_argument("--ratio_inf", 
                        type=float, 
                        default=0.0, 
                        help="the ratio of selected audios to mix interference noise")
    

    # parse params
    args = parser.parse_args()
    rir_file = args.rir_file
    noise_file = args.noise_file
    clean_file = args.clean_file
    min_snr_bcknoise = args.min_snr_bcknoise
    max_snr_bcknoise = args.max_snr_bcknoise
    min_snr_infnoise = args.min_snr_infnoise
    max_snr_infnoise = args.max_snr_infnoise
    ratio_reverb = args.ratio_reverb
    ratio_inf = args.ratio_inf
    sr = 16000
    dir=args.dir
    dir = f"{dir}/train_bcksnr{min_snr_bcknoise}-{max_snr_bcknoise}_infsnr{min_snr_infnoise}-{max_snr_infnoise}_reverb_{ratio_reverb}_inf_{ratio_inf}"



    save_dir_clean = dir + "/clean"
    save_dir_noisy = dir + "/noisy"

    if not os.path.isdir(dir):
        os.mkdir(dir)
    
    if not os.path.isdir(save_dir_clean):
        os.mkdir(save_dir_clean)

    if not os.path.isdir(save_dir_noisy):
        os.mkdir(save_dir_noisy)


    rir_db = []
    with open(rir_file, 'r') as reader:
        rir_db = [ line.strip() for line in reader.readlines() ]
    rir_db_samples = len(rir_db)

    noise_db = []
    with open(noise_file, 'r') as reader:
        noise_db = [ line.strip() for line in reader.readlines() ]
    
    n_noises_samples = len(noise_db)

    writer = open(f"{dir}/statistical.txt", 'w')
    writer.write("{}\t\t{}\t{}\t{}\t{}\n".format("filename", "   snr_noise", "snr_inf", "use_reverb", "user_inf"))
    
    # rm -rf old_folder
    subprocess.run(f"rm -rf {save_dir_clean}/*wav", shell=True)
    subprocess.run(f"rm -rf {save_dir_noisy}/*wav", shell=True)

    with open(clean_file, 'r') as reader:
        testset = [ line.strip() for line in reader.readlines() ]
        n_testset_sample = len(testset)
        for i, clean_audio_path in enumerate(testset):
            clean_audio_path = clean_audio_path.strip()
            filename = os.path.basename(clean_audio_path)

            inf_audio_path = None
            rir_audio_path = None

            use_reverb = False
            use_inf = False
            if random.random() < ratio_reverb:
                use_reverb = True
            if random.random() < ratio_inf:
                use_inf = True
            
            
            snr_noise = random.randint(min_snr_bcknoise, max_snr_bcknoise)
            snr_inf = random.randint(min_snr_infnoise, max_snr_infnoise)
            noise_audio_idx = random.randint(0, n_noises_samples-1)
            noise_audio_path = noise_db[noise_audio_idx]


            if use_reverb:
                rir_audio_idx = random.randint(0, rir_db_samples-1)
                rir_audio_path = rir_db[rir_audio_idx]

            if use_inf:
                inf_audio_idx = random.randint(0, n_testset_sample-1)
                if inf_audio_idx == noise_audio_idx:
                    if noise_audio_idx > 0:
                        inf_audio_idx -=1
                    else: inf_audio_idx += 1
                inf_audio_path = testset[inf_audio_idx]

            # statistic
            writer.write("{}\t\t   {}\t   {}\t   {}\t   {}\n".format(filename, snr_noise, snr_inf, use_reverb, use_inf))

            print("-----i: ", i)            
            print(noise_audio_path)
            print(clean_audio_path)
            print(inf_audio_path)
            print(rir_audio_path)

            # gen noisy audio
            clean, noisy = gen_noisy_audio(
                                    use_inf,
                                    use_reverb,
                                    clean_audio_path,
                                    noise_audio_path,
                                    snr_noise,
                                    inf_audio_path,
                                    rir_audio_path,
                                    snr_inf
                                )
            
            torchaudio.backend.sox_io_backend.save(f"{save_dir_clean}/{filename}", clean, sr, encoding="PCM_S", bits_per_sample=16)
            torchaudio.backend.sox_io_backend.save(f"{save_dir_noisy}/{filename}", noisy, sr, encoding="PCM_S", bits_per_sample=16)
        
        reader.close()
        writer.close()

    

