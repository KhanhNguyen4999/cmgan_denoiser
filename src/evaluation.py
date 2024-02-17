import numpy as np
from models import generator
from natsort import natsorted
import os
from tools.compute_metrics import compute_metrics
from joblib import Parallel, delayed
from utils import *
import torchaudio
import soundfile as sf
import argparse
import time
import logging 

logging.basicConfig(filename="train.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)


logger = logging.getLogger(__name__)

@torch.no_grad()
def enhance_one_track(model, audio_path, saved_dir, cut_len, n_fft=400, hop=100, save_tracks=False):
    try:  
        name = os.path.split(audio_path)[-1]
        noisy, sr = torchaudio.load(audio_path)
        assert sr == 16000
        noisy = noisy.cuda()

        c = torch.sqrt(noisy.size(-1) / torch.sum((noisy ** 2.0), dim=-1))
        noisy = torch.transpose(noisy, 0, 1)
        noisy = torch.transpose(noisy * c, 0, 1)

        length = noisy.size(-1)
        frame_num = int(np.ceil(length / 100))
        padded_len = frame_num * 100
        padding_len = padded_len - length
        noisy = torch.cat([noisy, noisy[:, :padding_len]], dim=-1)
        if padded_len > cut_len:
            batch_size = int(np.ceil(padded_len/cut_len))
            while 100 % batch_size != 0:
                batch_size += 1
            noisy = torch.reshape(noisy, (batch_size, -1))

        noisy_spec = torch.stft(noisy, n_fft, hop, window=torch.hamming_window(n_fft).cuda(), onesided=True)
        noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
        time_start = time.time()
        
        if noisy_spec.shape[2] % 2 == 0:
            import torch.nn.functional as F
            noisy_spec = F.pad(noisy_spec, (0, 0, 0, 1))

        est_real, est_imag = model(noisy_spec)
        time_end = time.time()
        est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)

        est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
        est_audio = torch.istft(est_spec_uncompress, n_fft, hop, window=torch.hamming_window(n_fft).cuda(),
                                onesided=True)
        est_audio = est_audio / c
        est_audio = torch.flatten(est_audio)[:length].cpu().numpy()
        assert len(est_audio) == length
        if save_tracks:
            saved_path = os.path.join(saved_dir, name)
            sf.write(saved_path, est_audio, sr)

        RTF = (time_end - time_start) / (length / sr)
    except:
        est_audio = None
        length = None
    return est_audio, length, RTF


def evaluation(model_path, noisy_dir, clean_dir, save_tracks, saved_dir):
    n_fft = 400
  
    state_dict = torch.load(model_path)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove 'module.' of DataParallel/DistributedDataParallel
        # name = k
        new_state_dict[name] = v

    model = generator.TSCNet(num_channel=64, num_features=n_fft//2+1).cuda()
    model.load_state_dict(new_state_dict)
    model.eval()

    if not os.path.exists(saved_dir):
        os.mkdir(saved_dir)

    audio_list = os.listdir(noisy_dir)
    audio_list = natsorted(audio_list)
    
    ls_est_audio = Parallel(n_jobs=10)(
                delayed(enhance_one_track)(model, 
                                            os.path.join(noisy_dir, audio),
                                            saved_dir,
                                            16000*10, 
                                            n_fft, 
                                            n_fft//4, 
                                            save_tracks
                                            ) for audio in audio_list)
    
    RTF = np.mean([e[2] for e in ls_est_audio])
    
    sr = 16000
    metrics = Parallel(n_jobs=10)(
        delayed(compute_metrics)(sf.read(os.path.join(clean_dir, audio_list[i]))[0],
                                ls_est_audio[i][0],
                                sr,
                                0) for i in range(len(ls_est_audio))
    )

    metrics_avg = np.mean(metrics, 0)

    print('pesq: ', metrics_avg[0], 
          'csig: ', metrics_avg[1], 
          'cbak: ', metrics_avg[2], 
          'covl: ', metrics_avg[3], 
          'ssnr: ', metrics_avg[4], 
          'stoi: ', metrics_avg[5],
          'RTF: ', RTF )

def evaluation_model(model, noisy_dir, clean_dir, save_tracks, saved_dir):
    n_fft = 400
    model.eval()

    if not os.path.exists(saved_dir):
        os.mkdir(saved_dir)

    audio_list = os.listdir(noisy_dir)
    audio_list = natsorted(audio_list)
    
    ls_est_audio = Parallel(n_jobs=1)(
                delayed(enhance_one_track)(model, 
                                            os.path.join(noisy_dir, audio),
                                            saved_dir,
                                            16000*10, 
                                            n_fft, 
                                            n_fft//4, 
                                            save_tracks
                                            ) for audio in audio_list)
    
    sr = 16000
    metrics = Parallel(n_jobs=10)(
        delayed(compute_metrics)(sf.read(os.path.join(clean_dir, audio_list[i]))[0],
                                ls_est_audio[i][0],
                                sr,
                                0) for i in range(len(ls_est_audio))
    )

    metrics_avg = np.mean(metrics, 0)
    RTF = np.mean([e[2] for e in ls_est_audio])
    
    # print('pesq: ', metrics_avg[0], 'csig: ', metrics_avg[1], 'cbak: ', metrics_avg[2], 'covl: ',
    #       metrics_avg[3], 'ssnr: ', metrics_avg[4], 'stoi: ', metrics_avg[5])

    metrics_avg_dict = {"pesq": metrics_avg[0], 
                        "csig": metrics_avg[1], 
                        "cbak": metrics_avg[2], 
                        "covl": metrics_avg[3], 
                        "ssnr": metrics_avg[4],
                        "stoi": metrics_avg[5],
                        "RTF": RTF}
    
    return metrics_avg_dict

def eval_best_loss(checkpoint_path):
    package = torch.load(checkpoint_path, map_location = "cpu")
 
    best_loss = package['loss']
    epochs = package['epoch']
    print(f"Best loss in {best_loss} epochs: {epochs}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='/home/minhkhanh/Desktop/work/denoiser/CMGAN/src/best_ckpt/ckpt',
                        help="the path where the model is saved")
    parser.add_argument("--test_dir", type=str, default='/home/minhkhanh/Desktop/work/denoiser/dataset/voice_bank_demand/test',
                        help="noisy tracks dir to be enhanced")
    parser.add_argument("--save_tracks", type=str, default=True, help="save predicted tracks or not")
    parser.add_argument("--save_dir", type=str, default='./saved_tracks_best', help="where enhanced tracks to be saved")

    parser.add_argument("-e", "--eval", required=False, 
                                        type=str, 
                                        help="Path to checkpoint dir", 
                                        default="")

    args = parser.parse_args()

    if args.eval:
        eval_best_loss(args.eval)
    else:
        noisy_dir = os.path.join(args.test_dir, 'noisy')
        clean_dir = os.path.join(args.test_dir, 'clean')
        load_from_checkpoint = True
        evaluation(args.model_path, noisy_dir, clean_dir, args.save_tracks, args.save_dir)
