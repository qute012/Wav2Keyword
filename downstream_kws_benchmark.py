from fairseq import models
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq import (
    checkpoint_utils,
    distributed_utils,
    options,
    quantization_utils,
    tasks,
    utils,
)

#=====================Model Preparation=====================

import argparse, textwrap
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import sys
import os
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

torch.cuda.set_device(0)

parser = argparse.ArgumentParser(description='downstream pretrained wav2vec model for keyword spotting systems.')
parser.add_argument('pt', type = str, help='pretrained model path')
parser.add_argument('dataset', type = str, help='dataset root directory')
parser.add_argument('name', type = str, help='save model name')

args = parser.parse_args()

save_path = os.path.join('checkpoint',args.name)
os.makedirs(save_path, exist_ok=True)

state_dict = torch.load(args.pt)
cfg = convert_namespace_to_omegaconf(state_dict['args'])

task = tasks.setup_task(cfg.task)
w2v_encoder = task.build_model(cfg.model)
    
class KWS(nn.Module):
    def __init__(self, n_class=30, encoder_hidden_dim=768, cfg=None, state_dict=None):
        super(KWS, self).__init__()
        self.n_class = n_class
        assert not cfg is None
        assert not state_dict is None
        
        self.w2v_encoder = task.build_model(cfg.model)
        self.w2v_encoder.load_state_dict(state_dict)
        
        out_channels = 112
        self.decoder = nn.Sequential(
            nn.Conv1d(encoder_hidden_dim, out_channels, 25, dilation=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, self.n_class, 1)
        )
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        output = self.w2v_encoder(**x, features_only=True)
        output= output['x']
        b,t,c = output.shape
        output = output.reshape(b,c,t)
        output = self.decoder(output).squeeze()
        if self.training:
            return self.softmax(output)
        else:
            return output
        
#=====================Data Loader=====================
import os
import soundfile as sf
from fairseq.data.audio.raw_audio_dataset import * 
import torch.utils.data as data
from torch.utils.data.sampler import WeightedRandomSampler
import librosa
import numpy as np
import random
from speech_commands.input_data import AudioProcessor

12 keywords
CLASSES = 'unknown, silence, yes, no, up, down, left, right, on, off, stop, go'.split(', ')
#20+2 keywords
#CLASSES = 'unknown, silence, yes, no, up, down, left, right, on, off, stop, go, zero, one, two, three, four, five, six, seven, eight, nine'.split(', ')

logging.info(f"classes: {CLASSES}")

"""
model = KWS(n_class=len(CLASSES), cfg=cfg, state_dict=state_dict['model']).cuda()
optimizer = torch.optim.Adam([
    {'params': model.w2v_encoder.parameters(), 'lr': 1e-5},
    #{'params': model.conv.parameters(), 'lr': 5e-4},
    {'params': model.decoder.parameters(), 'lr': 5e-4},
], weight_decay=1e-5)
#,  weight_decay=1e-5
criterion = torch.nn.CrossEntropyLoss().cuda()
"""

class SpeechCommandsDataset(RawAudioDataset):
    
    #sil=0.1, np=0.5, nl=0.7, sp=0.5, mp=0.5
    def __init__(self, mode='train', root='/root/storage/dataset/speech_commands/', 
                 sample_rate=16000, loudest_section=True, silence_percentage=0.1, noise_prob=0.5, noise_level=0.7, shift_prob=0.5, mask_prob=0.5, mask_len=0.1, tf_audio_processor=None, benc_size=None):
        super(SpeechCommandsDataset, self).__init__(
            sample_rate,
            pad=False
        )
        self.mode = mode
        self.root = root
        self.mode_root = os.path.join(root,self.mode)
        self.ap = tf_audio_processor
        self.benc_size = benc_size
        self.loudest_section = loudest_section
        self.sample_rate = sample_rate
        self.data = list()
        if self.benc_size == None:
            self.prep_dataset()
        else:
            self.prep_benc_dataset()
        
        if self.mode=='training':
            self.noise_data = list()
            self.prep_noise_dataset()
            self.noise_prob = noise_prob
            self.noise_level = noise_level
            self.shift_prob = shift_prob
            self.mask_prob = mask_prob
            self.mask_len = mask_len
    
    def prep_dataset(self):
        if self.ap is None:
            self.id = 0
            for c in CLASSES:
                for root, dir, files in os.walk(os.path.join(self.mode_root,c)):
                    for file in files:
                        f_path, cmd = os.path.join(root, file), c
                        self.data.append((f_path, cmd, self.id))
                        self.id += 1
        else:
            self.id = 0
            tf_data = self.ap.data_index[self.mode]
            for td in tf_data:
                f_path, cmd = td['file'], td['label']
                if cmd=='_silence_':
                    self.data += [('','silence',self.id)]
                elif cmd in CLASSES:
                    self.data.append((f_path, cmd, self.id))
                elif not cmd in CLASSES:
                    self.data.append((f_path, 'unknown', self.id))
                self.id += 1
        print(f"{self.mode} data number: {len(self.data)}")
            
    def prep_benc_dataset(self):
        data_num_chk = dict()
        for c in CLASSES:
            data_num_chk[c] = 0
        
        if self.ap is None:
            self.id = 0
            for c in CLASSES:
                for root, dir, files in os.walk(os.path.join(self.mode_root,c)):
                    for file in files:
                        if data_num_chk[c] == self.benc_size:
                            break
                        f_path, cmd = os.path.join(root, file), c
                        self.data.append((f_path, cmd, self.id))
                        self.id += 1
                        data_num_chk[c] += 1
                    if data_num_chk[c] == self.benc_size:
                        break
        else:
            self.id = 0
            tf_data = self.ap.data_index[self.mode]
            for td in tf_data:
                f_path, cmd = td['file'], td['label']
                if not cmd in CLASSES:
                    if cmd=='_silence_':
                        cmd = 'silence'
                    else:
                        cmd = 'unknown'
                        
                if data_num_chk[cmd] == self.benc_size:
                    continue
                
                if cmd=='silence':
                    self.data += [('','silence',self.id)]
                elif cmd=='unknown':
                    self.data.append((f_path, 'unknown', self.id))
                elif cmd in CLASSES:
                    self.data.append((f_path, cmd, self.id))
                
                data_num_chk[cmd] += 1
                self.id += 1
        print(f"{self.mode} data number: {len(self.data)}")
    
    def prep_noise_dataset(self):
        noise_path = os.path.join(self.root,'_background_noise_')
        samples = []
        for root, dir, files in os.walk(noise_path):
            for file in files:
                f_path = os.path.join(root,file)
                wav, _ = sf.read(f_path)
                samples.append(wav)
        samples = np.hstack(samples)
        c = int(self.sample_rate)
        r = len(samples) // c
        self.noise_data = samples[:r*c].reshape(-1, c)
    
    def __getitem__(self, idx):
        f_path, cmd, id = self.data[idx]
        
        if f_path:
            wav, curr_sample_rate = sf.read(f_path)
            if curr_sample_rate!=self.sample_rate:
                #scale=True, size=self.sample_rate, mode='edge'
                wav, curr_sample_rate = librosa.resample(wav, curr_sample_rate, self.sample_rate), self.sample_rate
                
            if len(wav.shape)==2:
                wav = librosa.to_mono(wav.transpose(1,0))
                
            if self.loudest_section:
                wav = self.extract_loudest_section(wav)
            
            wav_len = len(wav)
            if wav_len < self.sample_rate:
                pad_size = self.sample_rate - wav_len
                wav = np.pad(wav, (round(pad_size/2)+1,round(pad_size/2)+1), 'constant', constant_values=0)
        else:
            wav, curr_sample_rate = np.zeros(self.sample_rate, dtype=np.float32), self.sample_rate
        wav_len = len(wav)

        mid = int(len(wav)/2)
        cut_off = int(self.sample_rate/2)
        wav = wav[mid-cut_off:mid+cut_off]

        if self.mode=='training':
            if random.random()<self.shift_prob:
                percentage = random.uniform(-self.shift_prob, self.shift_prob)
                d = int(self.sample_rate*percentage)
                wav = np.roll(wav, d)
                if d>0:
                    wav[:d] = 0
                else:
                    wav[d:] = 0
            
            if random.random()<self.mask_prob:
                t = int(self.mask_len*self.sample_rate)
                t0 = random.randint(0, self.sample_rate - t)
                wav[t0:t+t0] = 0
            
            if random.random()<self.noise_prob:
                noise = random.choice(self.noise_data)
                if cmd=='silence':
                    percentage = random.uniform(0, 1)
                    wav = wav * (1 - percentage) + noise * percentage
                else:
                    percentage = random.uniform(0, self.noise_level)
                    wav = wav * (1 - percentage) + noise * percentage
        feats = torch.from_numpy(wav).float()
        feats = self.postprocess(feats, curr_sample_rate)
        y = CLASSES.index(cmd)
        return {"id": id, "target": y, "source": feats}
    
    def extract_loudest_section(self, wav, win_len=30):
        wav_len = len(wav)
        temp = abs(wav)

        st,et = 0,0
        max_dec = 0

        for ws in range(0, wav_len, win_len):
            cur_dec = temp[ws:ws+16000].sum()
            if cur_dec >= max_dec:
                max_dec = cur_dec
                st,et = ws, ws+16000
            if ws+16000 > wav_len:
                break

        return wav[st:et]
    
    def __len__(self):
        return len(self.data)
    
    def make_weights_for_balanced_classes(self):
        nclasses = len(CLASSES)
        count = np.zeros(nclasses)
        for item in self.data:
            count[CLASSES.index(item[1])] += 1

        N = float(sum(count))
        weight_per_class = N / count
        weight = np.zeros(len(self))
        for idx, item in enumerate(self.data):
            weight[idx] = weight_per_class[CLASSES.index(item[1])]
        return weight
    

tf_ap_model_settings = {
    "desired_samples": 160,
    "fingerprint_size": 40,
    "label_count": 10,
    "window_size_samples": 100,
    "window_stride_samples": 100,
    "fingerprint_width": 40,
    "preprocess": "mfcc",
}

tf_ap_classes = CLASSES[2:]

tf_ap = AudioProcessor("", args.dataset, 10, 10, 
                    tf_ap_classes, 10, 10, tf_ap_model_settings, 
                    os.path.join(args.dataset,'split'))

def _collate_fn(samples):
    sub_samples = [s for s in samples if s["source"] is not None]
    if len(sub_samples) == 0:
        return {}

    batch = test_dataset.collater(samples)
    batch['target'] = torch.LongTensor([s["target"] for s in sub_samples])
    return batch

#=====================Training=====================

logging.info(model)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
logging.info(params)

with open(args.name+'.txt', 'a+') as f:
    f.write(f"{args.name} dataset benchmark\n")

from tqdm import tqdm
import logging
logging.basicConfig(filename='log.log',level=logging.INFO)

def save(name, model, epoch, optimizer, loss, acc):
    path = os.path.join(save_path,name)
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'acc': acc,
                }, path)

epochs = 100
print_step = 10
save_epoch = 10

for bes in [200]:
    
    model = KWS(n_class=len(CLASSES), cfg=cfg, state_dict=state_dict['model']).cuda()
    optimizer = torch.optim.Adam([
        {'params': model.w2v_encoder.parameters(), 'lr': 1e-5},
        #{'params': model.conv.parameters(), 'lr': 5e-4},
        {'params': model.decoder.parameters(), 'lr': 5e-4},
    ], weight_decay=1e-5)
    #,  weight_decay=1e-5
    criterion = torch.nn.CrossEntropyLoss().cuda()

    batch_size = 128

    train_dataset = SpeechCommandsDataset(root=args.dataset, mode='training', tf_audio_processor=tf_ap, benc_size=bes)
    test_dataset = SpeechCommandsDataset(root=args.dataset, mode='testing', tf_audio_processor=tf_ap)

    #weights = train_dataset.make_weights_for_balanced_classes()
    #sampler = WeightedRandomSampler(weights, len(weights))

    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                  collate_fn=_collate_fn, num_workers=4)
    test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                 collate_fn=_collate_fn, num_workers=4)
    best_acc = 0
    best_bes_acc = 0
    for epoch in range(epochs):
        pbar = tqdm(train_dataloader)
        total_loss = 0
        total_sample = 0
        correct = 0
        model.train()
        cur_step = 0
        logging.info("[training]training start")
        for batch in pbar:
            optimizer.zero_grad()
            x, y = batch['net_input'], batch['target']

            for k in x.keys():
                x[k] = x[k].cuda()
            y = y.cuda()

            logits = model(x)
            #print(logits.shape)
            #print(logits)
            #print(y.shape)
            loss = criterion(logits,y)

            total_loss += loss.item()
            total_sample += y.size(0)

            pred = logits.data.max(1, keepdim=True)[1].squeeze()
            #print(pred)
            #print(y)
            correct += pred.eq(y.data.view_as(pred)).sum().item()
            #print(correct, total_sample)

            loss.backward()
            optimizer.step()
            cur_step += 1
            acc = (correct/total_sample)*100
            pbar.set_description("loss: {} acc:{:.2f}".format(total_loss/cur_step, acc))

            #if cur_step%print_step==0:
            #    logging.info("[training]epoch {}\tloss: {}\tacc:{}\t".format(epoch,total_loss/cur_step, (correct/total_sample)*100))

        logging.info("[test]test start")
        with torch.no_grad():
            pbar = tqdm(test_dataloader)
            total_loss = 0
            total_sample = 0
            correct = 0
            cur_step = 0
            model.eval()
            for batch in pbar:
                x, y = batch['net_input'], batch['target']

                for k in x.keys():
                    x[k] = x[k].cuda()
                y = y.cuda()

                logits = model(x)

                loss = criterion(logits,y)

                total_loss += loss.item()
                total_sample += y.size(0)

                pred = logits.data.max(1, keepdim=True)[1].squeeze()
                #print(pred)
                #print(y)
                cur_step += 1
                correct += pred.eq(y.data.view_as(pred)).sum().item()
                #print(correct, total_sample)
                acc = (correct/total_sample)*100
                pbar.set_description("[test]epoch {} loss: {} acc:{:.2f}".format(epoch,total_loss/cur_step, acc))
            if acc>best_acc:
                best_acc = acc
                best_bes_acc = acc
                save('best_model.pth', model, epoch, optimizer, loss, acc)


            #if epoch%save_epoch==0:
            #    save('{}epoch-{:.2f}acc.pth'.format(epoch, acc)
            #         , model, epoch, optimizer, loss, acc)
            save('latest_model.pth', model, epoch, optimizer, loss, acc)
        logging.info("best acc: {:.2f}".format(best_acc))
        #logging.info("[test]epoch {}\tloss: {}\tacc:{}\t".format(epoch,total_loss/cur_step, (correct/total_sample)*100))
    with open(args.name+'.txt', 'a+') as f:
        f.write(f"{bes}\t{best_bes_acc}\n")