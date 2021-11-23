#!/usr/bin/env python
# coding: utf-8

# In[90]:


import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import torch
from PIL import Image
from numpy import asarray

import argparse
from collections import namedtuple, OrderedDict
import itertools
import os
import numpy as np
from typing import Tuple
from typing import List
from typing import Dict
import random
from itertools import product
import copy
import re
import random
import hashlib
import pathlib
import json
import torch.nn.functional as F
from scipy.stats import pearsonr
import wandb

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
)

import logging

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm import tqdm, trange

# Building up our SEND model.
from models.BERT import *
from models.VGGFace2 import *
from models.optimization import *

class InputFeature:
    
    def __init__(
        self, video_id="",
        acoustic_feature=[],
        linguistic_feature=[],
        visual_feature=[],
        labels=[],
    ):
        self.video_id = video_id
        self.acoustic_feature = acoustic_feature
        self.linguistic_feature = linguistic_feature
        self.visual_feature = visual_feature
        self.labels = labels
        
def preprocess_SEND_files(
    data_dir, # Multitmodal X
    target_data_dir, # Y
    use_target_ratings,
    time_window_in_sec=4.0,
    modality_dir_map = {"acoustic": "acoustic-egemaps",  
                        "linguistic": "linguistic-word-level", # we don't load features
                        "visual": "image-raw", # image is nested,
                        "target": "observer_EWE",
                       },
    preprocess= {'acoustic': lambda df : df.loc[:,' F0semitoneFrom27.5Hz_sma3nz_amean':' equivalentSoundLevel_dBp'],
                 'acoustic_timer': lambda df : df.loc[:,' frameTime'],
                 'linguistic': lambda df : df.loc[:,'word'],
                 'linguistic_timer': lambda df : df.loc[:,'time-offset'],
                 'target': lambda df : ((df.loc[:,'evaluatorWeightedEstimate'] / 50.0) - 1.0),
                 'target_timer': lambda df : df.loc[:,'time'],
                },
    linguistic_tokenizer=None,
    pad_symbol=0,
    max_number_of_file=-1
):
    
    import time

    start = time.time()    
    SEND_videos = []
    
    # basically, let us gett all the video ids?
    a_ids = [f.split("_")[0]+"_"+f.split("_")[1] 
             for f in listdir(os.path.join(data_dir, modality_dir_map["acoustic"])) 
             if isfile(os.path.join(data_dir, modality_dir_map["acoustic"], f))]
    l_ids = [f.split("_")[0]+"_"+f.split("_")[1] 
             for f in listdir(os.path.join(data_dir, modality_dir_map["linguistic"])) 
             if isfile(os.path.join(data_dir, modality_dir_map["linguistic"], f))]
    v_ids = [f.split("_")[0]+"_"+f.split("_")[1] 
             for f in listdir(os.path.join(data_dir, modality_dir_map["visual"])) 
             if f != ".DS_Store"]
    assert len(a_ids) == len(l_ids) and len(l_ids) == len(v_ids)
    assert len(set(a_ids).intersection(set(l_ids))) == len(l_ids)
    assert len(set(a_ids).intersection(set(v_ids))) == len(v_ids)
    
    # We need the first pass for linguistic modality process?
    max_window_l_length = -1
    for video_id in a_ids: # pick any one!
        # linguistic features process
        l_file = os.path.join(data_dir, modality_dir_map["linguistic"], f"{video_id}_aligned.tsv")
        l_df = pd.read_csv(l_file, sep='\t')
        #l_words = np.array(preprocess["linguistic"](l_df))
        #l_words = [w.strip().lower() for w in l_words]
        l_words = []
        l_timestamps = []
        head = True
        with open(l_file) as fp:
            for line in fp:
                if head:
                    head = False
                    continue
                l_words.append(line.strip().split("\t")[2].lower().strip())
                l_timestamps.append(float(line.strip().split("\t")[1]))

        #l_timestamps = np.array(preprocess["linguistic_timer"](l_df))
        l_timestamps = np.array(l_timestamps)
        # sample based on interval
        current_time = 0.0
        keep_first = True
        sampled_l_words = [] # different from other modality, it is essentially a list of list!
        tmp_words = []
        for i in range(0, l_timestamps.shape[0]):
            if keep_first:
                sampled_l_words += [[]]
                keep_first = False
            if l_timestamps[i] >= current_time+time_window_in_sec:
                sampled_l_words.append(tmp_words)
                tmp_words = [l_words[i]] # reinit the buffer
                current_time += time_window_in_sec
                continue
            tmp_words += [l_words[i]]
        # overflow
        if len(tmp_words) > 0:
            sampled_l_words.append(tmp_words)
        for window_words in sampled_l_words:
            window_str = " ".join(window_words)
            window_tokens = linguistic_tokenizer.tokenize(window_str)
            token_ids = linguistic_tokenizer.convert_tokens_to_ids(window_tokens)
            if len(token_ids) > max_window_l_length:
                max_window_l_length = len(token_ids)

    max_window_l_length += 2 # the start and the end token
    
    if max_number_of_file != -1:
        logger.info(f"WARNING: Only loading #{max_number_of_file} videos.")
    max_seq_len = -1
    video_count = 0
    for video_id in a_ids: # pick any one!
        if max_number_of_file != -1 and video_count >= max_number_of_file:
            break # we enforce!
        if video_count > 1 and video_count%100 == 0:
            logger.info(f"Processed #{len(SEND_videos)} videos.")
            # logger.info(SEND_videos[-1])
        
        # we need to fix this to get features aligned.
        
        # Step 1: Load rating data, and we can get window partitioned according to our interval.
        target_id = video_id.split("_")[0][2:] + "_" + video_id.split("_")[1][3:]
        if use_target_ratings:
            target_file = os.path.join(target_data_dir, modality_dir_map["target"], f"target_{target_id}_normal.csv")
        else:
            target_file = os.path.join(target_data_dir, modality_dir_map["target"], f"results_{target_id}.csv")
        target_df = pd.read_csv(target_file)
        target_ratings = np.array(preprocess["target"](target_df))
        target_timestamps = np.array(preprocess["target_timer"](target_df))
        assert target_ratings.shape[0] == target_timestamps.shape[0]
        windows = []
        number_of_window = int(max(target_timestamps)//time_window_in_sec)
        for i in range(0, number_of_window):
            windows += [(i*time_window_in_sec, (i+1)*time_window_in_sec)]
        windows += [((i+1)*time_window_in_sec, max(target_timestamps))]
        # [(0, 5], (5, 10], ...]

        # acoustic features process
        a_file = os.path.join(data_dir, modality_dir_map["acoustic"], f"{video_id}_acousticFeatures.csv")
        a_df = pd.read_csv(a_file)
        a_features = np.array(preprocess["acoustic"](a_df))
        a_timestamps = np.array(preprocess["acoustic_timer"](a_df))
        a_feature_dim = a_features.shape[1]
        assert a_features.shape[0] == a_timestamps.shape[0]
        sampled_a_features_raw = [[] for i in range(len(windows))]
        for i in range(0, a_timestamps.shape[0]):
            # using mod to hash to the correct bucket.
            hash_in_window = int(a_timestamps[i]//time_window_in_sec)
            if hash_in_window >= len(windows):
                continue # we cannot predict after ratings max.
            sampled_a_features_raw[hash_in_window].append(a_features[i])
        sampled_a_features = []
        for window in sampled_a_features_raw:
            # only acoustic need to consider this I think.
            if len(window) == 0:
                collate_window = np.zeros(a_feature_dim)
            else:
                collate_window = np.mean(np.array(window), axis=0)
            sampled_a_features.append(collate_window)
        
        
        # linguistic features process
        l_file = os.path.join(data_dir, modality_dir_map["linguistic"], f"{video_id}_aligned.tsv")
        l_df = pd.read_csv(l_file, sep='\t')
        # the following line is buggy, it may parse file incorrectly!
        #l_words = np.array(preprocess["linguistic"](l_df))
        #l_words = [w.strip().lower() for w in l_words]
        l_words = []
        l_timestamps = []
        head = True
        with open(l_file) as fp:
            for line in fp:
                if head:
                    head = False
                    continue
                l_words.append(line.strip().split("\t")[2].lower().strip())
                l_timestamps.append(float(line.strip().split("\t")[1]))
        #l_timestamps = np.array(preprocess["linguistic_timer"](l_df))
        l_timestamps = np.array(l_timestamps)
        assert len(l_words) == l_timestamps.shape[0]
        
        sampled_l_features_raw = [[] for i in range(len(windows))]
        for i in range(0, l_timestamps.shape[0]):
            # using mod to hash to the correct bucket.
            hash_in_window = int(l_timestamps[i]//time_window_in_sec)
            if hash_in_window >= len(windows):
                continue # we cannot predict after ratings max.
            sampled_l_features_raw[hash_in_window].append(l_words[i])

        sampled_l_features = []
        sampled_l_mask = []
        sampled_l_segment_ids = []
        for window in sampled_l_features_raw:
            window_str = " ".join(window)
            window = linguistic_tokenizer.tokenize(window_str)
            complete_window_word = ["[CLS]"] + window + ["[SEP]"]
            token_ids = linguistic_tokenizer.convert_tokens_to_ids(complete_window_word)
            input_mask = [1 for _ in range(len(token_ids))]
            for _ in range(0, max_window_l_length-len(token_ids)):
                token_ids.append(linguistic_tokenizer.pad_token_id)
                input_mask.append(0)
            segment_ids = [0] * len(token_ids)
            sampled_l_features += [token_ids]
            sampled_l_mask += [input_mask]
            sampled_l_segment_ids += [segment_ids]


        # visual features process
        # for visual, we actually need to active control what image we load, we
        # cannot just load all images, it will below memory.
        fps=30 # We may need to dynamically figure out this number?
        frame_names = []
        for f in listdir(os.path.join(data_dir, modality_dir_map["visual"], video_id)):
            if ".jpg" in f:
                frame_names += [(int(f.split("_")[0][5:])*(1.0/fps), f)]
        frame_names.sort(key=lambda x:x[0])
        sampled_v_features_raw = [[] for i in range(len(windows))]
        for f in frame_names:
            # using mod to hash to the correct bucket.
            hash_in_window = int(f[0]//time_window_in_sec)
            if hash_in_window >= len(windows):
                continue # we cannot predict after ratings max.
            sampled_v_features_raw[hash_in_window].append(f)
            
        sampled_v_features = []
        for window in sampled_v_features_raw:
            if len(window) == 0:
                f_data = np.zeros((224,224,3))
            else:
                # we collate by using the last frame in the time window.
                f = window[-1]
                f_path = os.path.join(data_dir, modality_dir_map["visual"], video_id, f[1])
                f_image = Image.open(f_path)
                f_data = asarray(f_image)
                f_data = f_data[...,::-1] # reverse the order.
            sampled_v_features.append(f_data)

        # ratings (target)
        target_id = video_id.split("_")[0][2:] + "_" + video_id.split("_")[1][3:]
        if use_target_ratings:
            target_file = os.path.join(target_data_dir, modality_dir_map["target"], f"target_{target_id}_normal.csv")
        else:
            target_file = os.path.join(target_data_dir, modality_dir_map["target"], f"results_{target_id}.csv")
        target_df = pd.read_csv(target_file)
        target_ratings = np.array(preprocess["target"](target_df))
        target_timestamps = np.array(preprocess["target_timer"](target_df))
        assert target_ratings.shape[0] == target_timestamps.shape[0]
        sampled_ratings_raw = [[] for i in range(len(windows))]
        for i in range(0, target_timestamps.shape[0]):
            # using mod to hash to the correct bucket.
            hash_in_window = int(target_timestamps[i]//time_window_in_sec)
            sampled_ratings_raw[hash_in_window].append(target_ratings[i])
        sampled_ratings = []
        for window in sampled_ratings_raw:
            collate_window = np.mean(np.array(window), axis=0)
            sampled_ratings.append(collate_window)
        
        # we truncate features based on linguistic avaliabilities.
        assert len(sampled_a_features) == len(sampled_l_features)
        assert len(sampled_a_features) == len(sampled_v_features)
        
        max_window_cutoff_l = int(max(l_timestamps)//time_window_in_sec)
        max_window_cutoff_a = int(max(a_timestamps)//time_window_in_sec)
        max_window_cutoff_v = int(frame_names[-1][0]//time_window_in_sec)
        max_window_cutoff = min([max_window_cutoff_l, max_window_cutoff_a, max_window_cutoff_v])
        sampled_a_features = sampled_a_features[:max_window_cutoff]
        sampled_l_features = sampled_l_features[:max_window_cutoff]
        sampled_v_features = sampled_v_features[:max_window_cutoff]
        sampled_ratings = sampled_ratings[:max_window_cutoff]
        sampled_l_mask = sampled_l_mask[:max_window_cutoff]
        sampled_l_segment_ids = sampled_l_segment_ids[:max_window_cutoff]
        input_mask = np.ones(len(sampled_a_features)).tolist()
        max_seq_len = 60
        seq_len = len(sampled_a_features)
        for i in range(max_seq_len-len(sampled_a_features)):
            sampled_a_features.append(np.zeros(a_feature_dim))
            sampled_l_features.append(np.zeros(max_window_l_length))
            sampled_l_mask.append(np.zeros(max_window_l_length))
            sampled_l_segment_ids.append(np.zeros(max_window_l_length))
            sampled_v_features.append(np.zeros((224,224,3)))
            sampled_ratings.append(0.0)
            input_mask.append(0)

        sampled_a_features = torch.tensor(sampled_a_features)
        sampled_l_features = torch.LongTensor(sampled_l_features)
        sampled_l_mask = torch.LongTensor(sampled_l_mask)
        sampled_l_segment_ids = torch.LongTensor(sampled_l_segment_ids)
        processed_tensor = torch.tensor(sampled_v_features).float()
        processed_tensor[..., 0] -= 91.4953
        processed_tensor[..., 1] -= 103.8827
        processed_tensor[..., 2] -= 131.0912
        sampled_v_features = processed_tensor
        sampled_ratings = torch.tensor(sampled_ratings)
        input_mask = torch.LongTensor(input_mask)
        
        video_struct = {
            "video_id": video_id,
            "a_feature": sampled_a_features,
            "l_feature": sampled_l_features,
            "l_mask": sampled_l_mask,
            "l_segment_ids": sampled_l_segment_ids,
            "v_feature": sampled_v_features,
            "rating": sampled_ratings,
            "seq_len": seq_len,
            "input_mask": input_mask
        }
        video_count += 1
        SEND_videos += [video_struct]
    
    end = time.time()
    elapsed = end - start
    logger.info(f"Time elapsed for first-pass: {elapsed}")
        
    return SEND_videos

def eval_ccc(y_true, y_pred):
    """Computes concordance correlation coefficient."""
    true_mean = np.mean(y_true)
    true_var = np.var(y_true)
    pred_mean = np.mean(y_pred)
    pred_var = np.var(y_pred)
    covar = np.cov(y_true, y_pred, bias=True)[0][1]
    ccc = 2*covar / (true_var + pred_var +  (pred_mean-true_mean) ** 2)
    return ccc

class MultimodalEmotionPrediction(nn.Module):

    def __init__(
        self, 
        linguistic_model="bert-base-uncased",
        visual_model="vggface-2",
        visual_model_path="../saved-models/resnet50_scratch_dag.pth",
        acoustic_model="mlp",
        cache_dir="../.huggingface_cache/",
    ):
        super(MultimodalEmotionPrediction, self).__init__()
        
        # Loading BERT using huggingface?
        linguistic_config = AutoConfig.from_pretrained(
            linguistic_model,
            cache_dir=cache_dir
        )
        self.linguistic_encoder = LinguisticEncoderBERT.from_pretrained(
            linguistic_model,
            from_tf=False,
            config=linguistic_config,
            cache_dir=cache_dir
        )
        # let us disenable gradient prop
        # for name, param in self.linguistic_encoder.named_parameters():
        #     param.requires_grad = False
        
        # Loading visual model using vggface-2
        self.visual_encoder = Resnet50_scratch_dag()
        state_dict = torch.load(visual_model_path)
        self.visual_encoder.load_state_dict(state_dict)
        self.visual_reducer = nn.Linear(2048, 768)

        # Rating lstm.
        # hidden_dim = 128
        hidden_dim = 768        
        self.rating_decoder = nn.LSTM(
                                hidden_dim, 64, 1, 
                                batch_first=True, bidirectional=False)
                                              
        # Rating decoder.
        self.rating_output = nn.Sequential(
            nn.Linear(64, 1)
        )
        
        self.acoustic_encoder = nn.Linear(88, 32)
        self.rating_decoder_a = nn.LSTM(
                                32, 1, 1, 
                                batch_first=True, bidirectional=False)
            
        self.rating_decoder_v = nn.LSTM(
                                768, 1, 1, 
                                batch_first=True, bidirectional=False)
            
    def forward(
        self, input_a_feature, input_l_feature, 
        input_l_mask, input_l_segment_ids, 
        input_v_feature, train_rating_labels, input_mask,
    ):
        
        # linguistic encoder
        batch_size, seq_len = input_l_feature.shape[0], input_l_feature.shape[1]
        input_l_feature = input_l_feature.reshape(batch_size*seq_len, -1)
        input_l_mask = input_l_mask.reshape(batch_size*seq_len, -1)
        input_l_segment_ids = input_l_segment_ids.reshape(batch_size*seq_len, -1)
        
        l_decode = self.linguistic_encoder(
            input_ids=input_l_feature,
            attention_mask=input_l_mask,
            token_type_ids=input_l_segment_ids,
        )
        l_decode = l_decode.reshape(batch_size, seq_len, -1)
        
        # visual encoder
        input_v_feature = input_v_feature.reshape(batch_size*seq_len, 224, 224, 3)
        input_v_feature = input_v_feature.permute(0,3,1,2).contiguous()
        _, v_decode = self.visual_encoder(input_v_feature)
        v_decode = v_decode.squeeze(dim=-1).squeeze(dim=-1).contiguous()
        v_decode = v_decode.reshape(batch_size, seq_len, -1)
        v_decode = self.visual_reducer(v_decode)
        
        # decoding to ratings.
        output, (_, _) = self.rating_decoder(l_decode)
        output = self.rating_output(output)
        output = output.squeeze(dim=-1)
        output = output * input_mask
        
        a_decode = self.acoustic_encoder(input_a_feature)
        output_a, (_, _) = self.rating_decoder_a(a_decode)
        output_a = output_a.squeeze(dim=-1)
        output_a = output_a * input_mask
        
        output_v, (_, _) = self.rating_decoder_v(v_decode)
        output_v = output_v.squeeze(dim=-1)
        output_v = output_v * input_mask
        
        output += output_a
        output += output_v
        
        # get loss.
        criterion = nn.MSELoss(reduction='sum')
        loss = criterion(output, train_rating_labels)
        
        return loss, output
    
def evaluate(
    test_dataloader, model, device, args,
):
    pbar = tqdm(test_dataloader, desc="Iteration")
    ccc = []
    corr = []
    outputs = []
    total_loss = 0
    data_num = 0
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(pbar):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            input_a_feature, input_l_feature, input_l_mask, input_l_segment_ids,                 input_v_feature, rating_labels, seq_lens, input_mask = batch
            input_a_feature = input_a_feature.to(device)
            input_l_feature = input_l_feature.to(device)
            input_l_mask = input_l_mask.to(device)
            input_l_segment_ids = input_l_segment_ids.to(device)
            input_v_feature = input_v_feature.to(device)
            rating_labels = rating_labels.to(device)
            seq_lens = seq_lens.to(device)
            input_mask = input_mask.to(device)

            loss, output =                 model(input_a_feature, input_l_feature, input_l_mask, input_l_segment_ids,
                      input_v_feature, rating_labels, input_mask)
            n_gpu = torch.cuda.device_count()
            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            total_loss += loss.data.cpu().detach().tolist()
            data_num += torch.sum(seq_lens).tolist()
            output_array = output.cpu().detach().numpy()
            rating_labels_array = rating_labels.cpu().detach().numpy()
            for i in range(0, input_a_feature.shape[0]):
                ccc.append(eval_ccc(rating_labels_array[i][:int(seq_lens[i].tolist()[0])], output_array[i][:int(seq_lens[i].tolist()[0])]))
                corr.append(pearsonr(output_array[i][:int(seq_lens[i].tolist()[0])], rating_labels_array[i][:int(seq_lens[i].tolist()[0])])[0])
                outputs.append(output_array[i])
        total_loss /= data_num
    return total_loss, ccc, corr, outputs
    
def train(
    train_dataloader, test_dataloader, model, optimizer, 
    device, args
):
    global_step = 0
    best_ccc, best_corr = -1, -1
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        pbar = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(pbar):
            model.train()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            input_a_feature, input_l_feature, input_l_mask, input_l_segment_ids,                 input_v_feature, rating_labels, seq_lens, input_mask = batch
            input_a_feature = input_a_feature.to(device)
            input_l_feature = input_l_feature.to(device)
            input_l_mask = input_l_mask.to(device)
            input_l_segment_ids = input_l_segment_ids.to(device)
            input_v_feature = input_v_feature.to(device)
            rating_labels = rating_labels.to(device)
            seq_lens = seq_lens.to(device)
            input_mask = input_mask.to(device)
            
            loss, output =                 model(input_a_feature, input_l_feature, input_l_mask, input_l_segment_ids,
                      input_v_feature, rating_labels, input_mask)
            n_gpu = torch.cuda.device_count()
            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            loss /= (torch.sum(seq_lens).tolist())
            loss.backward() # uncomment this for actual run!
            optimizer.step()
            optimizer.zero_grad()

            pbar.set_description("loss: %.4f"%loss)
            if args.is_tensorboard:
                wandb.log({"train_loss": loss.cpu().detach().numpy()})
            
            if global_step%args.eval_interval == 0:
                logger.info('Evaluating the model...')
                # we need to evaluate!
                loss, ccc, corr, outputs = evaluate(
                    test_dataloader, model, device, args,
                )
                
                if np.mean(ccc) > best_ccc:
                    best_ccc = np.mean(ccc)
                    # save best ccc models.
                    if args.save_best_model:
                        logger.info('Saving the new best model for ccc...')
                        checkpoint = {'model': model.state_dict()}
                        checkpoint_path = os.path.join(args.output_dir, "best_ccc_pytorch_model.bin")
                        torch.save(checkpoint, checkpoint_path)
                if np.mean(corr) > best_corr:
                    best_corr = np.mean(corr)
                    # save best corr models.
                    if args.save_best_model:
                        logger.info('Saving the new best model for corr...')
                        checkpoint = {'model': model.state_dict()}
                        checkpoint_path = os.path.join(args.output_dir, "best_corr_pytorch_model.bin")
                        torch.save(checkpoint, checkpoint_path)
                        
                # Average statistics and print
                stats = {'eval_loss': loss, 'corr': np.mean(corr), 'corr_std': np.std(corr),
                         'ccc': np.mean(ccc), 'ccc_std': np.std(ccc), 
                         'best_ccc': best_ccc, 'best_corr': best_corr}
                if args.is_tensorboard:
                    wandb.log(stats)
                logger.info(f'Evaluation results: {stats}')
                
            global_step +=  1


# In[ ]:


def arg_parse():
    
    parser = argparse.ArgumentParser(description='multimodal emotion analysis argparse.')
    # Experiment management:

    parser.add_argument('--train_batch_size', type=int, default=6,
                        help='Training batch size.')
    parser.add_argument('--eval_batch_size', type=int, default=12,
                        help='Evaluation batch size.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate.')
    parser.add_argument('--warmup_proportion', type=float, default=0.1,
                        help='Warmup period.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    parser.add_argument('--num_train_epochs', type=float, default=3,
                        help='Number of training epochs.')
    parser.add_argument('--eval_interval', type=int, default=20,
                        help='Evaluation interval in steps.')
    parser.add_argument('--max_number_of_file', type=int, default=-1,
                        help='Maybe we just want to test with a few number of files.')
    
    parser.add_argument('--resumed_from_file_path', type=str, default="",
                        help='Whether to resume for this file.')
    parser.add_argument('--data_dir', type=str, default="../../SENDv1-data/",
                        help='Whether to resume for this file.')
    parser.add_argument('--output_dir', type=str, default="../default_output_log/",
                        help='Whether to resume for this file.')
    parser.add_argument("--is_tensorboard",
                        default=False,
                        action='store_true',
                        help="Whether to use tensorboard.")
    parser.add_argument("--save_best_model",
                        default=False,
                        action='store_true',
                        help="Whether to save the best model during eval.")
    parser.add_argument("--eval_only",
                        default=False,
                        action='store_true',
                        help="Whether we are evaluating the model only.")
    parser.add_argument("--debug_only",
                        default=False,
                        action='store_true',
                        help="Whether we are debugging the code only.")
    parser.add_argument("--use_target_ratings",
                        default=False,
                        action='store_true',
                        help="Whether to use target ratings from the dataset.")

    parser.set_defaults(
        # Exp management:
        seed=42,
    )
    try:
        get_ipython().run_line_magic('matplotlib', 'inline')
        args = parser.parse_args([])
    except:
        args = parser.parse_args()
    return args


# In[ ]:


if __name__ == "__main__":
    
    # Loading arguments
    args = arg_parse()
    try:        
        get_ipython().run_line_magic('matplotlib', 'inline')
        # Experiment management:
        args.train_batch_size=1
        args.eval_batch_size=1
        args.lr=8e-5
        args.seed=42
        args.is_tensorboard=True # Let us try this!
        args.output_dir="../default_output_log/"
        is_jupyter = True
    except:
        is_jupyter = False
        
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Create output directory if not exists.
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True) 
    
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s %(levelname)-8s %(message)s', 
        datefmt='%a, %d %b %Y %H:%M:%S', 
        filename=os.path.join(args.output_dir, "training.log"),
    )
    logger = logging.getLogger(__name__)
    logging.getLogger().addHandler(logging.StreamHandler(os.sys.stdout))
    
    logger.info("Training the model with the following parameters: ")
    logger.info(args)
    
    if args.is_tensorboard and not is_jupyter:
        logger.warning("Enabling wandb for tensorboard logging...")
        run = wandb.init(project="SEND-Multimodal", entity="wuzhengx")
        run_name = wandb.run.name
        wandb.config.update(args)
    else:
        wandb = None
    
    # We don't allow flexibility here..
#     tokenizer = AutoTokenizer.from_pretrained(
#         "bert-base-uncased",
#         use_fast=False,
#         cache_dir="../.huggingface_cache/"
#     )
    
#     train_SEND_features = None
#     test_SEND_features = None
    
#     if args.use_target_ratings:
#         logger.info("WARNING: use_target_ratings is setting to TRUE.")
#         modality_dir_map = {"acoustic": "acoustic-egemaps",  
#                             "linguistic": "linguistic-word-level", # we don't load features
#                             "visual": "image-raw", # image is nested,
#                             "target": "target"}
#         preprocess = {
#             'acoustic': lambda df : df.loc[:,' F0semitoneFrom27.5Hz_sma3nz_amean':' equivalentSoundLevel_dBp'],
#             'acoustic_timer': lambda df : df.loc[:,' frameTime'],
#             'linguistic': lambda df : df.loc[:,'word'],
#             'linguistic_timer': lambda df : df.loc[:,'time-offset'],
#             'target': lambda df : ((df.loc[:,' rating'] / 0.5) - 1.0),
#             'target_timer': lambda df : df.loc[:,'time'],
#         }
#     else:
#         logger.info("WARNING: use_target_ratings is setting to FALSE.")
#         modality_dir_map = {"acoustic": "acoustic-egemaps",  
#                             "linguistic": "linguistic-word-level", # we don't load features
#                             "visual": "image-raw", # image is nested,
#                             "target": "observer_EWE"}
#         preprocess = {
#             'acoustic': lambda df : df.loc[:,' F0semitoneFrom27.5Hz_sma3nz_amean':' equivalentSoundLevel_dBp'],
#             'acoustic_timer': lambda df : df.loc[:,' frameTime'],
#             'linguistic': lambda df : df.loc[:,'word'],
#             'linguistic_timer': lambda df : df.loc[:,'time-offset'],
#             'target': lambda df : ((df.loc[:,'evaluatorWeightedEstimate'] / 50.0) - 1.0),
#             'target_timer': lambda df : df.loc[:,'time'],
#         }
#     if not args.eval_only:
#         # Training data loading 
#         train_modalities_data_dir = os.path.join(args.data_dir, "features/Train/")
#         train_target_data_dir = os.path.join(args.data_dir, "ratings/Train")

#         test_modalities_data_dir = os.path.join(args.data_dir, "features/Valid/")
#         test_target_data_dir = os.path.join(args.data_dir, "ratings/Valid")
        
#         train_SEND_features = preprocess_SEND_files(
#             train_modalities_data_dir,
#             train_target_data_dir,
#             args.use_target_ratings,
#             modality_dir_map=modality_dir_map,
#             preprocess=preprocess,
#             linguistic_tokenizer=tokenizer,
#             max_number_of_file=args.max_number_of_file
#         )
#         if args.debug_only:
#             logger.info("WARNING: Debugging only. Evaluate and Train datasets are the same.")
#             test_SEND_features = copy.deepcopy(train_SEND_features)
#         else:
#             test_SEND_features = preprocess_SEND_files(
#                 test_modalities_data_dir,
#                 test_target_data_dir,
#                 args.use_target_ratings,
#                 modality_dir_map=modality_dir_map,
#                 preprocess=preprocess,
#                 linguistic_tokenizer=tokenizer,
#             )
        
#     else:
#         test_modalities_data_dir = os.path.join(args.data_dir, "features/Test/")
#         test_target_data_dir = os.path.join(args.data_dir, "ratings/Test")
    
#         test_SEND_features = preprocess_SEND_files(
#             test_modalities_data_dir,
#             test_target_data_dir,
#             args,
#             modality_dir_map=modality_dir_map,
#             preprocess=preprocess,
#             linguistic_tokenizer=tokenizer,
#             max_number_of_file=args.max_number_of_file
#         )
    train_data = torch.load('./train_data.pt')
    test_data = torch.load('./test_data.pt')
    logger.info("Finish Loading Datasets...")
    
    if not args.eval_only:
        # Initialize all the datasets
#         train_input_a_feature = torch.stack([video_struct["a_feature"] for video_struct in train_SEND_features]).float()
#         train_input_l_feature = torch.stack([video_struct["l_feature"] for video_struct in train_SEND_features])
#         train_input_l_mask = torch.stack([video_struct["l_mask"] for video_struct in train_SEND_features])
#         train_input_l_segment_ids = torch.stack([video_struct["l_segment_ids"] for video_struct in train_SEND_features])
#         train_input_v_feature = torch.stack([video_struct["v_feature"] for video_struct in train_SEND_features]).float()
#         train_rating_labels = torch.stack([video_struct["rating"] for video_struct in train_SEND_features]).float()
#         train_seq_lens = torch.tensor([[video_struct["seq_len"]] for video_struct in train_SEND_features]).float()
#         train_input_mask = torch.stack([video_struct["input_mask"] for video_struct in train_SEND_features])

#         test_input_a_feature = torch.stack([video_struct["a_feature"] for video_struct in test_SEND_features]).float()
#         test_input_l_feature = torch.stack([video_struct["l_feature"] for video_struct in test_SEND_features])
#         test_input_l_mask = torch.stack([video_struct["l_mask"] for video_struct in test_SEND_features])
#         test_input_l_segment_ids = torch.stack([video_struct["l_segment_ids"] for video_struct in test_SEND_features])
#         test_input_v_feature = torch.stack([video_struct["v_feature"] for video_struct in test_SEND_features]).float()
#         test_rating_labels = torch.stack([video_struct["rating"] for video_struct in test_SEND_features]).float()
#         test_seq_lens = torch.tensor([[video_struct["seq_len"]] for video_struct in test_SEND_features]).float()
#         test_input_mask = torch.stack([video_struct["input_mask"] for video_struct in test_SEND_features])

#         train_data = TensorDataset(
#             train_input_a_feature, 
#             train_input_l_feature, train_input_l_mask, train_input_l_segment_ids,
#             train_input_v_feature, train_rating_labels, train_seq_lens, train_input_mask
#         )
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

#         test_data = TensorDataset(
#             test_input_a_feature, 
#             test_input_l_feature, test_input_l_mask, test_input_l_segment_ids,
#             test_input_v_feature, test_rating_labels, test_seq_lens, test_input_mask
#         )
        test_dataloader = DataLoader(test_data, batch_size=args.eval_batch_size, shuffle=False)
    else:
        logger.info("Not implemented...")
        
    if not args.eval_only:
        # Init model with optimizer.
        model = MultimodalEmotionPrediction()
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() 
                if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
            {'params': [p for n, p in model.named_parameters() 
                if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
            ]
        num_train_steps = int(
            len(train_data) / args.train_batch_size * args.num_train_epochs)
        # We use the default BERT optimz to do gradient descent.
        # optimizer = BERTAdam(optimizer_parameters,
        #                     lr=args.lr,
        #                     warmup=args.warmup_proportion,
        #                     t_total=num_train_steps)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        # Determine the device.
        if not torch.cuda.is_available() or is_jupyter:
            device = torch.device("cpu")
            n_gpu = -1
        else:
            device = torch.device("cuda")
            n_gpu = torch.cuda.device_count()

        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        model = model.to(device)

        train(
            train_dataloader, test_dataloader, model, optimizer,
            device, args
        )
    else:
        logger.info("Not implemented...")

