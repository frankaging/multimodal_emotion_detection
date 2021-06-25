# coding=utf-8

"""Processors for different tasks."""

import csv
import os
import json

import pandas as pd
import pickle

import re
import sys

import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import torch
import argparse
import numpy as np
import pandas as pd

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, linguistic_modality=None, 
                 acoustic_modality=None, 
                 visual_modality=None, ratings=None):
        """Constructs a InputExample.
        """
        self.guid = guid
        self.linguistic_modality = linguistic_modality
        self.acoustic_modality = acoustic_modality
        self.visual_modality = visual_modality
        self.ratings = ratings

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()
    
    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

preprocess = {
    'linguistic_timer': lambda df : df.loc[:,'time'],
    'linguistic': lambda df : df.loc[:,'glove0':'glove299'],
    'emotient_timer': lambda df : df.loc[:,'Frametime'],
    'emotient': lambda df : df.loc[:,'AU1':'AU43'],
    'ratings' : lambda df : df.loc[:,'evaluatorWeightedEstimate'] / 100.0,
    'ratings_timer' : lambda df : df.loc[:,'time'],
    'image': lambda df : df.loc[:,'vector0':'vector999'],
    'image_timer': lambda df : df.loc[:,['Frametime']],
    'acoustic': lambda df : df.loc[:,' F0semitoneFrom27.5Hz_sma3nz_amean':' equivalentSoundLevel_dBp'],
    'acoustic_timer': lambda df : df.loc[:,' frameTime']
}
    
class SENDv1_Processor(DataProcessor):
    """Processor for the SENDv1 data set."""

    def __init__(self):
        """load everything into memory first"""

    def get_train_examples(self, data_dir, sample_limit=None):
        """See base class."""
        train_dir = os.path.join(data_dir, "features", "Train")
        return self._create_examples(train_dir, "Train", sample_limit=sample_limit)

    def get_validation_examples(self, data_dir, sample_limit=None):
        """See base class."""
        validation_dir = os.path.join(data_dir, "features", "Valid")
        return self._create_examples(validation_dir, "Valid", sample_limit=sample_limit)

    def get_test_examples(self, data_dir, sample_limit=None):
        """See base class."""
        test_dir = os.path.join(data_dir, "features", "Test")
        return self._create_examples(test_dir, "Test", sample_limit=sample_limit)

    def get_labels(self):
        """See base class."""
        return -1 # this is a continuous scale

    def _timer_syncer(self, feature_timer, ratings_timer, sync_type="up_sample"):
        if sync_type == "up_sample":
            modality_index = 0
            feature_syncer = []
            for i in range(0, len(ratings_timer)):
                if modality_index+1 < len(feature_timer):
                    if ratings_timer[i] >= feature_timer[modality_index+1]:
                        modality_index += 1
                feature_syncer.append([modality_index])
        else:
            modality_index = 0
            feature_syncer = []
            for i in range(0, len(ratings_timer)):
                ratings_window = []
                increment = 0
                for j in range(modality_index, len(feature_timer)):
                    if feature_timer[j] < ratings_timer[min(i+1, len(ratings_timer)-1)]:
                        ratings_window.append(j)
                        increment += 1
                modality_index += increment
                if len(ratings_window) == 0:
                    if len(feature_syncer) >= 1:
                        feature_syncer.append(feature_syncer[-1])
                    else:
                        feature_syncer.append([0]) # let us use the first feature!
                else:
                    feature_syncer.append(ratings_window)
            
        return feature_syncer

    def _feature_collate(self, feature_synced_timer, features, _collate_fn="mean"):
        collate_features = []
        for step in feature_synced_timer:
            step_features = []
            for micro_step in step:
                step_features.append(torch.tensor(features[micro_step]))
            if _collate_fn == "mean":
                step_features = torch.stack(step_features, dim=0)
                step_features = step_features.mean(dim=0)
            else:
                pass # not implemented
            collate_features.append(step_features)
        return torch.stack(collate_features, dim=0)

    def _create_examples(self, data_dir, set_type, debug=True, sample_limit=None):
        examples = []
        # Let us start with linguistics and use it as our baseline.
        vid_set = set([])
        for file in os.listdir(os.path.join(data_dir, "linguistic")):
            if file.endswith(".tsv"):
                file = file.split("_")
                vid_set.add(f"{file[0]}_{file[1]}")
        
        sample_count = 0
        for vid in vid_set:
            # Get features for all modalities.
            if sample_limit and sample_count >= sample_limit:
                break # we are sample how many training examples
            if sample_count % 20 == 0:
                print(f"processing example = {sample_count}, video = {vid} ...")
            # L
            linguistic_file = f"{vid}_aligned.tsv"
            linguistic_file = os.path.join(data_dir, "linguistic", linguistic_file)
            linguistic_data = pd.read_csv(linguistic_file, sep='\t')
            linguistic_features = np.array(preprocess["linguistic"](linguistic_data))
            linguistic_timer = np.array(preprocess["linguistic_timer"](linguistic_data))

            # A
            acoustic_file = f"{vid}_acousticFeatures.csv"
            acoustic_file = os.path.join(data_dir, "acoustic-egemaps", acoustic_file)
            acoustic_data = pd.read_csv(acoustic_file)
            acoustic_features = np.array(preprocess["acoustic"](acoustic_data))
            acoustic_timer = np.array(preprocess["acoustic_timer"](acoustic_data))
            
            # V
            visual_file = f"{vid}_crop_downsized.txt"
            visual_file = os.path.join(data_dir, "emotient", visual_file)
            if not os.path.isfile(visual_file):
                visual_file = f"{vid}_downsized.txt"
                visual_file = os.path.join(data_dir, "emotient", visual_file)
            visual_data = pd.read_csv(visual_file)
            visual_features = np.array(preprocess["emotient"](visual_data))
            visual_timer = np.array(preprocess["emotient_timer"](visual_data))
            
            # ratings
            vid_s = vid.split("_")
            ratings_file = f"results_{vid_s[0][2:]}_{vid_s[1][3:]}.csv"
            ratings_file = os.path.join(
                "/".join(data_dir.split("/")[:-2]), "ratings", set_type, 
                "observer_EWE", ratings_file
            )
            ratings_data = pd.read_csv(ratings_file)
            ratings = np.array(preprocess["ratings"](ratings_data))
            ratings_timer = np.array(preprocess["ratings_timer"](ratings_data))
            
            # Let us sync them together just in time!
            linguistic_synced_timer = self._timer_syncer(linguistic_timer, ratings_timer, sync_type="up_sample")
            acoustic_synced_timer = self._timer_syncer(acoustic_timer, ratings_timer, sync_type="up_sample")
            visual_synced_timer = self._timer_syncer(visual_timer, ratings_timer, sync_type="down_sample")
            assert len(linguistic_synced_timer) == len(ratings_timer)
            assert len(acoustic_synced_timer) == len(ratings_timer)
            assert len(visual_synced_timer) == len(ratings_timer)
            
            # Let us create features based on synced times.
            l_channel = self._feature_collate(linguistic_synced_timer, linguistic_features)
            a_channel = self._feature_collate(acoustic_synced_timer, acoustic_features)
            v_channel = self._feature_collate(visual_synced_timer, visual_features)
            
            example = InputExample(
                vid, linguistic_modality=l_channel, 
                acoustic_modality=a_channel, 
                visual_modality=v_channel, ratings=torch.tensor(ratings)
            )
            examples.append(example)
            sample_count += 1

        return examples