import os
import numpy as np
import pandas as pd
from collections import defaultdict
from modules.basic_utils import load_json
from torch.utils.data import Dataset
from config.base_config import Config
from datasets.video_capture import VideoCapture

class MSRVTTDataset(Dataset):

    def __init__(self, config: Config, split_type='train', img_transforms=None):
        self.config = config
        self.videos_dir = config.videos_dir
        self.img_transforms = img_transforms
        self.split_type = split_type

        dir = 'MSRVTT'
        train_json_file = dir + '/MSRVTT_data.json'
        self.train_json = load_json(train_json_file)

        seqcap_json_file = dir + '/MSRVTT_VILA_F6.json'
        self.seqcap_json = load_json(seqcap_json_file)

        train_csv_path = dir + '/MSRVTT_train.9000.csv'
        test_csv_path = dir + '/MSRVTT_test.1000.csv'

        if split_type == 'train':
            self.train_csv = pd.read_csv(train_csv_path)
            self.train_vids = self.train_csv['video_id']
            self._compute_vid2caption()
            self._construct_all_pairs()
        else:
            self.test_csv = pd.read_csv(test_csv_path)
            self.test_vids = self.test_csv['video_id']
            self._compute_vid2caption()
            self._construct_all_pairs()

    def __getitem__(self, index):

        if self.split_type == 'train':
            vid, video_path, cap, seq_cap = self._get_vidpath_and_caption_by_index(index)
            imgs, idxs = VideoCapture.load_frames_from_video(video_path, self.config.num_frames, self.config.video_sample_type)

            if self.img_transforms is not None:
                imgs = self.img_transforms(imgs)

            return {
                'vid': vid,
                'video': imgs,
                'cap': cap,
                'seq_cap': seq_cap
            }
        else:
            vid, video_path, cap, seq_cap = self._get_vidpath_and_caption_by_index(index)
            imgs, idxs = VideoCapture.load_frames_from_video(video_path, self.config.num_frames, self.config.video_sample_type)

            if self.img_transforms is not None:
                imgs = self.img_transforms(imgs)

            return {
                'vid': vid,
                'video': imgs,
                'cap': cap,
                'seq_cap': seq_cap
            }

    def __len__(self):
        if self.split_type == 'train':
            return len(self.all_train_pairs)
        return len(self.all_test_pairs)

    def _get_vidpath_and_caption_by_index(self, index):
        if self.split_type == 'train':
            vid, caption, seq_caption = self.all_train_pairs[index]
            video_path = os.path.join(self.videos_dir, vid + '.mp4')
            return vid, video_path, caption, seq_caption
        else:
            vid, caption, seq_caption = self.all_test_pairs[index]
            video_path = os.path.join(self.videos_dir, vid + '.mp4')
            return vid, video_path, caption, seq_caption
    
    def _construct_all_pairs(self):
        self.all_train_pairs, self.all_test_pairs = [], []
        if self.split_type == 'train':
            for vid in self.train_vids:
                for caption_list in self.train_vid2caption[vid]:
                    self.all_train_pairs.append([vid, caption_list[0], caption_list[1]])
            print("train len is", len(self.all_train_pairs))
        else:
            for vid in self.test_vids:
                for caption_list in self.test_vid2caption[vid]:
                    self.all_test_pairs.append([vid, caption_list[0], caption_list[1]])
            print("test len is", len(self.all_test_pairs))

    def _compute_vid2caption(self):
        if self.split_type == 'train':
            self.train_vid2caption = defaultdict(list)
            for anno in self.train_json['sentences']:
                caption = anno['caption']
                vid = anno['video_id']
                for seq_anno in self.seqcap_json:
                    if seq_anno['name'] == vid:
                        seq_cap = seq_anno['description']
                        seq_cap = seq_cap.split(".")
                        seq_cap.extend([seq_cap[-1]] * (6 - len(seq_cap)))
                        seq_cap = seq_cap[:6]
                        self.train_vid2caption[vid].append([caption, seq_cap])
                        break
        else:
            self.test_vid2caption = defaultdict(list)
            for index, row in self.test_csv.iterrows():
                cap = row['sentence']
                vid = row['video_id']
                for seq_anno in self.seqcap_json:
                    if seq_anno['name'] == vid:
                        seq_cap = seq_anno['description']
                        seq_cap = seq_cap.split(".")
                        seq_cap.extend([seq_cap[-1]] * (6 - len(seq_cap)))
                        seq_cap = seq_cap[:6]
                        self.test_vid2caption[vid].append([cap, seq_cap])
                        break