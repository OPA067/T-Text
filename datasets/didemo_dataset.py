import csv
import json
import torch
from torch.utils.data import Dataset
from config.base_config import Config
from datasets.rawvideo_util import RawVideoExtractor
from modules.basic_utils import load_json


def read_json(json_file):
    with open(json_file) as data_file:
        data = json.load(data_file)
    return data

class DiDeMoDataset(Dataset):

    def __init__(self, config: Config, split_type='train', img_transforms=None):
        self.config = config
        self.img_transforms = img_transforms
        self.split_type = split_type
        self.video_dir = config.videos_dir

        pth = './DiDeMo/'
        if split_type == 'train':
            self.label_csv = pth + 'didemo_train_label.csv'
        elif split_type == 'test':
            self.label_csv = pth + 'didemo_test_label.csv'
        else:
            print('unseen data split type!')
            raise NotImplementedError

        seqcap_json_file = pth + 'DiDeMo_VILA_F6.json'
        self.seqcap_json = load_json(seqcap_json_file)

        self.load_frames_from_preprocess_pth = False
        self.num_frames = config.num_frames
        self.rawVideoExtractor = RawVideoExtractor(framerate=1, size=224)
        self._construct_all_train_pairs()

    def _get_rawvideo(self, video_path, s, e):
        start_time = int(s)
        end_time = int(e)
        start_time = start_time if start_time >= 0. else 0.
        end_time = end_time if end_time >= 0. else 0.
        if start_time > end_time:
            start_time, end_time = end_time, start_time
        elif start_time == end_time:
            end_time = end_time + 1

        imgs = self.rawVideoExtractor.get_video_data(video_path, start_time, end_time, sample_type='uniform', num_frames=self.num_frames)

        return imgs

    def __getitem__(self, index):
        vid, data_id, cap, seq_cap, formatted_data_path, starts, ends, video_path = self._get_vidpath_and_caption_by_index(index)
        starts = int(starts.replace('tensor', '').replace('[', '').replace(']', '').replace(')', '').replace('(', ''))
        ends = int(ends.replace('tensor', '').replace('[', '').replace(']', '').replace(')', '').replace('(', ''))

        video_path = self.video_dir + video_path.split('/')[-1] + '.mp4'

        if self.load_frames_from_preprocess_pth:
            formatted_data = torch.load(formatted_data_path)
            imgs = formatted_data['frames'].squeeze()
        else:
            imgs = self._get_rawvideo(video_path, starts, ends)

        if self.img_transforms is not None:
            imgs = self.img_transforms(imgs)

        return {
            'vid': vid,
            'video': imgs,
            'cap': cap,
            'seq_cap': seq_cap,
        }

    def __len__(self):
        return len(self.all_pairs)

    def _get_vidpath_and_caption_by_index(self, index):
        vid, data_id, cap, seq_cap, formatted_data_path, starts, ends, video_path = self.all_pairs[index]
        data_id = '_'.join(data_id.split('_')[:-1] + ['{:05d}'.format(int(data_id.split('_')[-1]))])
        return vid, data_id, cap, seq_cap, formatted_data_path, starts, ends, video_path

    def _construct_all_train_pairs(self):
        self.all_pairs = []
        with open(self.label_csv, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for i, row in enumerate(csv_reader):
                if i > 0:
                    success, video_path, starts, ends, cap, formatted_data = row
                    data_id = formatted_data.split('/')[-1].split('.')[0]
                    vid = video_path.split('/')[-1]
                    for seq_anno in self.seqcap_json:
                        if seq_anno['name'] == vid:
                            seq_cap = seq_anno['description']
                            seq_cap = seq_cap.split(".")
                            seq_cap.extend([seq_cap[-1]] * (6 - len(seq_cap)))
                            seq_cap = seq_cap[:6]
                            self.all_pairs.append([vid, data_id, cap, seq_cap, formatted_data, starts, ends, video_path])
                            break
        print("train or test all_pairs len is:", len(self.all_pairs))
