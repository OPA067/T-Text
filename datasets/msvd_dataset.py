import os

from modules.basic_utils import load_json, read_lines
from torch.utils.data import Dataset
from config.base_config import Config
from datasets.video_capture import VideoCapture

class MSVDDataset(Dataset):

    def __init__(self, config: Config, split_type='train', img_transforms=None):
        self.config = config
        self.videos_dir = config.videos_dir
        self.img_transforms = img_transforms
        self.split_type = split_type
        json_file = './MSVD/captions_msvd_small.json'
        self.cap_json = load_json(json_file)
        seqcap_json_file = './MSVD/MSVD_VILA_F6.json'
        self.seqcap_json = load_json(seqcap_json_file)

        train_file = './MSVD/train_list.txt'
        val_file = './MSVD/val_list.txt'
        test_file = './MSVD/test_list.txt'

        if split_type == 'train':
            self.train_vids = read_lines(train_file)
            self._construct_all_train_pairs()
        else:
            self.test_vids = read_lines(test_file)
            self._construct_all_test_pairs()

    def __getitem__(self, index):
        if self.split_type == 'train':
            vid, video_path, cap, seq_cap = self._get_vidpath_and_caption_by_index_train(index)
        else:
            vid, video_path, cap, seq_cap = self._get_vidpath_and_caption_by_index_test(index)

        imgs, idxs = VideoCapture.load_frames_from_video(video_path, self.config.num_frames, self.config.video_sample_type)

        if self.img_transforms is not None:
            imgs = self.img_transforms(imgs)

        return {
            'vid': vid,
            'video': imgs,
            'cap': cap,
            'seq_cap': seq_cap
        }

    def _get_vidpath_and_caption_by_index_train(self, index):
        vid, cap, seq_cap = self.all_train_pairs[index]
        video_path = os.path.join(self.videos_dir, vid + '.avi')
        return vid, video_path, cap, seq_cap

    def _get_vidpath_and_caption_by_index_test(self, index):
        vid, cap, seq_cap = self.all_test_pairs[index]
        video_path = os.path.join(self.videos_dir, vid + '.avi')
        return vid, video_path, cap, seq_cap

    def __len__(self):
        if self.split_type == 'train':
            return len(self.all_train_pairs)
        return len(self.all_test_pairs)

    def _construct_all_train_pairs(self):
        self.all_train_pairs = []
        for vid in self.train_vids:
            cap = self.cap_json[vid]
            for seq_anno in self.seqcap_json:
                if seq_anno['name'] == vid:
                    seq_cap = seq_anno['description']
                    seq_cap = seq_cap.split(".")
                    seq_cap.extend([seq_cap[-1]] * (6 - len(seq_cap)))
                    seq_cap = seq_cap[:6]
                    self.all_train_pairs.append([vid, cap, seq_cap])
                    break
        print("The all_train_pairs len is:", len(self.all_train_pairs))

    def _construct_all_test_pairs(self):
        self.all_test_pairs = []
        for vid in self.test_vids:
            cap = self.cap_json[vid]
            for seq_anno in self.seqcap_json:
                if seq_anno['name'] == vid:
                    seq_cap = seq_anno['description']
                    seq_cap = seq_cap.split(".")
                    seq_cap.extend([seq_cap[-1]] * (6 - len(seq_cap)))
                    seq_cap = seq_cap[:6]
                    self.all_test_pairs.append([vid, cap, seq_cap])
                    break
        print("The all_test_pairs len is:", len(self.all_test_pairs))


