import torch
import torch.nn as nn

from config.base_config import Config
from modules.dtw import fast_dtw
from modules.loss import KL

class CLIPStochastic(nn.Module):
    def __init__(self, config: Config):
        super(CLIPStochastic, self).__init__()
        self.config = config

        from transformers import CLIPModel
        if config.clip_arch == 'ViT-B/32':
            self.clip = CLIPModel.from_pretrained("./openai/clip-vit-base-patch32")
        elif config.clip_arch == 'ViT-B/16':
            self.clip = CLIPModel.from_pretrained("./openai/clip-vit-base-patch16")
        else:
            raise ValueError

        self.embed = self.config.embed_dim
        self.dtw = fast_dtw()


    def forward(self, data, is_train=True):
        batch_size = data['video'].shape[0]
        video_data = data['video']
        video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res)
        f_feats = self.clip.get_image_features(video_data)
        f_feats = f_feats.reshape(batch_size, self.config.num_frames, -1)

        cap = data['cap']
        t_feats = self.clip.get_text_features(**cap)
        c_feats_list = []
        for i in range(self.config.num_frames // 2):
            seq_cap = data['seq_cap'][i]
            c_feats = self.clip.get_text_features(**seq_cap)
            c_feats_list.append(c_feats)
        c_feats = torch.stack(c_feats_list, dim=1)

        if is_train:
            b, c, d = c_feats.shape
            f = f_feats.shape[1]
            # I:    mlp&dtw ==>> c_feats & f_feats
            c_feats = c_feats / c_feats.norm(dim=-1, keepdim=True)
            f_feats = f_feats / f_feats.norm(dim=-1, keepdim=True)
            sims_cf = torch.einsum("acd,bfd->abcf", [c_feats, f_feats])
            mask_cf = self.dtw(c_feats, f_feats)
            sims_cf = torch.einsum("abcf,abcf->ab", [sims_cf, mask_cf])

            # II:   mlp&dtw ==>> t_feats & c_feats
            t_feats = t_feats / t_feats.norm(dim=-1, keepdim=True)
            c_feats = c_feats / c_feats.norm(dim=-1, keepdim=True)
            sims_tc = torch.einsum('ad,bcd->abc', [t_feats, c_feats])
            indices = torch.argsort(sims_tc, dim=-1, descending=True)
            mask_tc = torch.zeros_like(sims_tc, device=sims_tc.device)
            a_indices = torch.arange(sims_tc.shape[0], device=sims_tc.device)[:, None, None]
            b_indices = torch.arange(sims_tc.shape[1], device=sims_tc.device)[None, :, None]
            mask_tc[a_indices, b_indices, indices[:, :, :c // 2]] = 1
            sims_tc = torch.einsum('abc,abc->ab', [sims_tc, mask_tc])

            # IV:   mlp&dtw ==>> t_feats & f_feats
            t_feats = t_feats / t_feats.norm(dim=-1, keepdim=True)
            f_feats = f_feats / f_feats.norm(dim=-1, keepdim=True)
            sims_tf = torch.einsum('ad,bfd->abf', [t_feats, f_feats])
            indices = torch.argsort(sims_tf, dim=-1, descending=True)
            mask_tf = torch.zeros_like(sims_tf, device=sims_tf.device)
            a_indices = torch.arange(sims_tf.shape[0], device=sims_tf.device)[:, None, None]
            b_indices = torch.arange(sims_tf.shape[1], device=sims_tf.device)[None, :, None]
            mask_tf[a_indices, b_indices, indices[:, :, :f // 2]] = 1
            sims_tf = torch.einsum('abf,abf->ab', [sims_tf, mask_tf])

            return sims_cf, sims_tc, sims_tf
        else:
            return t_feats, c_feats, f_feats

    def get_similarity_logits(self, t_feats, c_feats, f_feats):

        b, c, d = c_feats.shape
        f = f_feats.shape[1]
        # I:    mlp&dtw ==>> c_feats & f_feats
        c_feats = c_feats / c_feats.norm(dim=-1, keepdim=True)
        f_feats = f_feats / f_feats.norm(dim=-1, keepdim=True)
        sims_cf = torch.einsum("acd,bfd->abcf", [c_feats, f_feats])
        mask_cf = self.dtw(c_feats, f_feats)
        sims_cf = torch.einsum("abcf,abcf->ab", [sims_cf, mask_cf])

        # II:   mlp&dtw ==>> t_feats & c_feats
        t_feats = t_feats / t_feats.norm(dim=-1, keepdim=True)
        c_feats = c_feats / c_feats.norm(dim=-1, keepdim=True)
        sims_tc = torch.einsum('ad,bcd->abc', [t_feats, c_feats])
        indices = torch.argsort(sims_tc, dim=-1, descending=True)
        mask_tc = torch.zeros_like(sims_tc, device=sims_tc.device)
        a_indices = torch.arange(sims_tc.shape[0], device=sims_tc.device)[:, None, None]
        b_indices = torch.arange(sims_tc.shape[1], device=sims_tc.device)[None, :, None]
        mask_tc[a_indices, b_indices, indices[:, :, :c // 2]] = 1
        sims_tc = torch.einsum('abc,abc->ab', [sims_tc, mask_tc])

        # IV:   mlp&dtw ==>> t_feats & f_feats
        t_feats = t_feats / t_feats.norm(dim=-1, keepdim=True)
        f_feats = f_feats / f_feats.norm(dim=-1, keepdim=True)
        sims_tf = torch.einsum('ad,bfd->abf', [t_feats, f_feats])
        indices = torch.argsort(sims_tf, dim=-1, descending=True)
        mask_tf = torch.zeros_like(sims_tf, device=sims_tf.device)
        a_indices = torch.arange(sims_tf.shape[0], device=sims_tf.device)[:, None, None]
        b_indices = torch.arange(sims_tf.shape[1], device=sims_tf.device)[None, :, None]
        mask_tf[a_indices, b_indices, indices[:, :, :f // 2]] = 1
        sims_tf = torch.einsum('abf,abf->ab', [sims_tf, mask_tf])

        sims_w = [1.0, 1.0, 1.0]
        sims = sims_cf * sims_w[0] + sims_tc * sims_w[1] + sims_tf * sims_w[2]

        return sims