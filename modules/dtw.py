from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn

class fast_dtw(nn.Module):
    def __init__(self):
        super(fast_dtw, self).__init__()

    def fastdtw(self, x, y, window=None):
        x = np.asanyarray(x, dtype='float')
        y = np.asanyarray(y, dtype='float')
        len_x, len_y = len(x), len(y)
        if window is None:
            window = [(i, j) for i in range(len_x) for j in range(len_y)]
        window = ((i + 1, j + 1) for i, j in window)
        D = defaultdict(lambda: (float('-inf'),))
        D[0, 0] = (0, 0, 0)
        for i, j in window:
            x[i - 1] = x[i - 1] / np.linalg.norm(x[i - 1], axis=-1, keepdims=True)
            y[i - 1] = y[i - 1] / np.linalg.norm(y[i - 1], axis=-1, keepdims=True)
            dt = x[i - 1] @ y[j - 1]
            D[i, j] = max((D[i - 1, j][0] + dt, i - 1, j), (D[i, j - 1][0] + dt, i, j - 1),
                          (D[i - 1, j - 1][0] + dt, i - 1, j - 1), key=lambda a: a[0])
        path = []
        i, j = len_x, len_y
        while not (i == j == 0):
            path.append((i - 1, j - 1))
            i, j = D[i, j][1], D[i, j][2]
        path.reverse()
        return D[len_x, len_y][0], path

    def forward(self, c_feats, f_feats):
        b, c, f = c_feats.size(0), c_feats.size(1), f_feats.size(1)
        mask = torch.zeros(b, b, c, f, device=c_feats.device)

        for m in range(b):
            for n in range(b):
                s_feat = c_feats[m].detach().cpu().numpy()
                f_feat = f_feats[n].detach().cpu().numpy()
                _, path = self.fastdtw(s_feat, f_feat)
                for (i, j) in path:
                    mask[m, n, i, j] = 1
        return mask

c_feats = torch.randn(32, 3, 512)
f_feats = torch.randn(32, 6, 512)
model = fast_dtw()
model(c_feats, f_feats)
