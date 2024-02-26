from CLUNet import CLUNet as create_model
import timm
import os
import torch
import cv2
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TransF
from einops import rearrange
import random
from torchvision.utils import draw_segmentation_masks
import nibabel as nib
import matplotlib.pyplot as plt


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def _preprocess(img_path):
    data = np.array(nib.load(img_path).get_fdata(), dtype="float32")
    data_raw = np.nan_to_num(data, neginf=0)
    data_raw = torch.tensor(data_raw)
    data_raw = data_raw[:, 12:-12, :]

    data_img = np.nan_to_num(data, neginf=0)
    data_img = normalization(data_img)
    data_img = torch.tensor(data_img)
    data_img = data_img[:, 12:-12, :]
    data_img = torch.unsqueeze(data_img, dim=0)

    return data_img, data_raw


def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)
    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim

def get_attention_score(self, input, output):
    x = input[0]
    value = self.v(x)
    x = self.f(x)
    x = rearrange(x, "b (e c) w h d-> (b e) c w h d", e=self.heads)
    value = rearrange(value, "b (e c) w h d-> (b e) c w h d", e=self.heads)
    if self.fold_w > 1 and self.fold_h > 1:
        b0, c0, w0, h0, d0 = x.shape
        assert w0 % self.fold_w == 0 and h0 % self.fold_h == 0, \
            f"Ensure the feature map size ({w0}*{h0}) can be divided by fold {self.fold_w}*{self.fold_h}"
        x = rearrange(x, "b c (f1 w) (f2 h) (f3 d)-> (b f1 f2 f3) c w h d", f1=self.fold_w,
                      f2=self.fold_h, f3=self.fold_d)  # [bs*blocks,c,ks[0],ks[1]]
        value = rearrange(value, "b c (f1 w) (f2 h) (f3 d)-> (b f1 f2 f3) c w h d", f1=self.fold_w, f2=self.fold_h,
                          f3=self.fold_d)
    b, c, w, h, d = x.shape
    centers = self.centers_proposal(x)
    value_centers = rearrange(self.centers_proposal(value), 'b c w h d-> b (w h d) c')
    b, c, ww, hh, dd = centers.shape
    sim = torch.sigmoid(
        self.sim_beta +
        self.sim_alpha * (pairwise_cos_sim(
            centers.reshape(b, c, -1).permute(0, 2, 1),
            x.reshape(b, c, -1).permute(0, 2, 1)
        ) + self.alp * self.tanh(-torch.sqrt(torch.cdist(centers.reshape(b, c, -1).permute(0, 2, 1),
                                                         x.reshape(b, c, -1).permute(0, 2, 1)))))
    )
    sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)
    mask = torch.zeros_like(sim)
    mask.scatter_(1, sim_max_idx, 1.)

    mask = mask.reshape(mask.shape[0], mask.shape[1], w, h, d)
    mask = rearrange(mask, "(h0 f1 f2 f3) m w h d-> h0 (f1 f2 f3) m w h d",
                     h0=self.heads, f1=self.fold_w, f2=self.fold_h, f3=self.fold_d)  # [head, (fold*fold),m, w,h]
    mask_list = []
    print('mmmm', mask.shape)
    for i in range(self.fold_w):
        for j in range(self.fold_h):
            for m in range(self.fold_d):
                for k in range(mask.shape[2]):
                    temp = torch.zeros(self.heads, w * self.fold_w, h * self.fold_h, d * self.fold_d)
                    temp[:, i * w:(i + 1) * w, j * h:(j + 1) * h, m * d:(m + 1) * d] = mask[:,
                                                                                       i * self.fold_w + j * self.fold_h + m,
                                                                                       k, :, :, :]
                    mask_list.append(temp.unsqueeze(dim=0))

    mask2 = torch.cat(mask_list, dim=0)
    global attention
    attention = mask2.detach()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_path = r'D:\Codedemo\cluster\vis\NC_057_S_0643.nii'
    stage = 2
    block = -1
    head = 0
    alpha = 0.5
    global attention
    image, raw_image = _preprocess(image_path)
    # batch size
    image = image.unsqueeze(dim=0)
    model = create_model()
    weights_dict = torch.load(r'D:\Codedemo\cluster\vis\best_model1.pth',
                              map_location=device)
    model.load_state_dict(weights_dict, strict=False)
    model.network[stage * 2][block].token_mixer.register_forward_hook(get_attention_score)
    out = model(image)
    if type(out) is tuple:
        out = out[0]
    possibility = torch.softmax(out, dim=1).max()
    value, index = torch.max(out, dim=1)
    print(f'==> possibility: {possibility * 100:.3f}%')
    print(f'==> value: {value}%')
    print(f'==> index: {index}%')

    attention = attention[:, head, :, :, :]
    mask = attention.unsqueeze(dim=0)
    mask = F.interpolate(mask, (image.shape[-3], image.shape[-2], image.shape[-1]))
    mask = mask.squeeze(dim=0)
    mask = mask > 0.5
    colors = ["#F8F4EC", "#D63484", "#80BCBD", "#D5F0C1", "#F9F7C9", "#BF3131", "#A367B1", "#7ED7C1",
              "#B06161", "#2D9596", "#83A2FF", "#F4E869", "#596FB7", "#AC87C5", "#DF826C", "#86B6F6"]
    if mask.shape[0] == 4:
        colors = colors[0:4]
    if mask.shape[0] > 4:
        colors = colors * (mask.shape[0] // 8)
        random.seed(3401)
        random.shuffle(colors)

    raw_image = np.array(nib.load(r'D:\Codedemo\cluster\vis\NC_057_S_0643.nii').get_fdata(), dtype="float32")
    raw_image = np.nan_to_num(raw_image, neginf=0)
    raw_image = raw_image[41, :, :]
    output_path = r"D:\Codedemo\cluster\vis\2.png"
    plt.axis('off')
    plt.margins(0, 0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.imsave(output_path, raw_image, cmap='gray')
    plt.imshow(raw_image, cmap='gray')
    plt.show()

    raw_image = torch.tensor(raw_image)
    raw_image = raw_image[12:-12, :]
    print('ii', raw_image.shape)
    raw_image = raw_image.unsqueeze(dim=0)
    raw_image = torch.cat([raw_image] * 3, dim=0)
    mask = mask[:, 41, :, :]
    print('mask', mask.shape)

    img_with_masks = draw_segmentation_masks((raw_image).to(torch.uint8), masks=mask, alpha=alpha, colors=colors)
    img_with_masks = img_with_masks.detach()
    img_with_masks = TransF.to_pil_image(img_with_masks)
    img_with_masks = np.asarray(img_with_masks)
    save_path = r"D:\Codedemo\cluster\vis\1.png"
    cv2.imwrite(save_path, img_with_masks)
    print(f"==> Generated image is saved to: {save_path}")


if __name__ == '__main__':
    main()
