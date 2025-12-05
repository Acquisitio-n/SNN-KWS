import os, torch, torchaudio, random
from torch.utils.data import Dataset
from pathlib import Path

LABELS_12 = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "silence", "unknown"]
SAMPLE_RATE = 16000
LENGTH = int(SAMPLE_RATE * 1.0)          # 1 秒
TIME_MASK_PARAM = int(0.1 * SAMPLE_RATE)  # 0.1 s

def load_list(txt):
    with open(txt) as f:
        return [l.strip() for l in f if l.strip()]

def time_mask_1d(wav, mask_len):
    """随机把一段连续的采样点置 0（1D 时间掩码）"""
    if wav.numel() <= mask_len:
        return wav
    start = random.randint(0, wav.numel() - mask_len)
    wav = wav.clone()
    wav[start:start + mask_len] = 0.0
    return wav
#   1D 时间掩码：随机把连续 0.1 s（1600 点）置 0，模拟丢包/噪声鲁棒。
#   仅在训练集启用，零计算开销。

class GSC12(Dataset):
    def __init__(self, root, split="training"):
        self.root = Path(root)
        self.split = split

        if split == "training":
            exclude = set(load_list(self.root / "validation_list.txt") +
                          load_list(self.root / "testing_list.txt"))
            files = [f for f in self._all_wav() if f not in exclude]
        elif split == "validation":
            files = load_list(self.root / "validation_list.txt")
        else:
            files = load_list(self.root / "testing_list.txt")
        self.files = files
        self.label2id = {w: i for i, w in enumerate(LABELS_12)}

    def _all_wav(self):
        return [str(p.relative_to(self.root)).replace("\\", "/")
                for p in self.root.rglob("*.wav")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):                        #取样本
        wav, sr = torchaudio.load(self.root / self.files[idx])
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)[0]
        wav = wav[:LENGTH] if wav.numel() >= LENGTH else \
              torch.cat([wav, torch.zeros(LENGTH - wav.numel())])

        # ===== 1D 时间掩码（仅训练集） =====
        if self.split == "training":
            wav = time_mask_1d(wav, TIME_MASK_PARAM)
        # ===================================

        word = self.files[idx].split("/")[0]
        if word not in self.label2id:
            word = "unknown"
        return wav.unsqueeze(0), self.label2id[word]