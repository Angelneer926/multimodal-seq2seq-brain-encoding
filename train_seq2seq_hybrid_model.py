import os, json, random, h5py, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from models.encoder_decoder_transformer_subject import EncoderDecoderTransformer

ROOT = os.path.dirname(os.path.abspath(__file__))
STIM_WINDOW, FMRI_WINDOW, HRF_DELAY, STRIDE = 45, 30, 10, 5

SUBJECT_MAP = {
    "sub-01": 0,
    "sub-02": 1,
    "sub-03": 2,
    "sub-05": 3,
}

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

class DynamicH5WindowDataset(Dataset):
    def __init__(self, h5_path, meta_path, stim_window=STIM_WINDOW, fmri_window=FMRI_WINDOW,
                 hrf_delay=HRF_DELAY, stride=STRIDE):
        self.h5_file = h5py.File(h5_path, "r")
        self.stim_data = self.h5_file["stimuli"]
        self.fmri_data = self.h5_file["fmri"]
        with open(meta_path, "r") as f: self.meta = json.load(f)
        self.stim_window = stim_window
        self.fmri_window = fmri_window
        self.hrf_delay = hrf_delay
        self.stride = stride
        self.index = []

        for i, m in enumerate(self.meta):
            length = int(m["length"])
            limit = min(length - stim_window, length - hrf_delay - fmri_window)
            if limit < 0: continue
            limit = (limit // stride) * stride
            self.index.extend((i, t) for t in range(0, limit + 1, stride))

        if not self.index: raise RuntimeError("No training windows found.")

    def __len__(self): return len(self.index)

    def __getitem__(self, idx):
        ep_idx, t = self.index[idx]
        stim = self.stim_data[ep_idx, t:t + self.stim_window]
        fmri = self.fmri_data[
            ep_idx,
            t + self.hrf_delay:t + self.hrf_delay + self.fmri_window,
        ]
        subject_id = SUBJECT_MAP[self.meta[ep_idx]["subject"]]
        return (
            torch.tensor(stim, dtype=torch.float32),
            torch.tensor(fmri, dtype=torch.float32),
            torch.tensor(subject_id, dtype=torch.long),
        )

    def close(self): self.h5_file.close()

def mse_loss(pred, target): return nn.functional.mse_loss(pred, target)

def pearson_corr_loss(pred, target, eps=1e-8):
    pred_centered = pred - pred.mean(dim=-1, keepdim=True)
    target_centered = target - target.mean(dim=-1, keepdim=True)
    numerator = (pred_centered * target_centered).sum(dim=-1)
    denominator = torch.sqrt(
        (pred_centered**2).sum(dim=-1) * (target_centered**2).sum(dim=-1)
    ) + eps
    return 1 - (numerator / denominator).mean()

def train_model(h5_path, meta_path, model, device, checkpoint_dir,
                batch_size=32, num_epochs=10, lr=1e-4, lambda_pearson=0.15):
    dataset = DynamicH5WindowDataset(h5_path, meta_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        total_mse = 0.0
        total_pearson = 0.0
        teacher_forcing_ratio = max(0.0, 1.0 - epoch / num_epochs)

        for x, y, subject_ids in tqdm(loader, desc=f"Epoch {epoch + 1}"):
            x, y, subject_ids = x.to(device), y.to(device), subject_ids.to(device)
            pred_mse, pred_pearson = model.forward_autoregressive(
                x, y, subject_ids, teacher_forcing_ratio=teacher_forcing_ratio
            )
            loss_mse = mse_loss(pred_mse, y)
            loss_pearson = pearson_corr_loss(pred_pearson, y)
            loss = loss_mse + lambda_pearson * loss_pearson

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_mse += loss_mse.item()
            total_pearson += loss_pearson.item()

        avg_mse = total_mse / len(loader)
        avg_pearson = total_pearson / len(loader)

        print(
            f"[Epoch {epoch + 1}] "
            f"MSE: {avg_mse:.4f} | "
            f"Pearson Loss: {avg_pearson:.4f} | "
            f"TF Ratio: {teacher_forcing_ratio:.2f}"
        )

        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"seq2seq_hybrid_epoch{epoch + 1}.pt",
        )
        model.save_checkpoint(checkpoint_path)

    dataset.close()

if __name__ == "__main__":
    set_seed(11)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EncoderDecoderTransformer(
        input_dim=6656,
        hidden_dim=512,
        num_layers=2,
        nhead=4,
        num_parcels=1000,
        num_subjects=4,
        max_len=1024,
        use_learnable_bos=True,
        predict_residual=False,
    )

    train_model(
        h5_path=os.path.join(ROOT, "full_episode_train_data.h5"),
        meta_path=os.path.join(ROOT, "full_episode_train_meta.json"),
        model=model,
        device=device,
        checkpoint_dir=os.path.join(ROOT, "checkpoints"),
        batch_size=32,
        num_epochs=10,
        lr=1e-4,
        lambda_pearson=0.15,
    )