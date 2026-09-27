import os, json, random, glob, re, h5py, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from encoder_decoder_transformer_subject import EncoderDecoderTransformer

ROOT = os.path.dirname(os.path.abspath(__file__))
K, TAU = 30, 25

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

class DynamicH5WindowDataset(Dataset):
    def __init__(self, h5_path, meta_path, stim_window, fmri_window, hrf_delay, stride=1,
                 subject_filter=None, episode_filter=None, subject_map=None):
        self.h5_file = h5py.File(h5_path, "r")
        self.stim_data = self.h5_file["stimuli"]
        self.fmri_data = self.h5_file["fmri"]
        with open(meta_path, "r") as f: self.meta = json.load(f)
        self.stim_window = stim_window; self.fmri_window = fmri_window
        self.hrf_delay = hrf_delay; self.stride = stride
        self.subject_map = subject_map or {"sub-01":0,"sub-02":1,"sub-03":2,"sub-05":3}
        self.subject_filter = set(subject_filter) if subject_filter else None
        self.episode_filter = set(episode_filter) if episode_filter else None
        self.index = []
        for i,m in enumerate(self.meta):
            s = m.get("subject"); ep = m.get("episode"); length = int(m["length"])
            if self.subject_filter and s not in self.subject_filter: continue
            if self.episode_filter and ep not in self.episode_filter: continue
            limit = min(length - self.stim_window, length - self.hrf_delay - self.fmri_window)
            if limit is None or limit < 0: continue
            limit = (limit // self.stride) * self.stride
            for t in range(0, limit + 1, self.stride): self.index.append((i, t))
        if len(self.index) == 0: raise RuntimeError("No training windows found.")

    def __len__(self): return len(self.index)

    def __getitem__(self, idx):
        ep_idx, t = self.index[idx]
        stim = self.stim_data[ep_idx, t : t + self.stim_window]
        fmri = self.fmri_data[ep_idx, t + self.hrf_delay : t + self.hrf_delay + self.fmri_window]
        subj_str = self.meta[ep_idx]["subject"]; subject_id = self.subject_map[subj_str]
        ep = self.meta[ep_idx]["episode"]; pos = t + self.hrf_delay
        return (torch.tensor(stim, dtype=torch.float32),
                torch.tensor(fmri, dtype=torch.float32),
                torch.tensor(subject_id, dtype=torch.long),
                subj_str, ep, pos)

    def close(self): self.h5_file.close()

def build_model(input_dim, num_subjects=1, hidden_dim=512, num_layers=2, nhead=4,
                num_parcels=1000, max_len=1024, use_learnable_bos=True, predict_residual=False):
    return EncoderDecoderTransformer(
        input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, nhead=nhead,
        num_parcels=num_parcels, num_subjects=num_subjects, max_len=max_len,
        use_learnable_bos=use_learnable_bos, predict_residual=predict_residual
    )

def mse_loss(pred, target): return nn.functional.mse_loss(pred, target)

def pearson_corr_loss(pred, target, eps=1e-8):
    pm = pred.mean(dim=-1, keepdim=True); tm = target.mean(dim=-1, keepdim=True)
    pc = pred - pm; tc = target - tm
    num = (pc * tc).sum(dim=-1)
    den = torch.sqrt((pc**2).sum(dim=-1) * (tc**2).sum(dim=-1)) + eps
    corr = num / den
    return 1 - corr.mean()

def train_one_epoch(model, loader, device, optimizer, lambda_pearson, tfr):
    model.train()
    for x, y, sid, _, _, _ in loader:
        x, y, sid = x.to(device), y.to(device), sid.to(device)
        pm, pp = model.forward_autoregressive(x, y, sid, teacher_forcing_ratio=tfr)
        loss = mse_loss(pm, y) + lambda_pearson * pearson_corr_loss(pp, y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

def generate_predictions(model_path, h5_path, meta_path, output_dir, device,
                         stim_window, hrf_delay, tgt_len=1, subject_filter=None,
                         subject_map=None, input_dim=6656, episode_prefix=None):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = build_model(input_dim=input_dim, num_subjects=len(subject_map) if subject_map else 1)
    model.load_checkpoint(model_path); model.to(device); model.eval()
    ep_filter = None
    if episode_prefix is not None:
        with open(meta_path, "r") as f:
            meta_all = json.load(f)
        ep_filter = {m["episode"] for m in meta_all
                     if (subject_filter is None or m["subject"] in subject_filter)
                     and str(m.get("episode","")).startswith(episode_prefix)}
    ds = DynamicH5WindowDataset(h5_path=h5_path, meta_path=meta_path,
                                stim_window=stim_window, fmri_window=tgt_len, hrf_delay=hrf_delay,
                                stride=1, subject_filter=subject_filter, subject_map=subject_map,
                                episode_filter=ep_filter)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    recon = defaultdict(lambda: defaultdict(list))
    with torch.no_grad():
        for x, _, sid, subj_str, ep, pos in loader:
            x = x.to(device); sid = sid.to(device)
            pred = model.generate(x, tgt_len=tgt_len, subject_ids=sid)[0].cpu().numpy()
            key = f"{subj_str[0].replace('sub-','')}_{ep[0]}"; start_pos = pos[0].item()
            for i in range(tgt_len): recon[key][start_pos + i].append(pred[i])
    os.makedirs(output_dir, exist_ok=True)
    for key, frame_dict in recon.items():
        max_len = max(frame_dict.keys()) + 1
        out = np.full((max_len, 1000), np.nan)
        for i, frames in frame_dict.items(): out[i] = np.mean(frames, axis=0)
        np.save(os.path.join(output_dir, f"{key}_pred.npy"), out)
    ds.close(); return output_dir

def _parcel_corrs(gt, pred, eps=1e-8):
    gt = gt - gt.mean(0, keepdims=True)
    pred = pred - pred.mean(0, keepdims=True)
    num = (gt * pred).sum(0)
    den = np.sqrt((gt * gt).sum(0) * (pred * pred).sum(0)) + eps
    r = num / den
    r[np.isinf(r)] = np.nan
    return r

def _mean_fisher_z_over_parcels(r_vec, eps=1e-6):
    r = np.clip(r_vec, -1 + eps, 1 - eps)
    z = np.arctanh(r)
    return float(np.nanmean(z))

def evaluate_friends(prediction_dir, fmri_dir, subjects, trim_front=10, trim_back=5):
    out = {}
    for subject in subjects:
        sid = f"{subject:02d}"
        patt = os.path.join(fmri_dir, f"sub-{sid}", "func", f"sub-{sid}_task-friends*_parcel-1000Par7Net*_bold.h5")
        matches = glob.glob(patt)
        if not matches:
            out[sid] = float('nan'); continue
        fmri_path = matches[0]
        all_gt, all_pred = [], []
        with h5py.File(fmri_path, "r") as f:
            for key in f.keys():
                m = re.search(r"task-(s\d+e\d+[abcd])", key)
                if not m: continue
                ep = f"friends_{m.group(1)}"
                pred_path = os.path.join(prediction_dir, f"{sid}_{ep}_pred.npy")
                if not os.path.exists(pred_path): continue
                gt = f[key][:]
                pred = np.load(pred_path)
                if gt.shape[0] <= (trim_front+trim_back) or pred.shape[0] <= (trim_front+trim_back): continue
                gt = gt[trim_front:-trim_back]; pred = pred[trim_front:-trim_back]
                L = min(len(gt), len(pred))
                if L==0 or gt.shape[1]!=pred.shape[1]: continue
                mask = ~np.isnan(pred).any(axis=1)
                gt, pred = gt[:L][mask], pred[:L][mask]
                if len(gt)>0: all_gt.append(gt); all_pred.append(pred)
        if not all_gt:
            out[sid] = float('nan'); continue
        GT = np.concatenate(all_gt, 0); PR = np.concatenate(all_pred, 0)
        r_parcel = _parcel_corrs(GT, PR)
        out[sid] = _mean_fisher_z_over_parcels(r_parcel)
    vals = [v for v in out.values() if v == v]
    return {"subjects": out, "mean_z": float(np.nanmean(vals)) if vals else float('nan')}

def train_with_val_early_stop(cfg, device, subj_code, pos_tgt, out_json):
    label = f"1-{K}->{pos_tgt}"
    if os.path.exists(out_json):
        with open(out_json, "r") as f:
            prev = json.load(f)
    else:
        prev = {"experiments": {}}
    prev["experiments"].setdefault(label, {})
    prev["experiments"][label].setdefault("friends", {})

    subj_map = {subj_code:0}
    ds = DynamicH5WindowDataset(cfg["train_h5"], cfg["train_meta"],
                                stim_window=K, fmri_window=1, hrf_delay=pos_tgt-1, stride=1,
                                subject_filter=[subj_code], subject_map=subj_map)
    loader = DataLoader(ds, batch_size=cfg["batch"], shuffle=True, num_workers=4)
    model = build_model(input_dim=cfg["input_dim"], num_subjects=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    model.to(device)

    ckpt_tpl = os.path.join(cfg["ckpt_dir"], f"seq2one_{subj_code.replace('-', '')}_epoch{{epoch}}.pt")
    best_z = -np.inf
    best_epoch = 0
    decreasing = False

    for e in range(1, cfg["epochs"] + 1):
        tfr = max(0.0, 1.0 - (e-1) / cfg["epochs"])
        train_one_epoch(model, loader, device, optimizer, cfg["lambda_pearson"], tfr)
        ckpt_path = ckpt_tpl.format(epoch=e)
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        model.save_checkpoint(ckpt_path)

        val_dir = os.path.join(cfg["work_dir"], subj_code, f"val_fix30_p{pos_tgt}_e{e}")
        generate_predictions(ckpt_path, cfg["val_h5"], cfg["val_meta"], val_dir, device.type,
                             stim_window=K, hrf_delay=pos_tgt-1, tgt_len=1,
                             subject_filter=[subj_code], subject_map=subj_map,
                             input_dim=cfg["input_dim"], episode_prefix="friends_")
        subj_id = int(subj_code.split("-")[1])
        fr = evaluate_friends(val_dir, cfg["fmri_dir"], [subj_id], cfg["trim_front"], cfg["trim_back"])
        z_vals = list(fr["subjects"].values())
        cur_z = float(z_vals[0]) if z_vals else float('nan')

        if cur_z == cur_z and cur_z >= best_z:
            best_z = cur_z
            best_epoch = e
            decreasing = False
        else:
            decreasing = True

        print(f"[fix30 pos={pos_tgt}] {subj_code} epoch={e} z={cur_z:.4f} best_z={best_z:.4f} best_epoch={best_epoch}")
        if decreasing:
            break

    ds.close()

    prev["experiments"][label]["friends"][subj_code] = best_z
    friend_vals = [v for s, v in prev["experiments"][label]["friends"].items()
                   if s.startswith("sub-") and v == v]
    prev["experiments"][label]["friends"]["subject_mean_z"] = float(np.nanmean(friend_vals)) if friend_vals else float('nan')

    tmp = out_json + ".tmp"
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(tmp, "w") as f: json.dump(prev, f, indent=2)
    os.replace(tmp, out_json)

    print(f"[SAVE] {label} {subj_code} best_epoch={best_epoch} best_z={best_z:.4f} -> {out_json}")

def run_all(cfg, device, subjects, out_json, positions):
    for pos in positions:
        for subj in subjects:
            train_with_val_early_stop(cfg, device, subj, pos, out_json)

if __name__ == "__main__":
    set_seed(11)
    CONFIG = {
        "input_dim": 6656,
        "batch": 32,
        "epochs": 5,
        "lr": 1e-4,
        "lambda_pearson": 0.15,
        "trim_front": 5,
        "trim_back": 5,
        "ckpt_dir": os.path.join(ROOT, "checkpoints"),       
        "work_dir": os.path.join(ROOT, "outputs_seq2one_30to25"),           
        "fmri_dir": os.path.join(ROOT, "fmri"),
        "train_h5": os.path.join(ROOT, "full_episode_train_data.h5"),
        "train_meta": os.path.join(ROOT, "full_episode_train_meta.json"),
        "val_h5": os.path.join(ROOT, "full_episode_val_data.h5"),
        "val_meta": os.path.join(ROOT, "full_episode_val_meta.json"),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subjects = ["sub-01","sub-02","sub-03","sub-05"]
    OUT_JSON = os.path.join(ROOT, "sup_seq2one_prediction_30to25.json")              
    POSITIONS = [TAU]
    run_all(CONFIG, device, subjects, OUT_JSON, POSITIONS)