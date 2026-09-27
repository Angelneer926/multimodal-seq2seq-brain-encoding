import os, json, glob, re, h5py, numpy as np, torch
from scipy.stats import pearsonr
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from models.encoder_decoder_transformer_subject import EncoderDecoderTransformer as SubjectTransformer
from models.encoder_decoder_transformer_nosubject import EncoderDecoderTransformer as SharedTransformer

ROOT = os.path.dirname(os.path.abspath(__file__))

MODEL_TYPE = "seq2seq_hybrid" # seq2one/seq2seq_individual/seq2seq_hybrid/seq2seq_shared
SUBJECTS = [1, 2, 3, 5]

SUBJECT_MAP = {
    "sub-01": 0,
    "sub-02": 1,
    "sub-03": 2,
    "sub-05": 3,
}

EVAL_H5 = os.path.join(ROOT, "full_episode_test_data.h5")
EVAL_META = os.path.join(ROOT, "full_episode_test_meta.json")
FMRI_DIR = os.path.join(ROOT, "fmri")

INPUT_DIM, TRIM = 6656, 5

MODEL_CONFIGS = {
    "seq2one": dict(stim_window=30, hrf_delay=24, tgt_len=1, stride=1),
    "seq2seq_individual": dict(stim_window=45, hrf_delay=10, tgt_len=30, stride=5),
    "seq2seq_hybrid": dict(stim_window=45, hrf_delay=10, tgt_len=30, stride=5),
    "seq2seq_shared": dict(stim_window=45, hrf_delay=10, tgt_len=30, stride=5),
}

CHECKPOINTS = {
    "seq2one": {
        1: os.path.join(ROOT, "checkpoints", "seq2one_sub01_epoch5.pt"),
        2: os.path.join(ROOT, "checkpoints", "seq2one_sub02_epoch5.pt"),
        3: os.path.join(ROOT, "checkpoints", "seq2one_sub03_epoch5.pt"),
        5: os.path.join(ROOT, "checkpoints", "seq2one_sub05_epoch5.pt"),
    },
    "seq2seq_individual": {
        1: os.path.join(ROOT, "checkpoints", "seq2seq_sub01_epoch12.pt"),
        2: os.path.join(ROOT, "checkpoints", "seq2seq_sub02_epoch12.pt"),
        3: os.path.join(ROOT, "checkpoints", "seq2seq_sub03_epoch12.pt"),
        5: os.path.join(ROOT, "checkpoints", "seq2seq_sub05_epoch12.pt"),
    },
    "seq2seq_hybrid": os.path.join(ROOT, "checkpoints", "seq2seq_hybrid_epoch8.pt"),
    "seq2seq_shared": os.path.join(ROOT, "checkpoints", "seq2seq_shared_epoch10.pt"),
}

OUTPUT_DIRS = {
    "seq2one": {
        "prediction": os.path.join(ROOT, "results_seq2one_prediction"),
        "friends": os.path.join(ROOT, "results_seq2one_friends"),
        "movies": os.path.join(ROOT, "results_seq2one_movies"),
    },
    "seq2seq_individual": {
        "prediction": os.path.join(ROOT, "results_seq2seq_individual_prediction"),
        "friends": os.path.join(ROOT, "results_seq2seq_individual_friends"),
        "movies": os.path.join(ROOT, "results_seq2seq_individual_movies"),
    },
    "seq2seq_hybrid": {
        "prediction": os.path.join(ROOT, "results_seq2seq_hybrid_prediction"),
        "friends": os.path.join(ROOT, "results_seq2seq_hybrid_friends"),
        "movies": os.path.join(ROOT, "results_seq2seq_hybrid_movies"),
    },
    "seq2seq_shared": {
        "prediction": os.path.join(ROOT, "results_seq2seq_shared_prediction"),
        "friends": os.path.join(ROOT, "results_seq2seq_shared_friends"),
        "movies": os.path.join(ROOT, "results_seq2seq_shared_movies"),
    },
}

class EvaluationDataset(Dataset):
    def __init__(self, subjects, stim_window, hrf_delay, tgt_len, stride):
        self.h5_file = h5py.File(EVAL_H5, "r")
        self.stim_data = self.h5_file["stimuli"]
        with open(EVAL_META, "r") as f: self.meta = json.load(f)

        subject_filter = {f"sub-{s:02d}" for s in subjects}
        self.stim_window = stim_window
        self.hrf_delay = hrf_delay
        self.index = []

        for i, m in enumerate(self.meta):
            if m["subject"] not in subject_filter: continue
            length = int(m["length"])
            limit = min(length - stim_window, length - hrf_delay - tgt_len)
            if limit < 0: continue
            limit = (limit // stride) * stride
            self.index.extend((i, t) for t in range(0, limit + 1, stride))

        if not self.index: raise RuntimeError("No evaluation windows found.")

    def __len__(self): return len(self.index)

    def __getitem__(self, idx):
        ep_idx, t = self.index[idx]
        m = self.meta[ep_idx]
        stim = self.stim_data[ep_idx, t:t + self.stim_window]
        return torch.tensor(stim, dtype=torch.float32), m["subject"], m["episode"], t + self.hrf_delay

    def close(self): self.h5_file.close()

def build_model(model_type):
    kwargs = dict(
        input_dim=INPUT_DIM,
        hidden_dim=512,
        num_layers=2,
        nhead=4,
        num_parcels=1000,
        max_len=1024,
        use_learnable_bos=True,
        predict_residual=False,
    )

    if model_type == "seq2seq_shared":
        return SharedTransformer(**kwargs)

    return SubjectTransformer(num_subjects=4, **kwargs)

def load_model(model_type, checkpoint, device):
    if not os.path.exists(checkpoint): raise FileNotFoundError(checkpoint)
    model = build_model(model_type)
    model.load_checkpoint(checkpoint); model.to(device); model.eval()
    return model

def save_predictions(recon, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for key, frame_dict in recon.items():
        out = np.full((max(frame_dict) + 1, 1000), np.nan)
        for i, frames in frame_dict.items(): out[i] = np.mean(frames, axis=0)
        np.save(os.path.join(output_dir, f"{key}_pred.npy"), out)

def run_generation(model, model_type, dataset, output_dir, device):
    cfg = MODEL_CONFIGS[model_type]
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    recon = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        for x, subj_str, ep, pos in loader:
            x = x.to(device)
            subj = subj_str[0]

            if model_type == "seq2seq_shared":
                pred = model.generate(x, tgt_len=cfg["tgt_len"])[0].cpu().numpy()
            else:
                sid = torch.tensor([SUBJECT_MAP[subj]], dtype=torch.long, device=device)
                pred = model.generate(
                    x, tgt_len=cfg["tgt_len"], subject_ids=sid
                )[0].cpu().numpy()

            key = f"{subj.replace('sub-', '')}_{ep[0]}"
            start = pos[0].item()
            for i in range(cfg["tgt_len"]): recon[key][start + i].append(pred[i])

    save_predictions(recon, output_dir)

def generate_predictions(model_type, subjects, output_dir, device):
    cfg = MODEL_CONFIGS[model_type]

    if "individual" in model_type:
        for subject in subjects:
            model = load_model(model_type, CHECKPOINTS[model_type][subject], device)
            dataset = EvaluationDataset(
                [subject], cfg["stim_window"], cfg["hrf_delay"], cfg["tgt_len"], cfg["stride"]
            )
            run_generation(model, model_type, dataset, output_dir, device)
            dataset.close(); del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        return

    model = load_model(model_type, CHECKPOINTS[model_type], device)
    dataset = EvaluationDataset(
        subjects, cfg["stim_window"], cfg["hrf_delay"], cfg["tgt_len"], cfg["stride"]
    )
    run_generation(model, model_type, dataset, output_dir, device)
    dataset.close(); del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def parcel_corrs(gt, pred):
    r = np.full(gt.shape[1], np.nan)

    for parcel in range(gt.shape[1]):
        try:
            r[parcel] = pearsonr(gt[:, parcel], pred[:, parcel])[0]
        except Exception:
            pass

    return r

def r_to_z(r, eps=1e-6):
    return np.arctanh(np.clip(np.asarray(r, dtype=float), -1 + eps, 1 - eps))

def align_segment(gt, pred):
    if len(gt) <= 2 * TRIM or len(pred) <= 2 * TRIM: return None

    gt = gt[TRIM:-TRIM]
    pred = pred[TRIM:-TRIM]
    length = min(len(gt), len(pred))

    if length == 0 or gt.shape[1] != pred.shape[1]: return None

    gt = gt[:length]
    pred = pred[:length]
    valid = ~np.isnan(pred).any(axis=1)

    if not valid.any(): return None
    return gt[valid], pred[valid]

def find_fmri_file(subject, task):
    sid = f"{subject:02d}"
    pattern = os.path.join(
        FMRI_DIR, f"sub-{sid}", "func",
        f"sub-{sid}_task-{task}*_parcel-1000Par7Net*_bold.h5",
    )
    matches = glob.glob(pattern)
    return matches[0] if matches else None

def evaluate_friends(prediction_dir, subjects, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for subject in subjects:
        sid = f"{subject:02d}"
        fmri_path = find_fmri_file(subject, "friends")
        if fmri_path is None: continue

        all_gt, all_pred = [], []

        with h5py.File(fmri_path, "r") as f:
            for key in f.keys():
                m = re.search(r"task-(s\d+e\d+[abcd])", key)
                if not m: continue

                ep = f"friends_{m.group(1)}"
                pred_path = os.path.join(prediction_dir, f"{sid}_{ep}_pred.npy")
                if not os.path.exists(pred_path): continue

                aligned = align_segment(f[key][:], np.load(pred_path))
                if aligned is None: continue

                gt, pred = aligned
                all_gt.append(gt); all_pred.append(pred)

        if not all_gt: continue

        gt = np.concatenate(all_gt, axis=0)
        pred = np.concatenate(all_pred, axis=0)
        parcel_z = r_to_z(parcel_corrs(gt, pred))

        np.save(os.path.join(output_dir, f"sub-{sid}_parcel_z.npy"), parcel_z)

        mean_z = float(np.nanmean(parcel_z))
        results[sid] = mean_z
        print(f"Friends | sub-{sid}: z = {mean_z:.6f}, r = {np.tanh(mean_z):.6f}")

    if results:
        mean_z = float(np.mean(list(results.values())))
        print(f"Friends | mean: z = {mean_z:.6f}, r = {np.tanh(mean_z):.6f}")

def evaluate_movies(prediction_dir, subjects, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for subject in subjects:
        sid = f"{subject:02d}"
        fmri_path = find_fmri_file(subject, "movie10")
        if fmri_path is None: continue

        film_gt, film_pred = defaultdict(list), defaultdict(list)

        with h5py.File(fmri_path, "r") as f:
            for key in f.keys():
                task_match = re.search(r"task-([a-z]+)([0-9]{2,})", key)
                run_match = re.search(r"run-(\d)", key)
                if not task_match or (run_match and run_match.group(1) == "2"): continue

                film, seg = task_match.groups()
                ep = f"movie10_{film}{seg}"
                pred_path = os.path.join(prediction_dir, f"{sid}_{ep}_pred.npy")
                if not os.path.exists(pred_path): continue

                aligned = align_segment(f[key][:], np.load(pred_path))
                if aligned is None: continue

                gt, pred = aligned
                film_gt[film].append(gt); film_pred[film].append(pred)

        film_z = []

        for film in film_gt:
            gt = np.concatenate(film_gt[film], axis=0)
            pred = np.concatenate(film_pred[film], axis=0)
            parcel_z = r_to_z(parcel_corrs(gt, pred))
            film_z.append(parcel_z)

            mean_z = float(np.nanmean(parcel_z))
            print(f"Movies | sub-{sid} | {film}: z = {mean_z:.6f}, r = {np.tanh(mean_z):.6f}")

        if not film_z: continue

        parcel_z = np.nanmean(np.stack(film_z), axis=0)
        np.save(os.path.join(output_dir, f"sub-{sid}_parcel_z.npy"), parcel_z)

        mean_z = float(np.nanmean(parcel_z))
        results[sid] = mean_z
        print(f"Movies | sub-{sid}: z = {mean_z:.6f}, r = {np.tanh(mean_z):.6f}")

    if results:
        mean_z = float(np.mean(list(results.values())))
        print(f"Movies | mean: z = {mean_z:.6f}, r = {np.tanh(mean_z):.6f}")

if __name__ == "__main__":
    if MODEL_TYPE not in MODEL_CONFIGS: raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dirs = OUTPUT_DIRS[MODEL_TYPE]

    generate_predictions(MODEL_TYPE, SUBJECTS, output_dirs["prediction"], device)
    evaluate_friends(output_dirs["prediction"], SUBJECTS, output_dirs["friends"])
    evaluate_movies(output_dirs["prediction"], SUBJECTS, output_dirs["movies"])