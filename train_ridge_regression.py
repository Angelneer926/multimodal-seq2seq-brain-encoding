import os, json, re, h5py, numpy as np, torch
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
SUBJECTS = ["sub-01", "sub-02", "sub-03", "sub-05"]
K, TAU = 15, 15
PCA_DIM, TRIM = 512, 5

def load_json(path):
    with open(path, "r") as f: return json.load(f)

def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f: json.dump(obj, f, indent=2)
    os.replace(tmp, path)

def load_alphas(path):
    d = load_json(path)
    sel = d["selection"]
    if sel["best_k"] != K or sel["best_tau"] != TAU:
        raise ValueError(f"Expected k={K}, tau={TAU}, got {sel}")
    exp = d["stage2"]["experiments"][f"tau={TAU}"]["subjects"]
    return {s: float(exp[s]["best_alpha"]) for s in SUBJECTS}

def load_split(name):
    h5 = h5py.File(os.path.join(ROOT, f"new_{name}_full.h5"), "r")
    meta = load_json(os.path.join(ROOT, f"new_{name}_full_meta.json"))
    lengths = [int(m["length"]) for m in meta]
    return h5, meta, lengths

def lagged(z):
    from numpy.lib.stride_tricks import sliding_window_view
    if len(z) < K: return np.empty((0, K * PCA_DIM), np.float32)
    return np.ascontiguousarray(
        sliding_window_view(z, K, axis=0)
        .transpose(0, 2, 1)[:, ::-1]
        .reshape(len(z) - K + 1, K * PCA_DIM),
        dtype=np.float32,
    )

def project(x, pca):
    mean, std, components = pca
    return (((x - mean) / std) @ components.T).astype(np.float32)

@torch.no_grad()
def fit_ridge(train_h5, train_meta, lengths, features, subject, alpha, device):
    fdim = K * PCA_DIM
    parcels = train_h5["fmri"].shape[-1]
    xx = torch.zeros((fdim, fdim), dtype=torch.float32, device=device)
    xy = torch.zeros((fdim, parcels), dtype=torch.float32, device=device)
    sx = np.zeros(fdim, np.float64); sx2 = np.zeros(fdim, np.float64)
    sy = np.zeros(parcels, np.float64); n_total = 0

    for i, m in enumerate(train_meta):
        if m["subject"] != subject: continue
        x_np = lagged(features[str(i)][:])
        n = len(x_np)
        if n == 0: continue
        y_np = np.asarray(train_h5["fmri"][i, TAU-1:TAU-1+n], np.float32)

        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)
        xx.addmm_(x.T, x); xy.addmm_(x.T, y)
        sx += x_np.sum(0, dtype=np.float64)
        sx2 += np.einsum("ij,ij->j", x_np, x_np, dtype=np.float64)
        sy += y_np.sum(0, dtype=np.float64); n_total += n
        del x, y

    mu = sx / n_total
    sd = np.sqrt(np.maximum((sx2 - n_total * mu**2) / (n_total - 1), 0)) + 1e-8
    ymean = sy / n_total

    mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
    sd_t = torch.tensor(sd, dtype=torch.float32, device=device)
    ymean_t = torch.tensor(ymean, dtype=torch.float32, device=device)

    xx.addmm_(mu_t[:, None], mu_t[None], beta=1, alpha=-n_total)
    xx.div_(sd_t[:, None]).div_(sd_t[None]); xx = (xx + xx.T) * 0.5
    xy.addmm_(mu_t[:, None], ymean_t[None], beta=1, alpha=-n_total)
    xy.div_(sd_t[:, None])

    a = xx.clone(); a.diagonal().add_(alpha)
    w = torch.cholesky_solve(xy, torch.linalg.cholesky(a))

    return w, mu.astype(np.float32), sd.astype(np.float32), ymean.astype(np.float32)

class Corr:
    def __init__(self, parcels):
        self.n = 0
        self.sy = np.zeros(parcels); self.sp = np.zeros(parcels)
        self.syy = np.zeros(parcels); self.spp = np.zeros(parcels); self.syp = np.zeros(parcels)

    def add(self, y, p):
        y = y.astype(np.float64); p = p.astype(np.float64)
        self.n += len(y); self.sy += y.sum(0); self.sp += p.sum(0)
        self.syy += (y * y).sum(0); self.spp += (p * p).sum(0); self.syp += (y * p).sum(0)

    def result(self):
        cov = self.syp - self.sy * self.sp / self.n
        vy = np.maximum(self.syy - self.sy**2 / self.n, 0)
        vp = np.maximum(self.spp - self.sp**2 / self.n, 0)
        den = np.sqrt(vy * vp)
        return np.divide(cov, den, out=np.full_like(cov, np.nan), where=den > 1e-12).clip(-1, 1)

def movie_name(ep):
    m = re.fullmatch(r"(?:movie10|movie)_(.*?)(\d+)", ep)
    return m.group(1) if m else ep

def fisher_mean(r, axis=None):
    z = np.arctanh(np.clip(r, -1 + 1e-6, 1 - 1e-6))
    return np.tanh(np.nanmean(z, axis=axis))

@torch.no_grad()
def evaluate(test_h5, test_meta, lengths, pca, subject, w, mu, sd, ymean, device):
    parcels = test_h5["fmri"].shape[-1]
    friends = Corr(parcels)
    movies = defaultdict(lambda: Corr(parcels))

    mu = torch.tensor(mu, device=device)
    sd = torch.tensor(sd, device=device)
    ymean = torch.tensor(ymean, device=device)

    for i, m in enumerate(test_meta):
        if m["subject"] != subject: continue
        ep, length = m["episode"], lengths[i]
        if not ep.startswith(("friends_", "movie10_", "movie_")): continue

        x = np.asarray(test_h5["stimuli"][i, :length], np.float32)
        x_np = lagged(project(x, pca))
        target = np.arange(len(x_np)) + TAU - 1
        keep = (target >= TRIM) & (target < length - TRIM)
        if not keep.any(): continue

        xb = torch.from_numpy(x_np[keep]).to(device)
        pred = (((xb - mu) / sd) @ w + ymean).cpu().numpy()
        gt = np.asarray(test_h5["fmri"][i, target[keep]], np.float32)

        if ep.startswith("friends_"): friends.add(gt, pred)
        else: movies[movie_name(ep)].add(gt, pred)
        del xb

    friends_r = friends.result()
    film_r = {film: acc.result() for film, acc in movies.items()}
    movies_r = fisher_mean(np.stack(list(film_r.values())), axis=0)
    return friends_r, movies_r, film_r

def subject_score(r):
    return float(fisher_mean(r))

def group_score(z):
    z = np.asarray(z, float)
    mean_z = z.mean()
    sem_z = z.std(ddof=1) / np.sqrt(len(z))
    mean_r = np.tanh(mean_z)
    sem_r = (np.tanh(mean_z + sem_z) - np.tanh(mean_z - sem_z)) / 2
    return float(mean_r), float(sem_r)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TUNING = os.path.join(ROOT, "ridge_tuning_run1", "ridge_tuning.json")
    PCA = os.path.join(ROOT, "ridge_tuning_run1", "pca_train_realtr_512d.npz")
    TRAIN_FEATURES = os.path.join(ROOT, "ridge_tuning_run1", "train_pca_unpadded.h5")
    OUTPUT = os.path.join(ROOT, "ridge_final_test")
    os.makedirs(OUTPUT, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT, "parcel_corr", "movies_by_film"), exist_ok=True)

    alphas = load_alphas(TUNING)
    with np.load(PCA, allow_pickle=False) as d:
        pca = (d["mean"], d["std"], d["components"])

    train_h5, train_meta, train_lengths = load_split("train")
    test_h5, test_meta, test_lengths = load_split("test")
    train_features = h5py.File(TRAIN_FEATURES, "r")

    result = {"k": K, "tau": TAU, "subjects": {}}

    try:
        for subject in SUBJECTS:
            w, mu, sd, ymean = fit_ridge(
                train_h5, train_meta, train_lengths, train_features,
                subject, alphas[subject], device,
            )
            fr, mv, films = evaluate(
                test_h5, test_meta, test_lengths, pca,
                subject, w, mu, sd, ymean, device,
            )

            np.save(os.path.join(OUTPUT, "parcel_corr", f"{subject}_friends.npy"), fr)
            np.save(os.path.join(OUTPUT, "parcel_corr", f"{subject}_movies.npy"), mv)
            for film, r in films.items():
                np.save(os.path.join(OUTPUT, "parcel_corr", "movies_by_film", f"{subject}_{film}.npy"), r)

            fr_r, mv_r = subject_score(fr), subject_score(mv)
            result["subjects"][subject] = {
                "alpha": alphas[subject],
                "friends_r": fr_r,
                "movies_r": mv_r,
                "friends_z": float(np.arctanh(fr_r)),
                "movies_z": float(np.arctanh(mv_r)),
            }
            save_json(os.path.join(OUTPUT, "ridge_test_summary.json"), result)
            print(f"{subject}: Friends r={fr_r:.4f}, Movies r={mv_r:.4f}")

            del w
            if device.type == "cuda": torch.cuda.empty_cache()

        friends_z = [result["subjects"][s]["friends_z"] for s in SUBJECTS]
        movies_z = [result["subjects"][s]["movies_z"] for s in SUBJECTS]
        fr_mean, fr_sem = group_score(friends_z)
        mv_mean, mv_sem = group_score(movies_z)

        result["friends"] = {"mean_r": fr_mean, "sem_r": fr_sem}
        result["movies"] = {"mean_r": mv_mean, "sem_r": mv_sem}
        save_json(os.path.join(OUTPUT, "ridge_test_summary.json"), result)

        print(f"\nFriends: r={fr_mean:.4f} ± {fr_sem:.4f}")
        print(f"Movies:  r={mv_mean:.4f} ± {mv_sem:.4f}")

    finally:
        train_features.close()
        train_h5.close()
        test_h5.close()
