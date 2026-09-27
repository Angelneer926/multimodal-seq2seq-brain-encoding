import glob, json, os
from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm

MODALITIES = (
    "language_context",
    "audio_layer9",
    "audio_layer3",
    "language_visual",
    "language_current",
    "visual_avg",
)

def find_stim_path(modality_root, episode):
    pattern = os.path.join(modality_root, "**", episode + ".npy")
    matches = glob.glob(pattern, recursive=True)
    if not matches: raise FileNotFoundError(f"Stimulus not found: {pattern}")
    return matches[0]

def load_fmri_for_episode(h5_path, episode):
    if "friends" in episode:
        target_suffix = "task-" + episode.replace("friends_", "")
        with h5py.File(h5_path, "r") as f:
            matching_keys = [k for k in f.keys() if target_suffix in k]
            if not matching_keys:
                raise KeyError(f"Could not find key matching '{target_suffix}' in {h5_path}")
            return f[matching_keys[0]][:].astype(np.float32)

    elif "movie10" in episode:
        target_suffix = "task-" + episode.replace("movie10_", "")
        with h5py.File(h5_path, "r") as f:
            run1_keys = [k for k in f.keys() if target_suffix in k and "_run-1" in k]
            if run1_keys: return f[run1_keys[0]][:].astype(np.float32)

            no_run_keys = [k for k in f.keys() if target_suffix in k and "_run" not in k]
            if no_run_keys: return f[no_run_keys[0]][:].astype(np.float32)

            raise KeyError(f"Could not find suitable key for '{target_suffix}' in {h5_path}")

    else:
        raise ValueError(f"Unrecognized episode format: {episode}")

def get_h5_path(fmri_root, subject, episode):
    h5_dir = os.path.join(fmri_root, f"sub-{subject:02}", "func")
    all_h5s = glob.glob(os.path.join(h5_dir, "*.h5"))
    for h5_path in all_h5s:
        try:
            load_fmri_for_episode(h5_path, episode)
            return h5_path
        except KeyError:
            continue
    raise FileNotFoundError(f"No HDF5 file contains episode {episode} for subject {subject}")

def load_stim_features(stim_root, episode, modalities):
    features = [
        np.load(find_stim_path(os.path.join(stim_root, modality), episode))
        for modality in modalities
    ]
    min_len = min(feature.shape[0] for feature in features)
    return np.concatenate([feature[:min_len] for feature in features], axis=1)

def load_features(stim_root, fmri_root, subject, episode, modalities):
    stim = load_stim_features(stim_root, episode, modalities)
    h5_path = get_h5_path(fmri_root, subject, episode)
    fmri = load_fmri_for_episode(h5_path, episode)
    return stim, fmri

def load_features_test(stim_root, fmri_root, subject, episode, modalities):
    return load_stim_features(stim_root, episode, modalities)

def append_to_h5(h5_file, stim_data, fmri_data):
    for key, data in (("stimuli", stim_data), ("fmri", fmri_data)):
        if key not in h5_file:
            h5_file.create_dataset(
                key,
                data=data,
                maxshape=(None,) + data.shape[1:],
                chunks=True,
                compression="gzip",
            )
        else:
            old_len = h5_file[key].shape[0]
            new_len = old_len + data.shape[0]
            h5_file[key].resize(new_len, axis=0)
            h5_file[key][old_len:new_len] = data

def save_dataset(data, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, stimuli=data["stimuli"], fmri=data["fmri"])
    with open(save_path.replace(".npz", "_meta.json"), "w") as f:
        json.dump(data["meta"], f)
    print(f"Saved: {save_path}")

def collect_existing_episodes(stim_root, modalities):
    episode_sets = []
    for modality in modalities:
        mod_path = os.path.join(stim_root, modality)
        if not os.path.exists(mod_path): continue

        episodes = set()
        for root, dirs, files in os.walk(mod_path):
            for filename in files:
                if filename.endswith(".npy"):
                    episodes.add(filename.replace(".npy", ""))
        episode_sets.append(episodes)

    return set.intersection(*episode_sets) if episode_sets else set()

def build_full_episode_dataset_h5(subject_to_eps, stim_root, fmri_root, h5_save_path,
                                  meta_save_path, modalities=MODALITIES,
                                  log_missing_path=None, max_len=1200):
    def pad_to_length(x, target_len):
        pad_width = ((0, target_len - x.shape[0]), (0, 0))
        return np.pad(x, pad_width, mode="constant")

    all_meta, missing = [], []
    os.makedirs(os.path.dirname(h5_save_path), exist_ok=True)

    with h5py.File(h5_save_path, "w") as h5_file:
        for subject, episode_list in subject_to_eps.items():
            for episode in tqdm(episode_list, desc=f"sub-{subject:02}"):
                try:
                    stim, fmri = load_features(
                        stim_root, fmri_root, subject, episode, modalities
                    )
                except (FileNotFoundError, KeyError) as e:
                    missing.append((f"sub-{subject:02}", episode, str(e)))
                    continue

                max_frame = min(len(stim), len(fmri))
                stim = stim[:max_frame]
                fmri = fmri[:max_frame]

                if max_frame > max_len:
                    missing.append((
                        f"sub-{subject:02}",
                        episode,
                        f"Episode too long: {max_frame} > {max_len}",
                    ))
                    continue

                stim = pad_to_length(stim, max_len)
                fmri = pad_to_length(fmri, max_len)
                append_to_h5(h5_file, stim[None], fmri[None])

                all_meta.append({
                    "subject": f"sub-{subject:02}",
                    "episode": episode,
                    "length": max_frame,
                })

    with open(meta_save_path, "w") as f:
        json.dump(all_meta, f)
    print(f"Saved meta: {meta_save_path}")

    if log_missing_path:
        with open(log_missing_path, "w") as f:
            json.dump(missing, f, indent=2)
        print(f"Missing entries saved to: {log_missing_path}")

def main(split="train"):
    root = Path(__file__).resolve().parent
    stim_root = root / "stimuli" / "stimulus_features" / "raw"
    fmri_root = root / "fmri"
    subjects = [1, 2, 3, 5]

    available = collect_existing_episodes(stim_root, MODALITIES)

    def filter_existing(episodes):
        return sorted({episode for episode in episodes if episode in available})

    def friends_eps(seasons, start=1, end=25, parts="abcd"):
        return filter_existing([
            f"friends_s0{season}e{ep:02}{part}"
            for season in seasons
            for ep in range(start, end + 1)
            for part in parts
        ])

    def movie_eps(name):
        return filter_existing([f"movie10_{name}{i:02}" for i in range(1, 18)])

    # The test file contains both ID (Friends) and OOD (movies) episodes.
    test_episodes = friends_eps([6], 13, 24) + [
        episode
        for movie in ("bourne", "figures", "life", "wolf")
        for episode in movie_eps(movie)
    ]

    datasets = {
        "train": (
            friends_eps(range(1, 6)),
            "full_episode_train_data.h5",
            "full_episode_train_meta.json",
            "full_episode_train_missing.json",
        ),
        "val": (
            friends_eps([6], 1, 12),
            "full_episode_val_data.h5",
            "full_episode_val_meta.json",
            "full_episode_val_missing.json",
        ),
        "test": (
            filter_existing(test_episodes),
            "full_episode_test_data.h5",
            "full_episode_test_meta.json",
            "full_episode_test_missing.json",
        ),
    }

    if split not in datasets: raise ValueError(f"Unknown split: {split}")

    episodes, h5_name, meta_name, missing_name = datasets[split]
    subject_to_eps = {subject: episodes for subject in subjects}

    print(f"{split.upper()}: {sum(map(len, subject_to_eps.values()))} subject-episodes")
    build_full_episode_dataset_h5(
        subject_to_eps=subject_to_eps,
        stim_root=stim_root,
        fmri_root=fmri_root,
        h5_save_path=root / h5_name,
        meta_save_path=root / meta_name,
        modalities=MODALITIES,
        log_missing_path=root / missing_name,
        max_len=1200,
    )

if __name__ == "__main__":
    main()