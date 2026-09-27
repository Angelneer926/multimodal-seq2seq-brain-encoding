import os, numpy as np, nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap

if not hasattr(mpl_cm, "register_cmap"):
    def register_cmap(name=None, cmap=None, **kwargs):
        if cmap is None: return
        try:
            matplotlib.colormaps.register(cmap, name=name)
        except ValueError:
            pass
    mpl_cm.register_cmap = register_cmap

from nilearn import surface, datasets, plotting
from nilearn.maskers import NiftiLabelsMasker

ROOT = os.path.dirname(os.path.abspath(__file__))

MODEL_TYPE = "seq2seq_hybrid" # seq2one/seq2seq_individual/seq2seq_hybrid/seq2seq_shared
MODE = "friends"  # friends/movies

SUBJECTS = [1, 2, 3, 5]
ATLAS_SUBJECT = 1
OUT_DIR = os.path.join(ROOT, "visualizations")
VMIN, VMAX = 0.0, 0.7

MODEL_NAMES = {
    "seq2one": "Seq2one",
    "seq2seq_individual": "Seq2seq Individual",
    "seq2seq_hybrid": "Seq2seq Hybrid",
    "seq2seq_shared": "Seq2seq Shared",
}

RESULT_DIRS = {
    "seq2one": {
        "friends": os.path.join(ROOT, "results_seq2one_friends"),
        "movies": os.path.join(ROOT, "results_seq2one_movies"),
    },
    "seq2seq_individual": {
        "friends": os.path.join(ROOT, "results_seq2seq_individual_friends"),
        "movies": os.path.join(ROOT, "results_seq2seq_individual_movies"),
    },
    "seq2seq_hybrid": {
        "friends": os.path.join(ROOT, "results_seq2seq_hybrid_friends"),
        "movies": os.path.join(ROOT, "results_seq2seq_hybrid_movies"),
    },
    "seq2seq_shared": {
        "friends": os.path.join(ROOT, "results_seq2seq_shared_friends"),
        "movies": os.path.join(ROOT, "results_seq2seq_shared_movies"),
    },
}

CMAP_POSITIVE = LinearSegmentedColormap.from_list(
    "black_hot",
    [
        (0.00, "#000000"),
        (0.28, "#720000"),
        (0.48, "#C80000"),
        (0.66, "#FF3500"),
        (0.84, "#FFD000"),
        (1.00, "#FFFFFF"),
    ],
)

def load_group_mean_r(result_dir):
    if not os.path.isdir(result_dir): raise FileNotFoundError(result_dir)
    zs = []

    for subject in SUBJECTS:
        path = os.path.join(result_dir, f"sub-{subject:02d}_parcel_z.npy")
        if not os.path.exists(path): raise FileNotFoundError(path)

        z = np.load(path)
        if z.shape != (1000,):
            raise ValueError(f"sub-{subject:02d}: expected (1000,), got {z.shape}")

        zs.append(z)
        print(f"Loaded sub-{subject:02d}")

    return np.tanh(np.nanmean(np.stack(zs), axis=0))

def make_nii_from_parcels(parcel_r):
    atlas_file = (
        f"sub-{ATLAS_SUBJECT:02d}_space-MNI152NLin2009cAsym_"
        "atlas-Schaefer18_parcel-1000Par7Net_desc-dseg_parcellation.nii.gz"
    )
    atlas_path = os.path.join(
        ROOT, "fmri", f"sub-{ATLAS_SUBJECT:02d}", "atlas", atlas_file
    )

    if not os.path.exists(atlas_path): raise FileNotFoundError(atlas_path)

    masker = NiftiLabelsMasker(labels_img=atlas_path)
    masker.fit()
    return masker.inverse_transform(parcel_r.reshape(1, -1))

def plot_surface(parcel_r, title, out_png):
    nii = make_nii_from_parcels(parcel_r)
    fsavg = datasets.fetch_surf_fsaverage("fsaverage5")

    tex_left = surface.vol_to_surf(nii, fsavg.pial_left)
    tex_right = surface.vol_to_surf(nii, fsavg.pial_right)
    tex_left = np.clip(tex_left, VMIN, VMAX)
    tex_right = np.clip(tex_right, VMIN, VMAX)

    print()
    print(f"[{title}] min={np.nanmin(parcel_r):.4f}, max={np.nanmax(parcel_r):.4f}")
    print(f"[{title}] mean={np.nanmean(parcel_r):.4f}, median={np.nanmedian(parcel_r):.4f}")
    print(f"[{title}] negative parcels={np.sum(parcel_r < 0)} / {len(parcel_r)}")

    fig, axes = plt.subplots(
        2, 2, figsize=(6.2, 5.1), dpi=300,
        subplot_kw={"projection": "3d"},
        gridspec_kw={"wspace": -0.45, "hspace": -0.25},
    )

    views = [
        (0, 0, "left", "lateral", "left - lateral", tex_left, fsavg.infl_left, fsavg.sulc_left),
        (0, 1, "right", "lateral", "right - lateral", tex_right, fsavg.infl_right, fsavg.sulc_right),
        (1, 0, "left", "medial", "left - medial", tex_left, fsavg.infl_left, fsavg.sulc_left),
        (1, 1, "right", "medial", "right - medial", tex_right, fsavg.infl_right, fsavg.sulc_right),
    ]

    for r, c, hemi, view, small_title, tex, mesh, sulc in views:
        ax = axes[r, c]
        plotting.plot_surf(
            surf_mesh=mesh,
            surf_map=tex,
            hemi=hemi,
            view=view,
            bg_map=sulc,
            bg_on_data=True,
            darkness=0.30,
            cmap=CMAP_POSITIVE,
            colorbar=False,
            vmin=VMIN,
            vmax=VMAX,
            figure=fig,
            axes=ax,
        )
        ax.set_title(small_title, fontsize=13, y=0.91, pad=0)
        ax.set_axis_off()

    fig.suptitle(title, fontsize=16, y=0.965)

    sm = ScalarMappable(norm=Normalize(vmin=VMIN, vmax=VMAX), cmap=CMAP_POSITIVE)
    sm.set_array([])

    cax = fig.add_axes([0.24, 0.07, 0.52, 0.025])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="max", extendfrac=0.08)
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.7])
    cbar.set_ticklabels(["0", "0.2", "0.4", "0.6", "0.7"])
    cbar.ax.tick_params(labelsize=10, length=2, pad=2)
    cbar.set_label("Pearson r", fontsize=11, labelpad=2)

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.05, top=0.90, wspace=-0.45, hspace=-0.25)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    return nii

if __name__ == "__main__":
    if MODEL_TYPE not in RESULT_DIRS: raise ValueError(f"Unknown model: {MODEL_TYPE}")
    if MODE not in ("friends", "movies"): raise ValueError(f"Unknown mode: {MODE}")

    result_dir = RESULT_DIRS[MODEL_TYPE][MODE]
    title = f"{MODEL_NAMES[MODEL_TYPE]} ({MODE.capitalize()})"

    print()
    print("=" * 72)
    print(f"{MODEL_TYPE.upper()} | {MODE.upper()}")
    print("=" * 72)
    print(f"Results: {result_dir}")

    parcel_r = load_group_mean_r(result_dir)

    print()
    print(f"min r            = {np.nanmin(parcel_r):.4f}")
    print(f"max r            = {np.nanmax(parcel_r):.4f}")
    print(f"mean r           = {np.nanmean(parcel_r):.4f}")
    print(f"median r         = {np.nanmedian(parcel_r):.4f}")
    print(f"negative parcels = {np.sum(parcel_r < 0)} / {len(parcel_r)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    prefix = f"{MODEL_TYPE}_{MODE}_surface_2x2"
    out_png = os.path.join(OUT_DIR, f"{prefix}.png")
    out_nii = os.path.join(OUT_DIR, f"{prefix}.nii.gz")

    nii = plot_surface(parcel_r, title, out_png)
    nib.save(nii, out_nii)

    print()
    print(f"Saved: {out_png}")
    print(f"Saved: {out_nii}")