"""Plotting helpers for the NGC causal-weight heatmap and the case-study figures."""
import json
import os
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FONT_TITLE, FONT_AXIS, FONT_TICK, FONT_LEGEND, FONT_ANNOT = 27, 25, 23, 21, 19
FONT_CBAR_LBL, FONT_CBAR_TICK, TITLE_PAD = 25, 21, 14

DATASET_DOMAIN = {5: "EPF", 6: "EPF", 7: "EPF", 8: "EPF", 9: "EPF", 10: "Energy", 11: "Hydro", 12: "Hydro",
                  13: "Wind", 14: "Wind", 15: "Wind", 16: "Wind"}
_EXO_JSON = {5: "EPF/BE.json", 6: "EPF/DE.json", 7: "EPF/FR.json", 8: "EPF/NP.json", 9: "EPF/PJM.json",
             10: "energy/Energy.json", 11: "hydro/Colbun.json", 12: "hydro/Rapel.json",
             13: "wind/Sdwpf.json", 14: "wind/Sdwpf.json", 15: "wind/Sdwpf.json", 16: "wind/Sdwpf.json"}
EXO_DISPLAY_NAME = {"generation_forecast": "gen_forecast"}


def get_exo_names(dataset_id):
    with open(os.path.join(ROOT, "data", "dataset", _EXO_JSON[dataset_id])) as f:
        return [v["name"] for v in json.load(f)["exo"]]


def _embed_pdf_fonts(pdf_path):
    """Embed fonts with Ghostscript if available (some submission systems require it); otherwise keep the PDF as is."""
    tmp = str(pdf_path) + ".embed_tmp"
    try:
        subprocess.run(["gs", "-dEmbedAllFonts=true", "-dSubsetFonts=true", "-dCompatibilityLevel=1.4",
                        "-dPDFSETTINGS=/prepress", "-dNOPAUSE", "-dBATCH", "-q", "-sDEVICE=pdfwrite",
                        f"-sOutputFile={tmp}", str(pdf_path)], check=True, capture_output=True, timeout=60)
        os.replace(tmp, pdf_path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)


def finalize_figure(fig, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    pdf = os.path.splitext(save_path)[0] + ".pdf"
    fig.savefig(pdf, bbox_inches="tight")
    _embed_pdf_fonts(pdf)
    plt.close(fig)
    print(f"  saved {save_path} (+ .pdf)")


def resample_patches(hm, target_p):
    p = hm.shape[0]
    if p == target_p:
        return hm
    src, dst = np.linspace(0.0, 1.0, p), np.linspace(0.0, 1.0, target_p)
    return np.stack([np.interp(dst, src, hm[:, k]) for k in range(hm.shape[1])], axis=1)


def plot_ngc_heatmap_grid(results, save_path, ncols=2):
    """results: {panel label: (heatmap [n_exo, n_patches], exo_names)}"""
    n = len(results)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.5 * ncols, 7.5 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, (label, (heatmap, exo_names)) in zip(axes, results.items()):
        hmin, hmax = float(heatmap.min()), float(heatmap.max())
        im = ax.imshow((heatmap - hmin) / (hmax - hmin + 1e-12), aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_yticks(range(len(exo_names)))
        ax.set_yticklabels(exo_names, fontsize=FONT_TICK, rotation=30, ha="right", rotation_mode="anchor")
        n_patches = heatmap.shape[1]
        ticks = list(range(0, n_patches, max(1, -(-n_patches // 8))))
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"P{i+1}" for i in ticks], rotation=45, ha="right", fontsize=FONT_TICK)
        ax.set_xlabel("Patch position (time)", fontsize=FONT_AXIS)
        ax.set_title(f"{label}\n(absolute range {hmin:.3f}\u2013{hmax:.3f})", fontsize=FONT_TITLE, pad=TITLE_PAD / 2)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Relative NGC weight\n(min-max per panel)", fontsize=FONT_CBAR_LBL)
        cbar.ax.tick_params(labelsize=FONT_CBAR_TICK)
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout(rect=[0, 0, 1, 1.0], h_pad=4.0, w_pad=3.0)
    finalize_figure(fig, save_path)


def plot_lag_heatmap_pairs(panels, term, save_path):
    """panels: {dataset: (strength [n_exo+1, K+1], row_names, pcmci |MCI| [n_exo+1, 4] or None, pcmci p<0.05 mask)}.
    Left of each pair: learned sparse-Granger group strength per (variable, lag); right: PCMCI+ reference."""
    n = len(panels)
    ncols = 2 if n > 1 else 1                                   # datasets per row; each takes two axes
    nrows = (n + ncols - 1) // ncols
    height = 1.4 + 3.4 * nrows
    widths = ([1, 1, 0.75] * ncols)[:-1]                       # spacer column between dataset pairs for the row labels
    fig = plt.figure(figsize=(8.5 * ncols, height))
    gs = fig.add_gridspec(nrows, len(widths), width_ratios=widths, wspace=0.08, hspace=0.5,
                          left=0.15 if ncols > 1 else 0.3, right=0.99, top=1 - 1.1 / height, bottom=0.35 / height)
    for i, (ds, (m, names, ref, sig)) in enumerate(panels.items()):
        r, c = divmod(i, ncols)
        for j, (mat, title, mask) in enumerate([(m, f"{ds}: MAGNet (sparse Granger)", None),
                                                (ref, f"{ds}: PCMCI+ reference", sig)]):
            ax = fig.add_subplot(gs[r, 3 * c + j])
            if mat is None:
                ax.axis("off")
                continue
            ax.imshow(np.ma.masked_invalid(mat), aspect="auto", cmap="YlOrRd", vmin=0, vmax=np.nanmax(mat))
            ax.set_xticks(range(mat.shape[1]))
            ax.set_xticklabels([str(k) for k in range(mat.shape[1])], fontsize=FONT_TICK * 0.5)
            ax.set_xlabel("lag (patches)", fontsize=FONT_AXIS * 0.45, labelpad=2)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names if j == 0 else [], fontsize=FONT_TICK * 0.5)
            ax.axhline(len(names) - 1.5, color="black", linewidth=1.2)
            for (y, x), v in np.ndenumerate(mat):
                if np.isfinite(v):
                    star = "*" if mask is not None and mask[y, x] else ""
                    ax.text(x, y, f"{v:.2f}{star}", ha="center", va="center", fontsize=FONT_ANNOT * 0.5,
                            color="white" if v > 0.6 * np.nanmax(mat) else "black")
            ax.set_title(title, fontsize=FONT_TITLE * 0.45)
    fig.suptitle(f"Exogenous \u2192 target strength by lag ({term}-term). Left: learned group norm; right: PCMCI+ |MCI| "
                 f"on the training split (* p<0.05). Lag 0 = same patch" + (".\n(SIG v2: lag 0 fitted as a nuisance, not used for the attention weights; "
                 "MAGNet inputs seasonally pre-whitened)." if os.environ.get("MAGNET_LAG0_NUISANCE") == "1" else "."), fontsize=FONT_TITLE * 0.5,
                 y=1 - 0.35 / height)
    finalize_figure(fig, save_path)


def plot_case_studies(windows, ds_name, term, save_path, label="DLinear"):
    n = len(windows)
    ncols, nrows = 2, (n + 1) // 2
    fig = plt.figure(figsize=(10 * ncols, 7.5 * nrows))
    gs = fig.add_gridspec(nrows * 2, ncols, height_ratios=[3, 1.4] * nrows, hspace=0.4, wspace=0.35)
    legend_handles = None
    for i, w in enumerate(windows):
        r, c = divmod(i, ncols)
        ax_pred = fig.add_subplot(gs[r * 2, c])
        ax_exo = fig.add_subplot(gs[r * 2 + 1, c], sharex=ax_pred)
        x = np.arange(len(w["gt"]))
        ax_pred.plot(x, w["gt"], color="black", linewidth=3.4, label="Ground truth")
        ax_pred.plot(x, w["pred_hybrid"], color="green", linewidth=3.2, label=f"MAGNet+{label}")
        ax_pred.plot(x, w["pred_vanilla"], color="red", linestyle="--", linewidth=3.2, label=f"{label} alone")
        ax_pred.set_title(f"Test window {w['window_idx']}  (\u0394|error|={w['improvement']:.3f})", fontsize=FONT_TITLE, pad=TITLE_PAD)
        ax_pred.tick_params(labelbottom=False, labelsize=FONT_TICK)
        if legend_handles is None:
            legend_handles = ax_pred.get_legend_handles_labels()
        exo_series = w["exo_series"]
        exo_label = EXO_DISPLAY_NAME.get(w["exo_name"], w["exo_name"])
        ax_exo.plot(np.arange(len(exo_series)), exo_series, color="steelblue", linewidth=2.2)
        ax_exo.set_ylabel(exo_label, fontsize=FONT_AXIS)
        ax_exo.set_xlabel("Horizon step", fontsize=FONT_AXIS)
        ax_exo.tick_params(labelsize=FONT_TICK)
        ax_exo.text(0.99, 0.06, f"weight={w['exo_weight']:.3f}", transform=ax_exo.transAxes,
                    fontsize=FONT_ANNOT, ha="right", va="bottom", color="gray")
        if len(exo_series) > 1:
            step = int(np.abs(np.diff(exo_series)).argmax()) + 1
            ha, xoff = ("left", 4) if step < len(exo_series) * 0.7 else ("right", -4)
            for ax in (ax_pred, ax_exo):
                ax.axvline(step, color="darkviolet", linestyle=":", linewidth=1.8, alpha=0.9)
            ax_exo.annotate(f"largest {exo_label}\nchange (step {step})", xy=(step, exo_series[step]),
                            xytext=(xoff, 4), textcoords="offset points", fontsize=FONT_ANNOT,
                            color="darkviolet", fontweight="bold", ha=ha)
    handles, labels = legend_handles
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=3, fontsize=FONT_LEGEND, frameon=False)
    fig.subplots_adjust(left=0.1, bottom=0.08, top=0.72)
    finalize_figure(fig, save_path)
