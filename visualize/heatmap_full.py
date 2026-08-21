"""
Full-feature heatmap visualization: Q / K / V / attention / m maps.
Compares two model checkpoints (with vs without CRM) using normalization
that reveals distributional differences.

Clean heatmap comparison: PRR vs Baseline.
Output per sample (4 figures total):
  1. overview.png      — original image + Q/K/V norm (3 key blocks)
  2. attention.png     — CLS-attn + m (PRR model only)
  3. compare_qkv.png   — Q/K/V: baseline vs PRR, shared scale
  4. compare_attn.png  — attention: baseline vs PRR, shared scale

Usage:
  python visualize/heatmap_full.py \
      --config_file configs/night600/PRR.yml \
      --output_dir ./heatmap/CRMV2/600/ \
      --model_with './0718_complex_enet/600/pos3_neg7_Frequency1/transformer_100.pth' \
      --model_without './0718_complex_enet/baseline_600/2vit_wo weight/pos3_neg7_Frequency1/transformer_100.pth' \
      --num_samples 8
  python visualize/heatmap_full.py \
      --config_file configs/night600/PRR.yml \
      --output_dir ./heatmap/Base/600/ \
      --model_with './0718_complex_enet/600/pos3_neg7_Frequency1/transformer_100.pth' \
      --model_without './0718_complex_enet/baseline_600/baseline/pos1_neg1_Frequency0_Tri1.0/transformer_100.pth' \
      --num_samples 8
      
  python visualize/heatmap_full.py \
    --config_file configs/nightreid/PRR.yml\
    --output_dir ./heatmap/CRMV2/nightreid/ \
    --model_with './0718_complex_enet/nightreid/528wD/pos3_neg1_Frequency1/transformer_110.pth' \
    --model_without './0718_complex_enet/baseline_nightreid/2vit_wo weight/528wD/pos3_neg1_Frequency1/transformer_110.pth' \
    --num_samples 8
  python visualize/heatmap_full.py \
    --config_file configs/nightreid/PRR.yml\
    --output_dir ./heatmap/Base/nightreid/ \
    --model_with './0718_complex_enet/nightreid/528wD/pos3_neg1_Frequency1/transformer_110.pth' \
    --model_without './0718_complex_enet/baseline_nightreid/baseline/pos1_neg1_Frequency0_Tri1.0/transformer_110.pth' \
    --num_samples 8
"""

import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from PIL import Image
import torch, cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import cfg
from model import make_model
from model.backbones.vit_pytorch import Block, Attention
from datasets import make_dataloader

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "figure.dpi": 180, "savefig.dpi": 180,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02})

KEY_BLOCKS = [0, 5, 11]


# ═══════════════ Probe (original working version) ═══════════════
class ModulationProbe:
    def __init__(self, model):
        self.model = model; self.originals = {}; self.captured = {}; self._on = False

    def enable(self):
        if self._on: return
        self.captured.clear()
        for name, mod in self.model.named_modules():
            if isinstance(mod, Attention):
                self.originals[name] = mod.forward
                mod.forward = self._patch(name)
        self._on = True

    def disable(self):
        if not self._on: return
        for name, mod in self.model.named_modules():
            if name in self.originals: mod.forward = self.originals[name]
        self.originals.clear(); self._on = False

    def _patch(self, name):
        cap = self.captured
        # walk model to find the correct Attention module reference
        mod = None
        for n, m in self.model.named_modules():
            if n == name: mod = m; break
        if mod is None: return self.originals[name]

        def patched_forward(x, a=None):
            B, N, C = x.shape
            qkv = mod.qkv(x).reshape(B, N, 3, mod.num_heads, C // mod.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            m = None; q_mod = q
            if a is not None:
                ap = mod.aid(a).view(B, N, mod.num_heads, C // mod.num_heads).permute(0, 2, 1, 3)
                m = (q @ ap.transpose(-2, -1))
                m = torch.mean(m, dim=-1, keepdim=True).softmax(dim=2)
                q_mod = q + m * q
            attn = (q_mod @ k.transpose(-2, -1)) * mod.scale
            attn = attn.softmax(dim=-1)
            cap.setdefault(name, []).append({
                'q': q.detach().cpu(), 'k': k.detach().cpu(), 'v': v.detach().cpu(),
                'attn': attn.detach().cpu(),
                'm': m.detach().cpu() if m is not None else None,
            })
            return mod.proj_drop(mod.proj((mod.attn_drop(attn) @ v).transpose(1, 2).reshape(B, N, C)))
        return patched_forward

    def collect(self):
        d = dict(self.captured); self.captured.clear(); return d


# ═══════════════ helpers ═══════════════
def _spat(t, ph=16, pw=8):
    if t is None: return None
    t = t[:, :, 1:]; t = t.norm(dim=-1) if t.dim() == 4 else t
    return t[0].reshape(t.shape[1], ph, pw).numpy()

def _cls_attn(attn, ph=16, pw=8):
    if attn is None: return None
    return attn[0, :, 0, 1:].reshape(attn.shape[1], ph, pw).numpy()

def _m_spat(m, ph=16, pw=8):
    if m is None: return None
    return m[0].squeeze(-1)[:, 1:].reshape(m.shape[1], ph, pw).numpy()

def _bi(name):
    parts = name.split('.')
    for i, p in enumerate(parts):
        if p == 'blocks' and i+1 < len(parts):
            try: return int(parts[i+1])
            except: pass
    return None

def _ov(img, hm, alpha=0.45, cmap='jet', vmin=None, vmax=None):
    h, w = img.shape[:2]
    arr = np.squeeze(hm.astype(np.float32))
    arr = cv2.resize(arr, (w, h))
    if vmin is None: vmin = arr.min()
    if vmax is None: vmax = arr.max()
    arr = np.clip((arr - vmin) / (vmax - vmin + 1e-12), 0, 1)
    colored = colormaps.get_cmap(cmap)(arr)[:, :, :3]
    colored = (colored * 255).astype(np.uint8)
    return ((img.astype(np.float32)*(1-alpha) + colored.astype(np.float32)*alpha)
            .astype(np.uint8))

def _ov_div(img, delta, alpha=0.45):
    h, w = img.shape[:2]
    d = np.squeeze(delta.astype(np.float32))
    d = cv2.resize(d, (w, h))
    v = max(abs(d.min()), abs(d.max()), 1e-12); d = d / v
    c = colormaps.get_cmap('RdBu')((d+1)/2)[:, :, :3]
    c = (c*255).astype(np.uint8)
    return ((img.astype(np.float32)*(1-alpha) + c.astype(np.float32)*alpha)
            .astype(np.uint8))

def _hide(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)

def _pnorm(arr, lo=2, hi=98):
    if arr is None: return None
    vlo, vhi = np.percentile(arr, lo), np.percentile(arr, hi)
    return np.clip((arr - vlo) / (vhi - vlo + 1e-12), 0, 1)


# ═══════════════ FIGURES ═══════════════
def figure_overview(img_np, data, layers_bi, save_path, tag="", ph=16, pw=8):
    """Col 0=original, Col 1-3=blocks. Row 0-2=Q/K/V."""
    n = len(KEY_BLOCKS) + 1
    fig, axes = plt.subplots(3, n, figsize=(3.6*n, 10), dpi=150)
    if axes.ndim == 1: axes = axes.reshape(3, n)
    # col 0: original
    axes[0, 0].set_title("Original", fontsize=12, fontweight='bold')
    for ri, m in enumerate(['q', 'k', 'v']):
        axes[ri, 0].imshow(img_np); _hide(axes[ri, 0])
        axes[ri, 0].set_ylabel(f"$\|{m.upper()}\|_2$", fontsize=13, fontweight='bold', labelpad=8)
    # blocks
    for ci, bi in enumerate(KEY_BLOCKS):
        axes[0, ci+1].set_title(f"Block {bi}", fontsize=12, fontweight='bold')
        ln = layers_bi.get(bi, None)
        entry = data.get(ln, [{}])[0] if ln else {}
        for ri, m in enumerate(['q', 'k', 'v']):
            ax = axes[ri, ci+1]
            hm = _spat(entry.get(m, None), ph, pw)
            if hm is not None: ax.imshow(_ov(img_np, _pnorm(hm.mean(axis=0))))
            _hide(ax)
    fig.suptitle(f"Q/K/V Spatial Norm — {tag}", fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout(); os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight'); plt.close()
    print(f"  Saved: {os.path.basename(save_path)}")


def figure_attn_m(img_np, data, layers_bi, save_path, ph=16, pw=8):
    """Col 0=original, Col 1-3=blocks. Row 0=attn, Row 1=m."""
    n = len(KEY_BLOCKS) + 1
    fig, axes = plt.subplots(2, n, figsize=(3.6*n, 7), dpi=150)
    if axes.ndim == 1: axes = axes.reshape(2, n)
    axes[0, 0].set_title("Original", fontsize=12, fontweight='bold')
    for ri in range(2):
        axes[ri, 0].imshow(img_np); _hide(axes[ri, 0])
    axes[0, 0].set_ylabel("CLS-Attn", fontsize=12, fontweight='bold', labelpad=8)
    axes[1, 0].set_ylabel("m vector", fontsize=12, fontweight='bold', labelpad=8)
    for ci, bi in enumerate(KEY_BLOCKS):
        axes[0, ci+1].set_title(f"Block {bi}", fontsize=12, fontweight='bold')
        ln = layers_bi.get(bi, None)
        entry = data.get(ln, [{}])[0] if ln else {}
        attn = _cls_attn(entry.get('attn', None), ph, pw)
        if attn is not None: axes[0, ci+1].imshow(_ov(img_np, _pnorm(attn.mean(axis=0))))
        _hide(axes[0, ci+1])
        m = _m_spat(entry.get('m', None), ph, pw)
        if m is not None:
            axes[1, ci+1].imshow(_ov(img_np, _pnorm(m.mean(axis=0)), cmap='hot'))
        else:
            axes[1, ci+1].text(0.5, 0.5, 'no modulation', ha='center', va='center',
                               transform=axes[1, ci+1].transAxes, fontsize=11, color='gray')
        _hide(axes[1, ci+1])
    fig.suptitle("Attention & Modulation (PRR)", fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout(); os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight'); plt.close()
    print(f"  Saved: {os.path.basename(save_path)}")


def figure_compare(img_np, data_a, data_b, la, lb, save_path, ph=16, pw=8):
    """Col 0=original, Col 1-3=blocks. 9 rows: BaseQ/PRRQ/ΔQ/..."""
    n = len(KEY_BLOCKS) + 1
    fig, axes = plt.subplots(9, n, figsize=(3.2*n, 1.8*9), dpi=150)
    if axes.ndim == 1: axes = axes.reshape(9, n)
    # col 0: original
    axes[4, 0].set_title("Original", fontsize=10, fontweight='bold')
    for rr in range(9): axes[rr, 0].imshow(img_np); _hide(axes[rr, 0])
    for mi, m in enumerate(['Q', 'K', 'V']):
        r = mi * 3
        axes[r, 0].set_ylabel(f"Base $\|{m}\|_2$", fontsize=8, fontweight='bold', labelpad=4)
        axes[r+1, 0].set_ylabel(f"PRR $\|{m}\|_2$", fontsize=8, fontweight='bold', labelpad=4)
        axes[r+2, 0].set_ylabel(f"$\Delta${m}", fontsize=8, fontweight='bold', labelpad=4)
    # blocks
    for ci, bi in enumerate(KEY_BLOCKS):
        axes[0, ci+1].set_title(f"Block {bi}", fontsize=10, fontweight='bold')
        e_a = data_a.get(la.get(bi, ''), [{}])[0]
        e_b = data_b.get(lb.get(bi, ''), [{}])[0]
        for mi, measure in enumerate(['q', 'k', 'v']):
            r = mi * 3
            ha = _spat(e_a.get(measure, None), ph, pw)
            hb = _spat(e_b.get(measure, None), ph, pw)
            if ha is not None and hb is not None:
                ham, hbm = ha.mean(axis=0), hb.mean(axis=0)
                combo = np.concatenate([hbm.flatten(), ham.flatten()])
                vlo, vhi = np.percentile(combo, 2), np.percentile(combo, 98)
                axes[r, ci+1].imshow(_ov(img_np, hbm, vmin=vlo, vmax=vhi))
                axes[r+1, ci+1].imshow(_ov(img_np, ham, vmin=vlo, vmax=vhi))
                axes[r+2, ci+1].imshow(_ov_div(img_np, ham - hbm))
            else:
                for rr in [r, r+1, r+2]:
                    axes[rr, ci+1].text(0.5, 0.5, 'N/A', ha='center', va='center',
                                        transform=axes[rr, ci+1].transAxes, color='gray')
            for rr in [r, r+1, r+2]: _hide(axes[rr, ci+1])
    fig.suptitle("Baseline vs PRR  [shared 2-98%, $\Delta$=RdBu]", fontsize=11, fontweight='bold', y=1.005)
    plt.tight_layout(); os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight'); plt.close()
    print(f"  Saved: {os.path.basename(save_path)}")


# ═══════════════ Baseline loading ═══════════════
def _load_baseline_into_prr(prr_model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in ckpt: ckpt = ckpt['state_dict']
    if 'model' in ckpt: ckpt = ckpt['model']
    prr_state = prr_model.state_dict()
    loaded = 0; skipped = 0
    for ckpt_key, ckpt_val in ckpt.items():
        key = ckpt_key.replace('module.', '')
        if key.startswith('tail.'):
            key = 'multi_tail.i_tail.' + key[5:]
        if key in prr_state:
            try: prr_state[key].copy_(ckpt_val); loaded += 1
            except: skipped += 1
        else: skipped += 1
    print(f"  Baseline→PRR: loaded {loaded}, skipped {skipped}")
    return prr_model


# ═══════════════ main ═══════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--model_with", type=str, default=None)
    parser.add_argument("--model_without", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./heatmap_full_results")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    if args.opts and len(args.opts) > 0:
        opts = [o for o in args.opts if o.strip()]
        if opts: cfg.merge_from_list(opts)
    cfg.freeze()

    did = cfg.MODEL.DEVICE_ID
    if isinstance(did, (tuple, list)): did = ','.join(str(d) for d in did)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(did)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    ph = (cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1
    pw = (cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1
    _, _, val_loader, num_q, num_cls, cam_n, view_n = make_dataloader(cfg)

    # Model A: PRR
    ckpt_a = args.model_with or cfg.TEST.WEIGHT
    print(f"[A] PRR  : {ckpt_a}")
    model_a = make_model(cfg, num_class=num_cls, camera_num=cam_n, view_num=view_n)
    model_a.load_param(ckpt_a); model_a.to(dev); model_a.eval()
    probe_a = ModulationProbe(model_a)

    # Model B: Baseline
    ckpt_b = args.model_without
    model_b = None; probe_b = None
    if ckpt_b and os.path.exists(ckpt_b):
        print(f"[B] Base : {ckpt_b}")
        model_b = make_model(cfg, num_class=num_cls, camera_num=cam_n, view_num=view_n)
        _load_baseline_into_prr(model_b, ckpt_b)
        model_b.to(dev); model_b.eval()
        probe_b = ModulationProbe(model_b)
    else:
        if ckpt_b:
            print(f"[B] NOT FOUND: '{os.path.abspath(ckpt_b)}' — using A with mod disabled")
        else:
            print("[B] Not provided — using A with mod disabled")
        model_b = model_a; probe_b = probe_a

    cnt = 0
    for img, pid, camid, camids, tgt_view, imgpath in val_loader:
        if cnt >= args.num_samples: break
        print(f"\n{'='*40}\nSample {cnt+1}")
        img = img.to(dev); camids = camids.to(dev); tgt_view = tgt_view.to(dev)

        # Forward A (CRM ON)
        probe_a.enable()
        with torch.no_grad(): _ = model_a(img, cam_label=camids, view_label=tgt_view)
        data_a = probe_a.collect(); probe_a.disable()

        # Forward B
        if model_b is model_a:
            saved = {}
            for n, m in model_a.named_modules():
                if isinstance(m, Block):
                    saved[n] = m.forward
                    orig = m.forward
                    def mk(old): return lambda x, y=None, _old=old: _old(x, None)
                    m.forward = mk(orig)
            probe_b.enable()
            with torch.no_grad(): _ = model_a(img, cam_label=camids, view_label=tgt_view)
            data_b = probe_b.collect(); probe_b.disable()
            for n, m in model_a.named_modules():
                if n in saved: m.forward = saved[n]
        else:
            saved = {}
            for n, m in model_b.named_modules():
                if isinstance(m, Block):
                    saved[n] = m.forward
                    orig = m.forward
                    def mk(old): return lambda x, y=None, _old=old: _old(x, None)
                    m.forward = mk(orig)
            probe_b.enable()
            with torch.no_grad(): _ = model_b(img, cam_label=camids, view_label=tgt_view)
            data_b = probe_b.collect(); probe_b.disable()
            for n, m in model_b.named_modules():
                if n in saved: m.forward = saved[n]

        # Layer maps (i_tail only for modulation-capable layers)
        def _lm(data):
            it = {_bi(k): k for k in data if _bi(k) is not None and 'i_tail' in k}
            return it if it else {_bi(k): k for k in data if _bi(k) is not None}
        la = _lm(data_a); lb = _lm(data_b)
        print(f"  i_tail layers: A={len(la)}, B={len(lb)}")

        pil = Image.open(imgpath[0]).convert('RGB')
        pil = pil.resize((cfg.INPUT.SIZE_TRAIN[1], cfg.INPUT.SIZE_TRAIN[0]))
        img_np = np.array(pil)
        pid_val = pid[0].item() if isinstance(pid, torch.Tensor) else pid[0]
        tag = f"sample{cnt+1:02d}_id{pid_val:04d}"
        print(f"  Image: {os.path.basename(imgpath[0])}  |  Person ID: {pid_val}")


        figure_overview(img_np, data_a, la,
                        os.path.join(args.output_dir, f"{tag}_overview_PRR.png"),
                        "PRR", ph, pw)
        figure_overview(img_np, data_b, lb,
                        os.path.join(args.output_dir, f"{tag}_overview_baseline.png"),
                        "Baseline", ph, pw)
        figure_attn_m(img_np, data_a, la,
                      os.path.join(args.output_dir, f"{tag}_attn_m.png"), ph, pw)
        if len(la) >= 3 and len(lb) >= 3:
            figure_compare(img_np, data_a, data_b, la, lb,
                           os.path.join(args.output_dir, f"{tag}_compare_qkv.png"), ph, pw)
        cnt += 1
    print(f"\nDone → {args.output_dir}")


if __name__ == "__main__":
    main()
