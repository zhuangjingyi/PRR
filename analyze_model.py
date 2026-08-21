"""
Model FLOPs & parameter analysis. Standalone — does not modify any project code.
Usage: python analyze_model.py --config_file configs/night600/PRR.yml
       python analyze_model.py --config_file configs/nightreid/PRR.yml
"""

import os, sys, argparse
import torch
import torch.nn as nn
from thop import profile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
# from model import make_model
from model_2vit_worelight import make_model
from datasets import make_dataloader


def analyze(model, input_tensor, name="Model"):
    model.eval()
    device = next(model.parameters()).device
    inp = input_tensor.to(device)
    flops, params = profile(model, inputs=(inp,), verbose=False)
    return flops, params


def analyze_by_module(model, input_tensor):
    """Breakdown FLOPs/params per top-level submodule."""
    model.eval()
    device = next(model.parameters()).device
    inp = input_tensor.to(device)
    results = []
    for n, m in model.named_children():
        try:
            flops, params = profile(m, inputs=(inp,), verbose=False)
            results.append((n, flops, params))
        except:
            results.append((n, 0, sum(p.numel() for p in m.parameters())))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--log_file", type=str, default='./0820_Params',
                        help="Save log to file (default: auto-generated under output_dir)")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    if args.opts: cfg.merge_from_list([o for o in args.opts if o.strip()])
    cfg.freeze()

    did = cfg.MODEL.DEVICE_ID
    if isinstance(did, (tuple, list)): did = ','.join(str(d) for d in did)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(did)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # log file
    interaction_interval = cfg.MODEL.INTERACTION_INTERVAL
    # log_file = args.log_file + "/Frequency{}".format(interaction_interval)
    log_file = args.log_file + "/worelight"
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if log_file is None:
        out_dir = cfg.OUTPUT_DIR or "./output"
        os.makedirs(out_dir, exist_ok=True)
        ds = cfg.DATASETS.NAMES[0] if isinstance(cfg.DATASETS.NAMES, (list, tuple)) else cfg.DATASETS.NAMES
        log_file = os.path.join(out_dir, f"model_analysis_{ds}.txt")

    # data
    _, _, val_loader, num_q, num_cls, cam_n, view_n = make_dataloader(cfg)

    # model
    model = make_model(cfg, num_class=num_cls, camera_num=cam_n, view_num=view_n)
    model.to(dev); model.eval()

    # input
    dummy = torch.randn(1, 3, cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1])
    cam_dummy = torch.zeros(1, dtype=torch.long)
    view_dummy = torch.zeros(1, dtype=torch.long)

    lines = []
    def log(s):
        print(s); lines.append(s)

    log("=" * 70)
    log(f"Model Analysis — {cfg.DATASETS.NAMES}")
    log(f"Config: {args.config_file}")
    log(f"Checkpoint: {cfg.TEST.WEIGHT}")
    log(f"Input size: {cfg.INPUT.SIZE_TRAIN}")
    log("=" * 70)

    # total
    flops, params = analyze(model, dummy, "Full Model")
    log(f"\nFull Model:")
    log(f"  FLOPs : {flops/1e9:.3f} G")
    log(f"  Params: {params/1e6:.3f} M")

    # per-module breakdown
    log(f"\nPer-module breakdown:")
    log(f"  {'Module':<25s} {'FLOPs(G)':>10s} {'Params(M)':>10s} {'%Params':>8s}")
    log(f"  {'-'*53}")
    total_p = sum(p.numel() for p in model.parameters())
    for n, m in model.named_children():
        p_m = sum(p.numel() for p in m.parameters())
        try:
            f_m, _ = profile(m, inputs=(dummy,), verbose=False)
            f_g = f_m / 1e9
        except:
            f_g = 0
        log(f"  {n:<25s} {f_g:>10.3f} {p_m/1e6:>10.3f} {p_m/total_p*100:>7.1f}%")

    # CRM-specific params
    log(f"\nCRM (aid) modules:")
    aid_params = 0
    for n, p in model.named_parameters():
        if 'aid' in n:
            log(f"  {n}: {p.numel():,}")
            aid_params += p.numel()
    log(f"  Total CRM params: {aid_params:,} ({aid_params/total_p*100:.2f}%)")

    # forward signature
    log(f"\nForward signature:")
    log(f"  {model.forward.__code__.co_varnames[:model.forward.__code__.co_argcount]}")

    # write
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\nLog saved → {log_file}")


if __name__ == "__main__":
    main()
