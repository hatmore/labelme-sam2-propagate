#!/usr/bin/env python
"""
用 SAM 2.1 video predictor 把已标注帧的多边形传播到同一摄像头序列的其余帧，
输出标准 labelme .json，直接用 labelme 打开该文件夹微调即可。

典型用法（在 sam2 环境里跑，不是 labelme 环境）：

    # 1) 先在 labelme 里手标 1 帧（或每隔 ~20 帧标一帧），存成 json
    # 2) 传播 + 生成预览图
    python sam2_propagate.py --dir 20260904_148 --preview

    # truck_wall/roof/floor/dock_board 直接照抄种子帧（相机与货柜不动时最稳最快）
    python sam2_propagate.py --dir 20260904_148 --static truck_wall,truck_roof,truck_floor,dock_board

    # 显存吃紧就换 tiny
    python sam2_propagate.py --dir 20260904_148 --model tiny

规则：
  * 目录里**任何已存在且 shapes 非空**的 .json 都被当作种子（conditioning frame）。
  * 只给没有 .json 的帧生成结果；已有 .json 永不覆盖，除非 --overwrite。
  * 两个种子之间的空档，前一半从左边种子正向传播、后一半从右边种子反向传播，
    这样单向漂移距离减半。
  * 按文件名里的 <prefix>_<sec>_<nsec> 分组排序，一个目录里混多个相机也没问题。
"""

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

# ---------------------------------------------------------------- 模型定义

MODELS = {
    # key: (hydra config, 权重文件名, HuggingFace repo)
    "tiny":  ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt", "facebook/sam2.1-hiera-tiny"),
    "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt", "facebook/sam2.1-hiera-small"),
    "base":  ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt", "facebook/sam2.1-hiera-base-plus"),
    "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt", "facebook/sam2.1-hiera-large"),
}
CKPT_DIR = os.path.join(os.path.expanduser("~"), ".cache", "sam2_ckpt")

FRAME_RE = re.compile(r"^(?P<prefix>.+)_(?P<sec>\d+)_(?P<nsec>\d+)$")


def ckpt_urls(model):
    _, name, repo = MODELS[model]
    return [
        f"https://hf-mirror.com/{repo}/resolve/main/{name}",          # 国内 HF 镜像
        f"https://dl.fbaipublicfiles.com/segment_anything_2/092824/{name}",
        f"https://huggingface.co/{repo}/resolve/main/{name}",
    ]


def ensure_ckpt(model):
    """按顺序试多个源，支持断点续传。"""
    import urllib.request
    _, name, _ = MODELS[model]
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = os.path.join(CKPT_DIR, name)
    if os.path.exists(path):
        return path
    tmp = path + ".part"
    for url in ckpt_urls(model):
        for attempt in range(4):
            have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            if have:
                req.add_header("Range", f"bytes={have}-")
            try:
                print(f"[ckpt] {url} (已有 {have/1e6:.1f} MB)")
                with urllib.request.urlopen(req, timeout=60) as r, \
                        open(tmp, "ab" if have and r.status == 206 else "wb") as f:
                    if have and r.status != 206:
                        f.seek(0), f.truncate()
                    total = int(r.headers.get("Content-Length", 0)) + (have if r.status == 206 else 0)
                    done = have if r.status == 206 else 0
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            print(f"\r[ckpt] {done/1e6:7.1f}/{total/1e6:.1f} MB", end="")
                print()
                os.replace(tmp, path)
                return path
            except Exception as e:
                print(f"\n[ckpt] 失败({attempt+1}/4): {e}")
    raise RuntimeError(
        f"权重 {name} 下载失败。可手动下载后放到 {CKPT_DIR}\\{name}，可选源：\n  "
        + "\n  ".join(ckpt_urls(model)))


# ---------------------------------------------------------------- 帧的收集与排序

def collect_sequences(d):
    """返回 {prefix: [(stem, img_path), ...]}，按时间戳排好序。"""
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    groups = defaultdict(list)
    for fn in os.listdir(d):
        stem, ext = os.path.splitext(fn)
        if ext.lower() not in exts:
            continue
        m = FRAME_RE.match(stem)
        if m:
            key = (int(m.group("sec")), int(m.group("nsec")))
            groups[m.group("prefix")].append((key, stem, os.path.join(d, fn)))
        else:
            groups["_misc"].append(((0, 0), stem, os.path.join(d, fn)))
    out = {}
    for prefix, items in groups.items():
        items.sort(key=lambda t: (t[0], t[1]))
        out[prefix] = [(stem, path) for _, stem, path in items]
    return out


# ---------------------------------------------------------------- mask <-> polygon

def poly_to_mask(points, h, w):
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).polygon([(float(x), float(y)) for x, y in points], outline=1, fill=1)
    return np.array(im, dtype=bool)


def mask_to_polygon(mask, min_area=80, max_pts=70, min_pts=8):
    """取最大连通轮廓，用 approxPolyDP 二分 epsilon 控制点数。"""
    m = mask.astype(np.uint8)
    if m.sum() < min_area:
        return None
    k = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None
    peri = cv2.arcLength(c, True)
    # 找「点数 <= max_pts 的最小 epsilon」，即在点数上限内保留最多细节。
    # 不要提前 break：一 break 就停在过粗的近似上。
    lo, hi = 0.0, 0.05
    best = cv2.approxPolyDP(c, hi * peri, True)
    for _ in range(30):
        mid = (lo + hi) / 2
        ap = cv2.approxPolyDP(c, mid * peri, True)
        if len(ap) > max_pts:
            lo = mid
        else:
            best, hi = ap, mid
    if len(best) < min_pts:  # 极简单的形状，二分收敛不到就退回原轮廓
        fine = cv2.approxPolyDP(c, 0.001 * peri, True)
        if min_pts <= len(fine) <= max_pts:
            best = fine
    pts = best.reshape(-1, 2).astype(float)
    if len(pts) < 3:
        return None
    return [[round(float(x), 1), round(float(y), 1)] for x, y in pts]


# ---------------------------------------------------------------- labelme json 读写

AUTO_FLAG = "sam2_auto"          # 旧版本写在 json flags 里的标记，仅作兼容回退
LEDGER = ".sam2_auto.json"      # 台账：stem -> 本脚本写出时 shapes 的哈希


def load_json(json_path):
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def read_shapes(json_path):
    return load_json(json_path).get("shapes", [])


def shapes_hash(shapes):
    return hashlib.sha1(
        json.dumps(shapes, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def load_ledger(d):
    p = os.path.join(d, LEDGER)
    try:
        return load_json(p)
    except Exception:
        return {}


def save_ledger(d, ledger):
    p = os.path.join(d, LEDGER)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=1, sort_keys=True)
    os.replace(tmp, p)


def is_auto(json_path, ledger):
    """本脚本生成、且之后没被人改过的结果。

    判据是内容哈希，不是 json 里的 flags —— labelme 加载时会把 image-level
    flags 灌进 flag dock、保存时再原样写回，所以 flags 标记在人工编辑后依然
    存在，靠它判断会把你改过的帧误判成机器结果。台账缺失时（旧版本产物）才
    退回看 flags。
    """
    stem = os.path.splitext(os.path.basename(json_path))[0]
    try:
        doc = load_json(json_path)
    except Exception:
        return False
    if stem in ledger:
        return ledger[stem] == shapes_hash(doc.get("shapes", []))
    return bool(doc.get("flags", {}).get(AUTO_FLAG))


def write_labelme(json_path, img_path, shapes, h, w, store_data, ledger=None,
                  version="5.10.1"):
    image_data = None
    if store_data:
        with open(img_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
    doc = {
        "version": version,
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(img_path),
        "imageData": image_data,
        "imageHeight": h,
        "imageWidth": w,
    }
    tmp = json_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    os.replace(tmp, json_path)
    if ledger is not None:
        stem = os.path.splitext(os.path.basename(json_path))[0]
        ledger[stem] = shapes_hash(shapes)


def make_shape(label, points, group_id=None, description=""):
    return {
        "label": label,
        "points": points,
        "group_id": group_id,
        "description": description,
        "shape_type": "polygon",
        "flags": {},
        "mask": None,
    }


# ---------------------------------------------------------------- 预览图

def write_preview(img_path, shapes, out_path):
    img = cv2.imread(img_path)
    if img is None:
        return
    overlay = img.copy()
    palette = [(0, 255, 0), (255, 128, 0), (0, 128, 255), (255, 0, 255),
               (0, 255, 255), (255, 255, 0), (128, 0, 255), (0, 0, 255)]
    labels = sorted({s["label"] for s in shapes})
    for s in shapes:
        col = palette[labels.index(s["label"]) % len(palette)]
        pts = np.array(s["points"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(overlay, [pts], col)
        cv2.polylines(img, [pts], True, col, 2)
    img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    for i, lb in enumerate(labels):
        col = palette[i % len(palette)]
        cv2.putText(img, lb, (10, 30 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])


# ---------------------------------------------------------------- 传播

def build_jpeg_dir(frames, idxs, workdir):
    """SAM2 video predictor 要一个 00000.jpg 这样命名的目录。"""
    os.makedirs(workdir, exist_ok=True)
    for i, gi in enumerate(idxs):
        img = Image.open(frames[gi][1]).convert("RGB")
        img.save(os.path.join(workdir, f"{i:05d}.jpg"), quality=95)
    return img.size  # (w, h)


def propagate_chunk(predictor, frames, seed_gi, target_gis, seed_shapes,
                    h, w, min_area, max_pts):
    """从 seed_gi 出发把 seed_shapes 传播到 target_gis（顺序即传播顺序）。"""
    if not target_gis:
        return {}
    order = [seed_gi] + list(target_gis)
    workdir = tempfile.mkdtemp(prefix="sam2frames_")
    try:
        build_jpeg_dir(frames, order, workdir)
        state = predictor.init_state(
            video_path=workdir,
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
        )
        predictor.reset_state(state)
        obj_meta = {}
        for oid, sh in enumerate(seed_shapes, start=1):
            m = poly_to_mask(sh["points"], h, w)
            if m.sum() < min_area:
                continue
            obj_meta[oid] = sh
            predictor.add_new_mask(state, frame_idx=0, obj_id=oid, mask=m)
        if not obj_meta:
            return {}

        results = {}
        for local_idx, obj_ids, logits in predictor.propagate_in_video(state):
            if local_idx == 0:
                continue
            gi = order[local_idx]
            shapes = []
            for j, oid in enumerate(obj_ids):
                if oid not in obj_meta:
                    continue
                mask = (logits[j] > 0.0).cpu().numpy().squeeze()
                poly = mask_to_polygon(mask, min_area=min_area, max_pts=max_pts)
                if poly is None:
                    continue
                src = obj_meta[oid]
                shapes.append(make_shape(src["label"], poly,
                                         src.get("group_id"),
                                         src.get("description", "")))
            results[gi] = shapes
        return results
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def find_frame(frames, needle, what):
    """按 stem 的子串定位一帧，例如只给时间戳 1788424312_531000000。"""
    hits = [i for i, (stem, _) in enumerate(frames) if needle in stem]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        return None
    sys.exit(f"--{what} {needle!r} 在本序列里匹配到 {len(hits)} 帧，请写得更具体")


def process_sequence(predictor, prefix, frames, args, device, ledger):
    d = args.dir
    n = len(frames)

    # ---- 显式指定种子/范围的模式 ----
    if args.seed:
        si = find_frame(frames, args.seed, "seed")
        if si is None:
            print(f"\n[{prefix}] 不含 --seed 指定的帧，跳过。")
            return 0
        lo = find_frame(frames, args.frm, "from") if args.frm else si + 1
        hi = find_frame(frames, args.to, "to") if args.to else n - 1
        if lo is None or hi is None:
            sys.exit("--from/--to 指定的帧不在 --seed 所在的序列里")
        seeds, empties = [si], [i for i in range(lo, hi + 1) if i != si]
        print(f"\n[{prefix}] {n} 帧：指定种子 {frames[si][0][-20:]}，"
              f"目标 {len(empties)} 帧（{frames[lo][0][-20:]} .. {frames[hi][0][-20:]}）")
        if not empties:
            print(f"[{prefix}] 范围内没有待生成帧。")
            return 0
        return run_chunks(predictor, prefix, frames, args, ledger, seeds, empties)

    # ---- 默认模式：所有人工标注帧都是种子 ----
    seeds, empties, n_auto, locked = [], [], 0, 0
    for i, (stem, _) in enumerate(frames):
        jp = os.path.join(d, stem + ".json")
        if not os.path.exists(jp):
            empties.append(i)
        elif is_auto(jp, ledger) and not args.trust_auto:
            n_auto += 1          # 上次自动生成、未经人工确认 -> 重算
            empties.append(i)
        elif read_shapes(jp):
            seeds.append(i)      # 人工标注（或已确认的自动结果）-> 种子
        elif args.overwrite:
            empties.append(i)    # 人工存过的空 json，只有 --overwrite 才动
        else:
            locked += 1

    print(f"\n[{prefix}] {n} 帧：种子 {len(seeds)}，待生成 {len(empties)}"
          f"（其中 {n_auto} 帧是上次自动结果，将重算）"
          + (f"，跳过 {locked} 帧空 json（加 --overwrite 才覆盖）" if locked else ""))
    if not seeds:
        print(f"[{prefix}] 没有种子帧，跳过。先在 labelme 里手标至少一帧。")
        return 0
    if not empties:
        print(f"[{prefix}] 全部已标注，跳过。")
        return 0
    return run_chunks(predictor, prefix, frames, args, ledger, seeds, empties)


def run_chunks(predictor, prefix, frames, args, ledger, seeds, empties):
    d = args.dir

    probe = Image.open(frames[0][1])
    w, h = probe.size

    # 把待生成帧分配给最近的种子，并决定正/反向
    todo_fwd = defaultdict(list)   # seed -> [gi 递增]
    todo_bwd = defaultdict(list)   # seed -> [gi 递减]
    for i in empties:
        prev = max([s for s in seeds if s < i], default=None)
        nxt = min([s for s in seeds if s > i], default=None)
        if prev is None:
            todo_bwd[nxt].append(i)          # 首个种子之前，只能倒推
        elif nxt is None or args.forward_only:
            todo_fwd[prev].append(i)
        elif (i - prev) <= (nxt - i):
            todo_fwd[prev].append(i)
        else:
            todo_bwd[nxt].append(i)
    for k in todo_fwd:
        todo_fwd[k].sort()
    for k in todo_bwd:
        todo_bwd[k].sort(reverse=True)

    static = set(x for x in args.static.split(",") if x)
    written = 0
    for seed in sorted(set(list(todo_fwd) + list(todo_bwd))):
        seed_shapes = read_shapes(os.path.join(d, frames[seed][0] + ".json"))
        track_shapes = [s for s in seed_shapes if s["label"] not in static]
        copy_shapes = [s for s in seed_shapes if s["label"] in static]

        for targets in (todo_fwd.get(seed, []), todo_bwd.get(seed, [])):
            if not targets:
                continue
            res = propagate_chunk(predictor, frames, seed, targets, track_shapes,
                                  h, w, args.min_area, args.max_points)
            for gi in targets:
                shapes = list(res.get(gi, []))
                shapes += [make_shape(s["label"], s["points"], s.get("group_id"),
                                      s.get("description", "")) for s in copy_shapes]
                if not shapes:
                    continue
                stem, img_path = frames[gi]
                jp = os.path.join(d, stem + ".json")
                if os.path.exists(jp) and not is_auto(jp, ledger):
                    if not args.overwrite:
                        continue
                    shutil.copy2(jp, jp + ".bak")   # 人工文件先备份
                write_labelme(jp, img_path, shapes, h, w, args.store_data, ledger)
                written += 1
                if args.preview:
                    write_preview(img_path, shapes,
                                  os.path.join(d, "_preview", stem + ".jpg"))
            print(f"  seed {frames[seed][0][-20:]} -> {len(targets)} 帧 已写出")
    return written


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="SAM2 视频传播 -> labelme json")
    ap.add_argument("--dir", required=True, help="图像+json 所在文件夹")
    ap.add_argument("--model", default="small", choices=list(MODELS))
    ap.add_argument("--only", metavar="PREFIX",
                    help="只处理这一个相机序列，例如 N_camera_2_link。"
                         "同一文件夹里两路相机需要不同参数时用它分开跑")
    ap.add_argument("--static", default="",
                    help="逗号分隔；这些标签直接照抄种子帧不做跟踪，"
                         "例如 truck_wall,truck_roof,truck_floor,dock_board")
    ap.add_argument("--overwrite", action="store_true",
                    help="连人工标注的 json 也覆盖（先备份 .bak）。"
                         "上次自动生成且未经人工修改的结果无需此项，默认就会重算")
    ap.add_argument("--trust-auto", action="store_true",
                    help="把上次的自动结果也当作种子（默认不当，避免误差滚雪球）")
    ap.add_argument("--seed", metavar="FRAME",
                    help="只用这一帧作 prompt（写 stem 或时间戳片段即可），"
                         "忽略其它已标注帧。配合 --from/--to 限定重跑范围")
    ap.add_argument("--from", dest="frm", metavar="FRAME",
                    help="重跑范围起点，默认为 --seed 的下一帧")
    ap.add_argument("--to", metavar="FRAME", help="重跑范围终点（含），默认到序列末尾")
    ap.add_argument("--forward-only", action="store_true",
                    help="每段空档只从前一个种子正推，不从后一个种子倒推。"
                         "当相邻种子的类别集不同（后面才出现新目标）时必须加，"
                         "否则倒推会把还没出现的目标硬塞进前面的帧")
    ap.add_argument("--preview", action="store_true", help="在 _preview/ 下写叠加图便于抽查")
    ap.add_argument("--no-imagedata", dest="store_data", action="store_false",
                    help="json 里不内嵌 base64 图像（文件小很多，labelme 照样能开）")
    ap.add_argument("--min-area", type=int, default=80)
    ap.add_argument("--max-points", type=int, default=70)
    ap.set_defaults(store_data=True)
    args = ap.parse_args()

    if not os.path.isdir(args.dir):
        sys.exit(f"目录不存在: {args.dir}")

    from sam2.build_sam import build_sam2_video_predictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] torch {torch.__version__}  device={device}  model={args.model}")
    if device == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    cfg = MODELS[args.model][0]
    ckpt = ensure_ckpt(args.model)
    predictor = build_sam2_video_predictor(cfg, ckpt, device=device)

    seqs = collect_sequences(args.dir)
    print(f"[scan] {args.dir}: {len(seqs)} 个序列 -> {', '.join(seqs)}")
    if args.only:
        seqs = {k: v for k, v in seqs.items() if args.only in k}
        if not seqs:
            sys.exit(f"--only {args.only!r} 没匹配到任何序列")
        print(f"[scan] --only 过滤后只处理: {', '.join(seqs)}")
    ledger = load_ledger(args.dir)
    total = 0
    try:
        for prefix, frames in sorted(seqs.items()):
            total += process_sequence(predictor, prefix, frames, args, device, ledger)
    finally:
        save_ledger(args.dir, ledger)
    print(f"\n完成：共写出 {total} 个 json。用 labelme 打开 {args.dir} 逐帧过一遍即可。")


if __name__ == "__main__":
    main()
