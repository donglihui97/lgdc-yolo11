"""Prepare auditable, group-disjoint YOLO datasets directly from source ZIPs.

Usage: python tools/prepare_data.py --project <dir with data/NEU-DET.zip and data/archive.zip>
Dependencies: Pillow, numpy, scipy. No source archive is modified or extracted.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import math
import random
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.fft import dct

NEU_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
GC_NAMES = ["punching_hole", "welding_line", "crescent_gap", "water_spot", "oil_spot", "silk_spot", "inclusion", "rolled_pit", "crease", "waist_folding"]
GC_LABELS = ["1_chongkong", "2_hanfeng", "3_yueyawan", "4_shuiban", "5_youban", "6_siban", "7_yiwu", "8_yahen", "9_zhehen", "10_yaozhe"]
SPLITS = ("train", "val", "test")
RATIOS = (0.70, 0.15, 0.15)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def pixel_features(content):
    with Image.open(io.BytesIO(content)) as image:
        image.load()
        size, mode = image.size, image.mode
        rgb = image.convert("RGB")
        pixel_hash = hashlib.sha256(str(size).encode() + rgb.tobytes()).hexdigest()
        gray = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
        # Equivalent to imagehash.phash(hash_size=8, highfreq_factor=4).
        low = dct(dct(gray, axis=0), axis=1)[:8, :8]
        bits = (low > np.median(low)).flatten()
        phash = sum(int(bit) << i for i, bit in enumerate(bits))
    return size, mode, pixel_hash, phash


class UnionFind:
    def __init__(self, count):
        self.parent = list(range(count))

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def group_images(records, threshold):
    uf = UnionFind(len(records))
    exact_pairs, near_pairs = [], []
    first_exact = {}
    for i, row in enumerate(records):
        previous = first_exact.setdefault(row["pixel_sha256"], i)
        if previous != i:
            uf.union(previous, i)
            exact_pairs.append([records[previous]["id"], row["id"]])
    for i, row in enumerate(records):
        for j in range(i):
            other = records[j]
            if row["pixel_sha256"] == other["pixel_sha256"]:
                continue
            distance = (row["phash_int"] ^ other["phash_int"]).bit_count()
            if distance <= threshold:
                uf.union(i, j)
                near_pairs.append({"a": other["id"], "b": row["id"], "phash_hamming": distance})
    groups = collections.defaultdict(list)
    for i, row in enumerate(records):
        groups[uf.find(i)].append(i)
    for indices in groups.values():
        group_id = "group_" + min(records[i]["id"] for i in indices)
        for i in indices:
            records[i]["group"] = group_id
    return list(groups.values()), exact_pairs, near_pairs


def integer_targets(count):
    train = int(math.floor(count * RATIOS[0] + 0.5))
    val = int(math.floor(count * RATIOS[1] + 0.5))
    return [train, val, count - train - val]


def stratified_group_split(records, groups, classes, seed):
    """Stratify source primary categories with indivisible duplicate groups.

    Full multilabel boxes are retained. The filename class (NEU) / source folder
    category (GC10) defines the primary stratum. Large duplicate groups are
    allocated first; deterministic randomization breaks equally sized ties.
    """
    vectors = []
    for group in groups:
        v = np.zeros(classes + 1, dtype=np.int64)
        for index in group:
            v[records[index]["primary_class"]] += 1
            v[-1] += 1
        vectors.append(v)
    totals = sum(vectors, start=np.zeros(classes + 1, dtype=np.int64))
    targets = np.asarray([integer_targets(int(n)) for n in totals], dtype=float).T
    current = np.zeros_like(targets)
    indices = list(range(len(groups)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    indices.sort(key=lambda i: (-len(groups[i]), -sum(vectors[i][:-1])))
    for i in indices:
        vector = vectors[i]
        # Class errors and total-count error have equal dimension-normalized influence.
        costs = []
        for split in range(3):
            before = ((current[split] - targets[split]) ** 2 / np.maximum(targets[split], 1)).sum()
            after = ((current[split] + vector - targets[split]) ** 2 / np.maximum(targets[split], 1)).sum()
            costs.append(after - before)
        split = min(range(3), key=lambda s: (costs[s], s))
        current[split] += vector
        for index in groups[i]:
            records[index]["split"] = SPLITS[split]
    return {s: {"target_images": int(targets[i, -1]), "actual_images": int(current[i, -1]),
                "target_primary_class_images": targets[i, :-1].astype(int).tolist(),
                "actual_primary_class_images": current[i, :-1].astype(int).tolist(),
                "actual_class_presence": [sum(c in r["class_ids"] for r in records if r["split"] == s) for c in range(classes)]}
            for i, s in enumerate(SPLITS)}


def write_similarity_sheet(project, key, rows, exact_groups, near_pairs):
    pairs = [(group[0], group[1], "EXACT decoded pixels") for group in exact_groups]
    pairs += [(pair["a"], pair["b"], f"pHash Hamming {pair['phash_hamming']} (candidate)") for pair in near_pairs]
    if not pairs:
        return None
    pairs = pairs[:12]
    lookup = {row["id"]: row for row in rows}
    sheet = Image.new("RGB", (680, 220 * len(pairs)), "white")
    draw = ImageDraw.Draw(sheet)
    output = project / "data" / "processed" / key
    for i, (left, right, reason) in enumerate(pairs):
        top = 220 * i
        draw.text((8, top + 4), f"{key}: {reason}; conservative grouping, manual provenance review open", fill="black")
        for j, identifier in enumerate((left, right)):
            row = lookup[identifier]
            with Image.open(output / row["image"]) as image:
                preview = ImageOps.contain(image.convert("RGB"), (328, 166))
                sheet.paste(preview, (8 + j * 340, top + 43))
            draw.text((8 + j * 340, top + 23), f"{identifier} [{row['split']}]", fill="black")
    path = project / "reports" / f"data_{key}_similarity_review.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return str(path)


def convert_one(zf, xml_name, image_members, label_map, alias_map):
    root = ET.fromstring(zf.read(xml_name))
    image_name = PurePosixPath(xml_name).stem + ".jpg"
    candidates = image_members.get(image_name, [])
    if not candidates:
        raise ValueError("missing_image_for_xml_stem")
    folder = root.findtext("folder", "").strip()
    preferred = [n for n in candidates if PurePosixPath(n).parent.name == folder]
    member = sorted(preferred or candidates)[0]
    content = zf.read(member)
    (width, height), mode, pixel_hash, phash = pixel_features(content)
    redundant = []
    for duplicate in candidates:
        if duplicate != member:
            other = zf.read(duplicate)
            if other != content and pixel_features(other)[2] != pixel_hash:
                raise ValueError("same_filename_different_pixels_requires_manual_review")
            redundant.append(duplicate)
    expected = (int(root.findtext("size/width", "0")), int(root.findtext("size/height", "0")))
    if expected != (width, height):
        raise ValueError(f"xml_image_size_mismatch:{expected}!={(width, height)}")
    boxes, changes = [], []
    for i, obj in enumerate(root.findall("object")):
        raw_name = obj.findtext("name", "").strip()
        name = alias_map.get(raw_name, raw_name)
        if raw_name != name:
            changes.append({"object": i, "rule": "class_alias", "from": raw_name, "to": name})
        if name not in label_map:
            # Reject the whole image; dropping only the unknown box creates false background.
            raise ValueError(f"unknown_class:{raw_name}")
        box = obj.find("bndbox")
        if box is None:
            raise ValueError(f"missing_bbox:object_{i}")
        raw = [float(box.findtext(k, "nan")) for k in ("xmin", "ymin", "xmax", "ymax")]
        if not all(math.isfinite(x) for x in raw):
            raise ValueError(f"nonfinite_bbox:object_{i}")
        # VOC 1-based inclusive coordinates -> zero-based half-open edges.
        edges = [raw[0] - 1, raw[1] - 1, raw[2], raw[3]]
        if edges[2] <= edges[0] or edges[3] <= edges[1]:
            raise ValueError(f"nonpositive_bbox:object_{i}")
        clipped = [max(0, min(width, edges[0])), max(0, min(height, edges[1])),
                   max(0, min(width, edges[2])), max(0, min(height, edges[3]))]
        if clipped != edges:
            changes.append({"object": i, "rule": "clip_to_image", "before_edges": edges, "after_edges": clipped})
        x1, y1, x2, y2 = clipped
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"bbox_empty_after_clip:object_{i}")
        boxes.append([label_map[name], (x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                      (x2 - x1) / width, (y2 - y1) / height])
    if not boxes:
        raise ValueError("no_objects_in_defect_dataset")
    unique_boxes = []
    for i, box in enumerate(boxes):
        if box in unique_boxes:
            changes.append({"object": i, "rule": "remove_exact_duplicate_box"})
        else:
            unique_boxes.append(box)
    primary = (label_map.get(PurePosixPath(xml_name).stem.rsplit("_", 1)[0])
               if "crazing" in label_map else int(PurePosixPath(member).parent.name) - 1)
    if primary is None or primary not in range(len(label_map)):
        raise ValueError("unrecognized_primary_stratum")
    return {"id": PurePosixPath(xml_name).stem, "xml": xml_name, "source_image": member,
            "redundant_image_members": redundant, "width": width, "height": height,
            "source_mode": mode, "pixel_sha256": pixel_hash, "phash_int": phash,
            "class_ids": sorted(set(int(b[0]) for b in unique_boxes)), "boxes": unique_boxes,
            "primary_class": primary, "changes": changes, "source_xml_filename": root.findtext("filename", "")}


def prepare_dataset(project, key, seed, phash_threshold):
    is_neu = key == "neu_det"
    archive = project / "data" / ("NEU-DET.zip" if is_neu else "archive.zip")
    names = NEU_NAMES if is_neu else GC_NAMES
    label_map = {name: i for i, name in enumerate(NEU_NAMES if is_neu else GC_LABELS)}
    aliases = {} if is_neu else {"10_yaozhed": "10_yaozhe"}
    output = project / "data" / "processed" / key
    previous = output / "audit.json"
    source_hash = sha_file(archive)
    config = {"seed": seed, "ratios": list(RATIOS), "phash_hamming_threshold": phash_threshold,
              "source_sha256": source_hash, "converter_version": 2}
    if previous.exists():
        existing = json.loads(previous.read_text(encoding="utf-8"))
        if existing.get("config") != config:
            raise RuntimeError(f"{output} already exists with different settings; use another destination instead of silently changing the split")
    print(f"[{key}] reading {archive.name}", flush=True)
    records, rejected, missing_xml = [], [], []
    with zipfile.ZipFile(archive) as zf:
        image_members = collections.defaultdict(list)
        xml_members = []
        for member in zf.namelist():
            if member.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_members[PurePosixPath(member).name].append(member)
            if member.lower().endswith(".xml"):
                xml_members.append(member)
        raw_classes = collections.Counter()
        source_boxes = 0
        for i, member in enumerate(sorted(xml_members)):
            try:
                root = ET.fromstring(zf.read(member))
                for obj in root.findall("object"):
                    raw_classes[obj.findtext("name", "")] += 1
                    source_boxes += 1
                records.append(convert_one(zf, member, image_members, label_map, aliases))
            except (ValueError, OSError, ET.ParseError) as exc:
                rejected.append({"xml": member, "reason": str(exc)})
            if (i + 1) % 500 == 0:
                print(f"[{key}] inspected {i + 1}/{len(xml_members)} annotations", flush=True)
        xml_stems = {PurePosixPath(n).stem for n in xml_members}
        missing_xml = [n for members in image_members.values() for n in members if PurePosixPath(n).stem not in xml_stems]
        if len({r["id"] for r in records}) != len(records):
            raise RuntimeError("duplicate annotation identifiers")
        print(f"[{key}] auditing decoded exact duplicates and pHash <= {phash_threshold}", flush=True)
        groups, exact_pairs, near_pairs = group_images(records, phash_threshold)
        split_stats = stratified_group_split(records, groups, len(names), seed)
        print(f"[{key}] writing grouped YOLO split {dict(collections.Counter(r['split'] for r in records))}", flush=True)
        for row in records:
            image_path = output / "images" / row["split"] / (row["id"] + ".jpg")
            label_path = output / "labels" / row["split"] / (row["id"] + ".txt")
            image_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(zf.read(row["source_image"]))
            label_path.write_text("".join(str(int(b[0])) + " " + " ".join(f"{v:.9f}" for v in b[1:]) + "\n" for b in row["boxes"]), encoding="utf-8")
            row["image"] = image_path.relative_to(output).as_posix()
            row["label"] = label_path.relative_to(output).as_posix()
    for split in SPLITS:
        split_records = sorted((r for r in records if r["split"] == split), key=lambda r: r["id"])
        (output / f"{split}.txt").write_text("".join((output / r["image"]).as_posix() + "\n" for r in split_records), encoding="utf-8")
        (output / "splits").mkdir(parents=True, exist_ok=True)
        (output / "splits" / f"{split}.txt").write_text("".join(r["id"] + "\n" for r in split_records), encoding="utf-8")
    yaml = f"# Source: {archive.name}; fixed split seed={seed}; grouped duplicate candidates.\npath: {json.dumps(output.as_posix())}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
    yaml += "".join(f"  {i}: {name}\n" for i, name in enumerate(names))
    (output / "data.yaml").write_text(yaml, encoding="utf-8")
    fields = ["id", "split", "group", "image", "label", "source_image", "xml", "width", "height", "source_mode", "pixel_sha256", "phash", "primary_class", "class_ids", "box_count", "source_xml_filename"]
    with (output / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(records, key=lambda r: r["id"]):
            manifest_row = {k: row.get(k, "") for k in fields}
            manifest_row.update(phash=f"{row['phash_int']:016x}", class_ids=";".join(map(str, row["class_ids"])), box_count=len(row["boxes"]))
            writer.writerow(manifest_row)
    violations = []
    for group in groups:
        if len({records[i]["split"] for i in group}) != 1:
            violations.append([records[i]["id"] for i in group])
    class_boxes = {s: [sum(int(box[0]) == c for row in records if row["split"] == s for box in row["boxes"]) for c in range(len(names))] for s in SPLITS}
    duplicate_members = [{"id": r["id"], "selected": r["source_image"], "same_pixel_copies": r["redundant_image_members"]} for r in records if r["redundant_image_members"]]
    exact_groups = [sorted([r["id"] for r in records if r["pixel_sha256"] == h]) for h, count in collections.Counter(r["pixel_sha256"] for r in records).items() if count > 1]
    exact_annotation_conflicts = []
    for ids in exact_groups:
        related = [r for r in records if r["id"] in ids]
        signatures = {tuple(sorted(tuple(b) for b in r["boxes"])) for r in related}
        if len(signatures) > 1:
            exact_annotation_conflicts.append({"ids": ids, "boxes_by_id": {r["id"]: r["boxes"] for r in related}, "action": "Retain original distinct annotation records in one split; no geometric NMS or guessed correction. Flag for manual review."})
    audit = {
        "dataset": key, "created_utc": datetime.now(timezone.utc).isoformat(), "config": config,
        "source_archive": str(archive), "names": names, "class_mapping": label_map,
        "manifest_sha256": sha_file(output / "manifest.csv"),
        "split_list_sha256": {s: sha_file(output / "splits" / f"{s}.txt") for s in SPLITS},
        "source_inventory": {"annotation_files": len(xml_members), "image_archive_members": sum(map(len, image_members.values())), "unique_image_filenames": len(image_members), "raw_boxes": source_boxes, "raw_class_counts": dict(raw_classes)},
        "accepted_images": len(records), "accepted_boxes": sum(len(r["boxes"]) for r in records),
        "rejected_images": rejected, "images_without_xml": missing_xml,
        "split_policy": f"No train/val/test lists in source ZIP; create seed-{seed} 70/15/15 source-primary-category stratification with indivisible exact/pHash candidate groups. Retain all classes and boxes in multilabel XMLs. Approximate ratios are subordinate to group integrity. Validation selects checkpoints; test remains held out. Training seeds never change this dataset split.",
        "stratification": split_stats, "class_box_counts_by_split": class_boxes,
        "source_image_modes": dict(collections.Counter(r["source_mode"] for r in records)),
        "gray_channel_policy": "Preserve original compressed image bytes. Training loader must decode as RGB/BGR with 3 channels; grayscale values are replicated across channels. Decoded RGB pixels are used for exact duplicate hashing.",
        "annotation_policy": {"coordinates": "VOC 1-based inclusive -> zero-based half-open by subtracting 1 only from xmin/ymin, then normalize center/width/height by actual decoded size", "clipping": "Clip box edges to image bounds; log every clipped box. Reject whole image if unknown class, missing/nonfinite/degenerate box, size mismatch, unreadable image, or empty annotations. Do not infer unknown labels.", "class_aliases": aliases, "alias_evidence": "GC10 source ZIP's Defects Description.xlsx explicitly lists 10_yaozhe. 10_yaozhed differs by trailing d; mapped by prespecified spelling correction. Class d is not guessed.", "duplicate_boxes": "Remove exact duplicate class+coordinates only, with logged object index."},
        "annotation_changes": [{"id": r["id"], "changes": r["changes"]} for r in records if r["changes"]],
        "redundant_archive_image_copies": duplicate_members,
        "same_filename_image_copy_policy": "All XML files are parsed independently and all object classes retained. Duplicate image archive members with one shared basename refer to the same shared XML; require decoded pixel equality, then store one selected image. Never infer labels from directory name. Distinct XML identifiers sharing pixels are retained together and annotation disagreements are flagged.",
        "exact_pixel_annotation_conflicts": exact_annotation_conflicts,
        "leakage_audit": {"algorithm": "SHA256 of decoded RGB dimensions+pixels; 64-bit pHash of grayscale 32x32 LANCZOS image, 8x8 low-frequency DCT thresholded at median (includes DC). All-pairs Hamming scan.", "phash_threshold": phash_threshold, "exact_duplicate_groups": exact_groups, "exact_duplicate_pair_count": len(exact_pairs), "near_duplicate_candidate_pair_count": len(near_pairs), "near_duplicate_candidates": near_pairs, "groups": len(groups), "largest_group": max(map(len, groups), default=0), "cross_split_group_violations": violations, "interpretation": "Exact duplicate groups and every pHash candidate <= threshold stay in one split. pHash candidates are conservative similarity flags, not proof of shared provenance. Near-duplicate/manual source-provenance review remains open; this audit does not establish absence of all leakage."},
        "runtime": {"python": sys.version, "numpy": np.__version__},
    }
    audit["similarity_review_contact_sheet"] = write_similarity_sheet(project, key, records, exact_groups, near_pairs)
    write_json(previous, audit)
    write_json(project / "reports" / f"data_{key}_audit.json", audit)
    return audit


def inventory_pcb(project):
    archive = project / "data" / "PKU-Market-PCB.zip"
    classes, dimensions, splits = collections.Counter(), collections.Counter(), {}
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        images = [n for n in names if n.lower().endswith((".jpg", ".jpeg", ".png"))]
        annotations = [n for n in names if n.endswith(".xml")]
        mismatch_examples = []
        for member in annotations:
            root = ET.fromstring(zf.read(member))
            classes.update(o.findtext("name", "") for o in root.findall("object"))
            dimensions[(root.findtext("size/width"), root.findtext("size/height"))] += 1
            if PurePosixPath(root.findtext("filename", "")).stem != PurePosixPath(member).stem and len(mismatch_examples) < 10:
                mismatch_examples.append({"xml": member, "filename_field": root.findtext("filename")})
        for member in names:
            if "/ImageSets/Main/" in member and member.endswith(".txt"):
                splits[PurePosixPath(member).stem] = zf.read(member).decode("utf-8-sig").split()
    def family(stem):
        stem = re.sub(r"^(?:rotation_(?:90|180|270)_|l_)", "", stem)
        return re.sub(r"_\d+_600$", "", stem)
    families = collections.defaultdict(set)
    for split in ("train", "val", "test"):
        for stem in splits.get(split, []):
            families[family(stem)].add(split)
    crossing = [{"family": k, "splits": sorted(v)} for k, v in sorted(families.items()) if len(v) > 1]
    report = {"dataset": "pcb_inventory", "archive": str(archive), "archive_sha256": sha_file(archive), "image_members": len(images), "xml_members": len(annotations), "xml_dimensions": {"x".join(k): v for k, v in dimensions.items()}, "class_boxes": dict(classes), "provided_split_counts": {k: len(v) for k, v in splits.items()}, "filename_field_mismatch_examples": mismatch_examples, "filename_family_rule": "strip rotation_90_/180_/270_ or l_ prefix, then strip crop index _N_600 suffix; filename heuristic, not confirmed provenance", "inferred_original_image_families": len(families), "families_crossing_provided_train_val_test": len(crossing), "crossing_family_examples": crossing[:30], "status": "Inventory only. Package contains pre-cropped and augmented images, unlike the 693 full-resolution originals in the research plan. Existing splits require provenance/group review before use. No extraction or YOLO split produced."}
    write_json(project / "reports" / "data_pcb_inventory.json", report)
    return report


def write_summary(project, reports):
    lines = ["# Dataset preparation report", "", "Source ZIP archives are unchanged; processed images retain original encoded bytes.", ""]
    for report in reports:
        lines += [f"## {report['dataset']}", ""]
        if report["dataset"] == "pcb_inventory":
            lines += [f"{report['image_members']} cropped/augmented images, {report['xml_members']} annotations; inferred original families: {report['inferred_original_image_families']}.", f"Filename-family candidates spanning supplied splits: {report['families_crossing_provided_train_val_test']}. Inventory only; not used in training.", ""]
            continue
        audit = report["leakage_audit"]
        lines += [f"Accepted {report['accepted_images']} images / {report['accepted_boxes']} boxes; rejected {len(report['rejected_images'])} images.", "", "| Split | Images | Primary-category image counts |", "|---|---:|---|"]
        for split, stat in report["stratification"].items():
            lines.append(f"| {split} | {stat['actual_images']} | {stat['actual_primary_class_images']} |")
        lines += ["", f"Class order: {', '.join(report['names'])}.", f"Exact duplicate groups: {len(audit['exact_duplicate_groups'])}; pHash candidate pairs: {audit['near_duplicate_candidate_pair_count']}; largest connected group: {audit['largest_group']}; cross-split candidate-group violations: {len(audit['cross_split_group_violations'])}.", "", audit["interpretation"], "", report["gray_channel_policy"], "", "Coordinates: VOC 1-based inclusive converted to zero-based half-open before YOLO normalization. Every clip/alias/duplicate-box removal is logged in audit JSON.", ""]
        if report.get("similarity_review_contact_sheet"):
            lines += [f"Similarity review contact sheet: `{report['similarity_review_contact_sheet']}`.", ""]
        if report["rejected_images"]:
            lines += ["Rejected annotations:", ""] + [f"- `{r['xml']}`: {r['reason']}" for r in report["rejected_images"]] + [""]
    path = project / "reports" / "data_preparation.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--datasets", nargs="+", choices=["neu_det", "gc10_det", "pcb_inventory"], default=["neu_det", "gc10_det"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phash-threshold", type=int, default=4)
    args = parser.parse_args()
    if not 0 <= args.phash_threshold <= 16:
        parser.error("--phash-threshold must be between 0 and 16")
    reports = []
    for dataset in args.datasets:
        reports.append(inventory_pcb(args.project) if dataset == "pcb_inventory" else prepare_dataset(args.project, dataset, args.seed, args.phash_threshold))
    write_summary(args.project, reports)
    print(json.dumps({r["dataset"]: r.get("accepted_images", r.get("image_members")) for r in reports}), flush=True)


if __name__ == "__main__":
    main()
