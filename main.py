import os
import json
import math
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import networkx as nx

import insightface
from insightface.app import FaceAnalysis


# -----------------------------
# Config
# -----------------------------
IMAGES_DIR = Path("/Volumes/Extreme SSD/Highlight")          # input album folder
OUT_DIR = Path("/Volumes/Extreme SSD/faces_output_Highlight")       # output root
CLUSTERS_DIR = OUT_DIR / "clusters"

# Face/embedding extraction
DET_SIZE = (640, 640)
DET_THRESH = 0.5
MIN_FACE_SIZE = 40  # px (min(w,h) for bbox)
CTX_ID = 0          # 0=GPU, -1=CPU

# Graph + Chinese Whispers params
CW_THRESHOLD = 0.58            # cosine distance threshold to CREATE edge (lower => stricter)
CW_ITERATIONS = 20

# Refinement params (optional but was in the "advanced refinement" version)
SPLIT_VARIANCE_THRESHOLD = 0.35   # higher => tolerate more diversity inside a cluster
MERGE_CENTROID_THRESHOLD = 0.50   # cosine distance between cluster centroids to MERGE (lower => stricter)

# Output options
COPY_FACE_THUMBS = True   # save face crops into cluster folders
THUMB_SIZE = 256


# -----------------------------
# Helpers
# -----------------------------
def ensure_clean_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    # assumes normalized vectors but remains safe
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(1.0 - np.dot(a, b))


def crop_face(img_bgr: np.ndarray, bbox: np.ndarray, pad: float = 0.15) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = bbox.astype(float)

    bw = x2 - x1
    bh = y2 - y1
    x1 -= pad * bw
    y1 -= pad * bh
    x2 += pad * bw
    y2 += pad * bh

    x1 = int(max(0, math.floor(x1)))
    y1 = int(max(0, math.floor(y1)))
    x2 = int(min(w, math.ceil(x2)))
    y2 = int(min(h, math.ceil(y2)))

    return img_bgr[y1:y2, x1:x2].copy()


def resize_keep_aspect(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return img
    scale = max_side / float(s)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


# -----------------------------
# Chinese Whispers
# -----------------------------
def chinese_whispers_clustering(
    G: nx.Graph,
    iterations: int = 20,
    weight_key: str = "weight",
    seed: int = 42,
) -> Dict[int, int]:
    """
    Returns: node_id -> label_id
    """
    rng = np.random.default_rng(seed)

    # init each node with its own label
    for n in G.nodes:
        G.nodes[n]["label"] = int(n)

    nodes = list(G.nodes)

    for _ in range(iterations):
        rng.shuffle(nodes)
        for n in nodes:
            # collect weighted votes from neighbor labels
            votes: Dict[int, float] = {}
            for nb, attrs in G[n].items():
                w = float(attrs.get(weight_key, 1.0))
                lab = G.nodes[nb]["label"]
                votes[lab] = votes.get(lab, 0.0) + w

            if not votes:
                continue

            # choose the label with maximum vote; break ties randomly
            max_vote = max(votes.values())
            best = [lab for lab, v in votes.items() if v == max_vote]
            G.nodes[n]["label"] = int(rng.choice(best))

    return {int(n): int(G.nodes[n]["label"]) for n in G.nodes}


# -----------------------------
# Refinements (split / merge)
# -----------------------------
def cluster_centroid(embs: np.ndarray) -> np.ndarray:
    c = np.mean(embs, axis=0)
    return c / (np.linalg.norm(c) + 1e-12)


def intra_cluster_variance(embs: np.ndarray) -> float:
    c = cluster_centroid(embs)
    d = [cosine_distance(e, c) for e in embs]
    return float(np.mean(d))


def split_impure_clusters(
    labels: np.ndarray,
    embeddings: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    If a cluster is too diverse (high mean distance to centroid),
    re-run CW inside that cluster with a stricter threshold.
    """
    new_labels = labels.copy()
    next_label = int(new_labels.max() + 1) if new_labels.size else 0

    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        if idx.size < 3:
            continue
        embs = embeddings[idx]
        var = intra_cluster_variance(embs)
        if var <= threshold:
            continue

        # build subgraph with stricter edge threshold
        subG = nx.Graph()
        for i_local, i_global in enumerate(idx):
            subG.add_node(i_local, global_index=int(i_global))
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                d = cosine_distance(embeddings[idx[i]], embeddings[idx[j]])
                if d < CW_THRESHOLD * 0.95:  # slightly stricter than global
                    # weight = similarity for voting
                    subG.add_edge(i, j, weight=(1.0 - d))

        sub_map = chinese_whispers_clustering(subG, iterations=max(10, CW_ITERATIONS // 2))
        sub_labels = np.array([sub_map[i] for i in range(len(idx))], dtype=np.int64)

        # relabel subclusters into global label space
        remap = {}
        for sl in np.unique(sub_labels):
            remap[int(sl)] = next_label
            next_label += 1

        for k, i_global in enumerate(idx):
            new_labels[i_global] = remap[int(sub_labels[k])]

    return new_labels


def merge_close_clusters(
    labels: np.ndarray,
    embeddings: np.ndarray,
    merge_thresh: float,
) -> np.ndarray:
    labs = list(map(int, np.unique(labels)))
    if len(labs) <= 1:
        return labels

    # compute centroids
    centroids = {}
    for lab in labs:
        idx = np.where(labels == lab)[0]
        centroids[lab] = cluster_centroid(embeddings[idx])

    # union-find merge
    parent = {lab: lab for lab in labs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            a, b = labs[i], labs[j]
            d = cosine_distance(centroids[a], centroids[b])
            if d < merge_thresh:
                union(a, b)

    # apply merges
    root_map = {}
    next_id = 0
    merged = labels.copy()
    for i in range(merged.shape[0]):
        r = find(int(merged[i]))
        if r not in root_map:
            root_map[r] = next_id
            next_id += 1
        merged[i] = root_map[r]

    return merged


# -----------------------------
# Main pipeline
# -----------------------------
@dataclass
class FaceRecord:
    face_id: str
    image_path: str
    det_score: float
    bbox: np.ndarray           # (4,)
    embedding: np.ndarray      # (512,)
    thumb_path: str = ""


def extract_faces_and_embeddings(images_dir: Path) -> List[FaceRecord]:
    app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=CTX_ID, det_thresh=DET_THRESH, det_size=DET_SIZE)  # InsightFace usage [web:26]

    records: List[FaceRecord] = []
    img_paths = sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]])

    for p in tqdm(img_paths, desc="Detect + embed"):
        img = cv2.imread(str(p))
        if img is None:
            continue

        faces = app.get(img)
        for f in faces:
            bbox = f.bbox.astype(np.float32)  # [x1,y1,x2,y2]
            x1, y1, x2, y2 = bbox
            if min((x2 - x1), (y2 - y1)) < MIN_FACE_SIZE:
                continue

            emb = f.embedding.astype(np.float32)
            emb = l2_normalize(emb)

            rec = FaceRecord(
                face_id=str(uuid.uuid4())[:8],
                image_path=str(p),
                det_score=float(f.det_score),
                bbox=bbox,
                embedding=emb,
            )
            records.append(rec)

    return records


def build_similarity_graph(embeddings: np.ndarray, threshold: float) -> nx.Graph:
    """
    Edge created if cosine_distance < threshold.
    Weight uses similarity (1 - distance) so stronger edges vote more.
    """
    n = embeddings.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # O(n^2) reference build (OK for smaller sets). For huge N, switch to ANN (Faiss/HNSW).
    for i in tqdm(range(n), desc="Build graph"):
        ei = embeddings[i]
        for j in range(i + 1, n):
            d = cosine_distance(ei, embeddings[j])
            if d < threshold:
                G.add_edge(i, j, weight=(1.0 - d))
    return G


def write_outputs(
    records: List[FaceRecord],
    labels: np.ndarray,
):
    ensure_clean_dir(OUT_DIR)
    ensure_clean_dir(CLUSTERS_DIR)

    # save thumbs + cluster folders
    if COPY_FACE_THUMBS:
        for rec, lab in tqdm(list(zip(records, labels)), total=len(records), desc="Write thumbs"):
            img = cv2.imread(rec.image_path)
            if img is None:
                continue
            face_crop = crop_face(img, rec.bbox, pad=0.15)
            face_crop = resize_keep_aspect(face_crop, THUMB_SIZE)

            cluster_dir = CLUSTERS_DIR / f"cluster_{int(lab):04d}"
            cluster_dir.mkdir(parents=True, exist_ok=True)

            out_name = f"{Path(rec.image_path).stem}__{rec.face_id}.jpg"
            out_path = cluster_dir / out_name
            cv2.imwrite(str(out_path), face_crop)
            rec.thumb_path = str(out_path)

    # csv
    df = pd.DataFrame([{
        "face_id": r.face_id,
        "cluster_id": int(labels[i]),
        "image_path": r.image_path,
        "thumb_path": r.thumb_path,
        "det_score": r.det_score,
        "bbox_x1": float(r.bbox[0]),
        "bbox_y1": float(r.bbox[1]),
        "bbox_x2": float(r.bbox[2]),
        "bbox_y2": float(r.bbox[3]),
    } for i, r in enumerate(records)])
    df.to_csv(OUT_DIR / "face_clusters.csv", index=False)

    # summary
    summary = {
        "num_faces": int(len(records)),
        "num_clusters": int(len(np.unique(labels))) if len(records) else 0,
        "cw_threshold": float(CW_THRESHOLD),
        "cw_iterations": int(CW_ITERATIONS),
        "split_variance_threshold": float(SPLIT_VARIANCE_THRESHOLD),
        "merge_centroid_threshold": float(MERGE_CENTROID_THRESHOLD),
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Results saved to: {OUT_DIR}")
    print(f"   - Clusters: {CLUSTERS_DIR}")
    print(f"   - CSV: {OUT_DIR}/face_clusters.csv")
    print(f"   - Summary: {OUT_DIR}/summary.json")


def main():
    records = extract_faces_and_embeddings(IMAGES_DIR)
    if not records:
        print("No faces found.")
        return

    embeddings = np.stack([r.embedding for r in records], axis=0).astype(np.float32)

    G = build_similarity_graph(embeddings, threshold=CW_THRESHOLD)

    label_map = chinese_whispers_clustering(G, iterations=CW_ITERATIONS)
    labels = np.array([label_map[i] for i in range(len(records))], dtype=np.int64)

    # optional refinements (this was the “advanced refinement” version)
    labels = split_impure_clusters(labels, embeddings, threshold=SPLIT_VARIANCE_THRESHOLD)
    labels = merge_close_clusters(labels, embeddings, merge_thresh=MERGE_CENTROID_THRESHOLD)

    # compact cluster ids to 0..K-1
    uniq = {lab: i for i, lab in enumerate(sorted(np.unique(labels)))}
    labels = np.array([uniq[int(x)] for x in labels], dtype=np.int64)

    write_outputs(records, labels)


if __name__ == "__main__":

    main()
