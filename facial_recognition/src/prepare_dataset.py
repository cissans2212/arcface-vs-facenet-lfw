"""
Script Persiapan Dataset LFW untuk Face Verification (Tahap 1).
- Melakukan inspeksi dataset LFW di CV/dataset/
- Membangun 1000 pasangan gambar (500 SAME, 500 DIFFERENT)
- Menerapkan pembagian 80% Development (800 pairs) dan 20% Test (200 pairs)
- Menggunakan strategi Identity-Disjoint Split untuk mencegah data leakage
- Menyimpan metadata ke dataset/processed/pairs.csv
- Memvalidasi seluruh integritas dataset dan pasangan gambar
"""

import argparse
import itertools
import random
from pathlib import Path
import pandas as pd

from utils import get_dataset_dir, get_project_dir, inspect_raw_dataset, validate_pairs_csv


def create_verification_pairs(
    dataset_dir: Path,
    seed: int = 42,
    total_pairs: int = 1000,
    dev_ratio: float = 0.8,
) -> pd.DataFrame:
    """
    Membangun pasangan gambar Face Verification dengan pemisahan identity-disjoint.
    - Development (80% = 800 pairs): 400 SAME, 400 DIFFERENT
    - Test (20% = 200 pairs): 100 SAME, 100 DIFFERENT
    """
    rng = random.Random(seed)
    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"

    # 1. Kumpulkan seluruh identitas dan hitung jumlah gambarnya
    all_identities = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    id_counts = []
    for id_name in all_identities:
        n_tr = len(list((train_dir / id_name).glob("*.jpg")))
        n_te = len(list((test_dir / id_name).glob("*.jpg")))
        id_counts.append((id_name, n_tr, n_te, n_tr + n_te))

    # Urutkan berdasarkan total gambar untuk stratified identity-disjoint sampling
    id_counts.sort(key=lambda x: x[3], reverse=True)

    # Pilih 12 identitas untuk Test (~20% dari 62 identitas) secara terstratifikasi
    # (mengambil indeks 4, 9, 14, ... hingga terkumpul 12 identitas representatif)
    test_idx_set = set(range(4, len(id_counts), 5))
    test_identities = sorted([id_counts[i][0] for i in range(len(id_counts)) if i in test_idx_set])
    dev_identities = sorted([id_counts[i][0] for i in range(len(id_counts)) if i not in test_idx_set])

    dev_target_same = int(round(total_pairs * dev_ratio * 0.5))  # 400
    dev_target_diff = int(round(total_pairs * dev_ratio * 0.5))  # 400
    test_target_same = int(round(total_pairs * (1.0 - dev_ratio) * 0.5))  # 100
    test_target_diff = int(round(total_pairs * (1.0 - dev_ratio) * 0.5))  # 100

    # 2. Bangun Development SAME Pairs (400 pairs dari 50 identitas, ~8 pairs/identitas)
    dev_same_pairs = []
    dev_imgs_by_id = {}
    for id_name in dev_identities:
        imgs = sorted([f"train/{id_name}/{p.name}" for p in (train_dir / id_name).glob("*.jpg")])
        dev_imgs_by_id[id_name] = imgs
        all_possible_same = list(itertools.combinations(imgs, 2))
        rng.shuffle(all_possible_same)
        # Ambil 8 pasang per identitas (50 * 8 = 400)
        for img1, img2 in all_possible_same[:8]:
            dev_same_pairs.append({
                "image1": img1,
                "image2": img2,
                "identity1": id_name,
                "identity2": id_name,
                "label": 1,
                "split": "development",
            })

    # 3. Bangun Development DIFFERENT Pairs (400 pairs dari kombinasi 50 identitas)
    dev_diff_pairs = []
    dev_id_combos = list(itertools.combinations(dev_identities, 2))  # 1225 kombinasi
    rng.shuffle(dev_id_combos)
    chosen_dev_id_combos = dev_id_combos[:dev_target_diff]

    dev_diff_pair_set = set()
    for id1, id2 in chosen_dev_id_combos:
        attempts = 0
        while attempts < 100:
            img1 = rng.choice(dev_imgs_by_id[id1])
            img2 = rng.choice(dev_imgs_by_id[id2])
            pair_key = tuple(sorted([img1, img2]))
            if pair_key not in dev_diff_pair_set:
                dev_diff_pair_set.add(pair_key)
                dev_diff_pairs.append({
                    "image1": img1,
                    "image2": img2,
                    "identity1": id1,
                    "identity2": id2,
                    "label": 0,
                    "split": "development",
                })
                break
            attempts += 1

    # 4. Bangun Test SAME Pairs (100 pairs dari 12 identitas test)
    test_imgs_by_id = {}
    test_same_candidates = []
    for id_name in test_identities:
        imgs = sorted([f"test/{id_name}/{p.name}" for p in (test_dir / id_name).glob("*.jpg")])
        test_imgs_by_id[id_name] = imgs
        all_possible_same = list(itertools.combinations(imgs, 2))
        rng.shuffle(all_possible_same)
        test_same_candidates.append((id_name, all_possible_same))

    test_same_pairs = []
    while len(test_same_pairs) < test_target_same:
        added = 0
        for id_name, pool in test_same_candidates:
            if pool and len(test_same_pairs) < test_target_same:
                img1, img2 = pool.pop()
                test_same_pairs.append({
                    "image1": img1,
                    "image2": img2,
                    "identity1": id_name,
                    "identity2": id_name,
                    "label": 1,
                    "split": "test",
                })
                added += 1
        if added == 0:
            break

    # 5. Bangun Test DIFFERENT Pairs (100 pairs dari 12 identitas test)
    test_diff_pairs = []
    test_id_combos = list(itertools.combinations(test_identities, 2))  # 66 kombinasi unik
    rng.shuffle(test_id_combos)
    # Tambahkan 34 kombinasi tambahan dengan shuffle untuk mencapai target 100
    test_combos_pool = test_id_combos + test_id_combos[:34]
    rng.shuffle(test_combos_pool)

    test_diff_pair_set = set()
    for id1, id2 in test_combos_pool:
        attempts = 0
        while attempts < 100:
            img1 = rng.choice(test_imgs_by_id[id1])
            img2 = rng.choice(test_imgs_by_id[id2])
            pair_key = tuple(sorted([img1, img2]))
            if pair_key not in test_diff_pair_set:
                test_diff_pair_set.add(pair_key)
                test_diff_pairs.append({
                    "image1": img1,
                    "image2": img2,
                    "identity1": id1,
                    "identity2": id2,
                    "label": 0,
                    "split": "test",
                })
                break
            attempts += 1

    all_pairs = dev_same_pairs + dev_diff_pairs + test_same_pairs + test_diff_pairs
    rng.shuffle(all_pairs)
    df = pd.DataFrame(all_pairs)

    return df


def main():
    parser = argparse.ArgumentParser(description="LFW Dataset Preparation for Face Verification")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--total-pairs", type=int, default=1000, help="Total pairs to generate")
    args = parser.parse_args()

    project_dir = get_project_dir()
    dataset_dir = get_dataset_dir()
    output_dir = project_dir / "dataset" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "pairs.csv"

    print("=" * 60)
    print("1. INSPEKSI DATASET LFW")
    print("=" * 60)
    print(f"Direktori Dataset: {dataset_dir}")
    report = inspect_raw_dataset(dataset_dir)
    print(f"Total Identitas/Orang        : {report['total_identities']}")
    print(f"Total Gambar Raw di Dataset  : {report['total_images_raw']}")
    print(f"  - Folder train             : {report['folder_stats']['train']['total_images']} gambar (62 identitas)")
    print(f"  - Folder test              : {report['folder_stats']['test']['total_images']} gambar (62 identitas)")
    print(f"  - Folder database          : {report['folder_stats']['database']['total_images']} gambar (duplikat train)")
    print(f"Total Gambar Unik (Train+Test): {report['total_unique_images']}")
    print(f"Format Gambar                : {', '.join(report['formats'])}")
    print(f"Ukuran/Resolusi Gambar       : {report['resolutions']}")
    print(f"Identitas dengan gambar < 2  : {len(report['identities_less_than_2'])}")
    print(f"File Rusak (Corrupted)       : {report['corrupted_count']}")

    print("\n" + "=" * 60)
    print("2. PEMBENTUKAN PAIR & IDENTITY-DISJOINT SPLIT")
    print("=" * 60)
    df_pairs = create_verification_pairs(
        dataset_dir=dataset_dir,
        seed=args.seed,
        total_pairs=args.total_pairs,
        dev_ratio=0.8,
    )

    # Simpan ke CSV
    df_pairs.to_csv(output_csv, index=False)
    print(f"File pasangan berhasil disimpan di: {output_csv}")

    print("\n" + "=" * 60)
    print("3. VALIDASI PASANGAN GAMBAR (pairs.csv)")
    print("=" * 60)
    val_res = validate_pairs_csv(output_csv, dataset_dir)
    print(f"Status Validasi             : {'BERHASIL (VALID)' if val_res['is_valid'] else 'GAGAL'}")
    print(f"Total Pasangan              : {val_res['total_pairs']}")
    print(f"  - Pasangan SAME (label=1) : {val_res['total_same']}")
    print(f"  - Pasangan DIFF (label=0) : {val_res['total_diff']}")
    print(f"  - Split Development (80%) : {val_res['development_pairs']} (SAME: {val_res['dev_same']}, DIFF: {val_res['dev_diff']})")
    print(f"  - Split Test (20%)        : {val_res['test_pairs']} (SAME: {val_res['test_same']}, DIFF: {val_res['test_diff']})")
    print(f"Self-pair (image1 == image2): {val_res['self_pair_count']}")
    print(f"Duplicate pair              : {val_res['duplicate_pair_count']}")
    print(f"Label error                 : {val_res['label_error_count']}")
    print(f"Missing images              : {val_res['missing_images_count']}")
    print(f"Unreadable images           : {val_res['unreadable_images_count']}")
    print(f"Identitas Development      : {val_res['dev_identities_count']}")
    print(f"Identitas Test              : {val_res['test_identities_count']}")
    print(f"Identity Overlap (Leakage)  : {val_res['identity_overlap_count']}")
    print(f"Image Overlap (Leakage)     : {val_res['image_overlap_count']}")

    if not val_res["is_valid"]:
        raise RuntimeError("Validasi pasangan gagal! Silakan periksa konfigurasi dataset.")


if __name__ == "__main__":
    main()
