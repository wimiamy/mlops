"""Стадия split: разбиение на train/val/test."""
import json
import random
import time
from pathlib import Path
from collections import defaultdict  # ← добавили
from src.config import load_params
from src.contamination import report
from src.schema import Example, dump, iter_examples
from src.textnorm import normalize_group

def row_split(count: int, ratios: dict[str, float], seed: int) -> list[str]:
    """Раздать ГРУППАМ метки сплита в заданных долях."""
    order = list(range(count))
    random.Random(seed).shuffle(order)
    labels = [""] * count
    
    # Явный порядок: train → val → test
    start = 0
    
    # Train
    stop = round(count * ratios["train"])
    for pos in order[start:stop]:
        labels[pos] = "train"
    start = stop
    
    # Val
    stop = start + round(count * ratios["val"])
    for pos in order[start:stop]:
        labels[pos] = "val"
    start = stop
    
    # Test (всё остальное)
    for pos in order[start:]:
        labels[pos] = "test"
    
    return labels

def main() -> None:
    params = load_params()
    paths = params["paths"]
    cfg = params["split"]
    started = time.perf_counter()
    
    examples: list[Example] = list(iter_examples(paths["clean"]))
    
    if cfg["group_key"] != "topic":
        raise SystemExit(f"неизвестный split.group_key: {cfg['group_key']!r}")
    
    # ← ИСПРАВЛЕНО: Группируем примеры по topic
    groups: dict[str, list[Example]] = defaultdict(list)
    for ex in examples:
        key = normalize_group(ex.topic)
        groups[key].append(ex)
    
    # ← ИСПРАВЛЕНО: Сплитим ГРУППЫ, а не строки
    group_names = list(groups.keys())
    labels = row_split(len(group_names), cfg["ratios"], cfg["seed"])
    
    buckets: dict[str, list[Example]] = {name: [] for name in cfg["ratios"]}
    for label, group_name in zip(labels, group_names):
        buckets[label].extend(groups[group_name])  # ← добавляем всю группу
    
    for name, rows in buckets.items():
        out = Path(paths[name])
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for ex in rows:
                fh.write(dump(ex) + "\n")
    
    nd = params["clean"]["near_dup"]
    rep = report(
        buckets["train"],
        buckets["test"],
        shingle_words=nd["shingle_words"],
        num_perm=nd["num_perm"],
        threshold=params["contamination"]["threshold"],
    )
    
    metrics = {
        "version": params["collect"]["version"],
        "seed": cfg["seed"],
        "group_key": cfg["group_key"],
        "groups_total": len(groups),  # ← исправлено
        "sizes": {name: len(rows) for name, rows in buckets.items()},
        "groups": {
            name: len({normalize_group(ex.topic) for ex in rows}) for name, rows in buckets.items()
        },
        "ratios_actual": {
            name: round(len(rows) / len(examples), 4) for name, rows in buckets.items()
        },
        "contamination": rep,
        "seconds": round(time.perf_counter() - started, 2),
    }
    
    mpath = Path(paths["metrics_split"])
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    
    print(
        "split: "
        + ", ".join(f"{name} {len(rows)}" for name, rows in buckets.items())
        + f" (групп {len(groups)}, {metrics['seconds']} с)"  # ← исправлено
    )

if __name__ == "__main__":
    main()