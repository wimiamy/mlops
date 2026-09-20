"""Стадия collect: источник → data/raw.jsonl.
Для финансового датасета financial-qa-10K.
"""
import hashlib
import json
import time
from pathlib import Path
import polars as pl
from src.config import load_params, source_files

BATCH = 2000

def pick_prompt(example_id: str, variants: list[str]) -> str:
    """Детерминированно выбрать вариант инструкции по id примера.
    Именно sha1, а не встроенный hash(): тот солится на каждый запуск процесса,
    и raw.jsonl переставал бы быть воспроизводимым.
    """
    digest = hashlib.sha1(example_id.encode("utf-8")).hexdigest()
    return variants[int(digest, 16) % len(variants)]

def main() -> None:
    params = load_params()
    cfg = params["collect"]
    paths = params["paths"]
    n_rows = cfg["n_rows"]
    variants = cfg["system_prompts"]
    
    if not variants:
        raise SystemExit("collect.system_prompts пуст: инструкцию брать неоткуда")
    
    topics = cfg["topics"]
    wanted = set(topics) if topics else None
    
    out = Path(paths["raw"])
    out.parent.mkdir(parents=True, exist_ok=True)
    
    started = time.perf_counter()
    scanned = written = dropped_topic = dropped_empty = 0
    prompts_used: set[str] = set()
    topics_seen: set[str] = set()
    
    with out.open("w", encoding="utf-8") as fh:
        for src in source_files(params):
            if not src.exists():
                raise SystemExit(f"нет файла-источника: {src}")
            
            # Читаем через Polars (financial-qa-10K это parquet)
            df = pl.read_parquet(src)
            
            taken = 0
            for idx, row in enumerate(df.iter_rows(named=True)):
                if taken >= n_rows:
                    break
                
                scanned += 1
                
                # Фильтр по темам (ticker)
                topic = row.get("ticker", "")
                if wanted is not None and topic not in wanted:
                    dropped_topic += 1
                    continue
                
                # Пропускаем строки с пустыми вопросами или ответами
                question = row.get("question", "")
                answer = row.get("answer", "")
                
                if not question or not answer:
                    dropped_empty += 1
                    continue
                
                # Генерируем ID если нет
                example_id = row.get("id", f"fin_{idx:05d}")
                
                # Выбираем системный промпт
                prompt = pick_prompt(example_id, variants)
                prompts_used.add(prompt)
                topics_seen.add(topic)
                
                # Формируем запись в формате чата
                record = {
                    "id": example_id,
                    "topic": topic,
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ],
                }
                
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                taken += 1
                written += 1
    
    metrics = {
        "version": cfg["version"],
        "files": len(source_files(params)),
        "rows_scanned": scanned,
        "rows_written": written,
        "dropped_topic_filter": dropped_topic,
        "dropped_empty": dropped_empty,
        "topics_filter": len(wanted) if wanted else 0,
        "topics_seen": len(topics_seen),
        "system_prompt_variants": len(prompts_used),
        "seconds": round(time.perf_counter() - started, 2),
    }
    
    mpath = Path(paths["metrics_collect"])
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    
    print(
        f"collect: версия {cfg['version']}, файлов {metrics['files']}, "
        f"просмотрено {scanned}, записано {written} "
        f"(фильтр тем -{dropped_topic}, пустые -{dropped_empty}), "
        f"тем {len(topics_seen)}, вариантов инструкции {len(prompts_used)}, "
        f"{metrics['seconds']} с → {out}"
    )

if __name__ == "__main__":
    main()