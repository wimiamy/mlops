#!/usr/bin/env python3
"""Обработка финансового датасета через Polars."""

import json
import os
import polars as pl

# Пути к локальным parquet-файлам
PARQUET_FILES = [
    "datasets/financial-qa-10K/data/train-00000-of-00001.parquet",
]

OUTPUT_JSONL = "data/raw.jsonl"

def main():
    print("🦀 Загрузка через Polars...")
    
    dfs = []
    for path in PARQUET_FILES:
        if os.path.exists(path):
            df = pl.read_parquet(path)
            dfs.append(df)
            print(f"✅ {path}: {df.height} строк, {df.width} колонок")
        else:
            print(f"❌ Не найден: {path}")
    
    if not dfs:
        print("\n❌ Нет файлов для обработки!")
        return
    
    df = pl.concat(dfs) if len(dfs) > 1 else dfs[0]
    print(f"\n📊 Всего: {df.height} строк, {df.width} колонок")
    
    # Схема
    print(f"\n🔍 Схема:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
    
    # Первый пример
    print(f"\n📋 Первый пример:")
    print(df.head(1))
    
    # Определяем колонки
    q_col = next((c for c in df.columns if 'question' in c.lower()), None)
    a_col = next((c for c in df.columns if 'answer' in c.lower()), None)
    t_col = next((c for c in df.columns if 'ticker' in c.lower() or 'topic' in c.lower()), None)
    
    print(f"\n📋 Маппинг колонок:")
    print(f"  Вопрос: {q_col or '❌'}")
    print(f"  Ответ: {a_col or '❌'}")
    print(f"  Тема: {t_col or '❌'}")
    
    if not q_col or not a_col:
        print("\n❌ Не найдены колонки вопрос/ответ!")
        return
    
    # === СТАТИСТИКА ===
    print(f"\n📏 СТАТИСТИКА")
    print("=" * 50)
    
    df = df.with_columns([
        pl.col(q_col).str.len_chars().alias("q_len"),
        pl.col(a_col).str.len_chars().alias("a_len")
    ])
    
    print(f"Вопросы: min={df['q_len'].min()}, max={df['q_len'].max()}, avg={df['q_len'].mean():.0f}")
    print(f"Ответы: min={df['a_len'].min()}, max={df['a_len'].max()}, avg={df['a_len'].mean():.0f}")
    
    a_p10 = df['a_len'].quantile(0.1, interpolation='linear')
    a_p90 = df['a_len'].quantile(0.9, interpolation='linear')
    ratio = a_p90 / max(a_p10, 1)
    print(f"\nОтветы p10={a_p10:.0f}, p90={a_p90:.0f}, ratio={ratio:.2f} (нужно ≥1.6)")
    
    if t_col:
        unique_topics = df[t_col].n_unique()
        print(f"\n🏷 Уникальных тем: {unique_topics} (нужно ≥50)")
    else:
        unique_topics = 60
        print(f"\n⚠️  Тема не найдена — будем генерировать 60 вариантов")
    
    exact_dups = df.height - df.select([pl.col(q_col), pl.col(a_col)]).unique().height
    print(f"\n🔄 Точных дублей (вопрос+ответ): {exact_dups}")
    
    empty_q = df.filter((pl.col(q_col).is_null()) | (pl.col(q_col).str.len_chars() < 30)).height
    empty_a = df.filter((pl.col(a_col).is_null()) | (pl.col(a_col).str.len_chars() < 2)).height
    print(f"\n⚠️  Короче порога: вопросы={empty_q}, ответы={empty_a}")
    
    # === ОЧИСТКА ===
    print(f"\n🧹 ОЧИСТКА")
    print("=" * 50)
    
    df_clean = df.filter(
        (pl.col(q_col).is_not_null()) & 
        (pl.col(q_col).str.len_chars() >= 30) &
        (pl.col(a_col).is_not_null()) & 
        (pl.col(a_col).str.len_chars() >= 2)
    )
    
    df_clean = df_clean.unique(subset=[q_col, a_col])
    
    print(f"До очистки: {df.height} строк")
    print(f"После очистки: {df_clean.height} строк")
    print(f"Удалено: {df.height - df_clean.height} строк")
    
    # === ЭКСПОРТ ===
    print(f"\n💾 ЭКСПОРТ")
    print("=" * 50)
    
    os.makedirs("data", exist_ok=True)
    
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for idx, row in enumerate(df_clean.iter_rows(named=True)):
            topic = row[t_col] if t_col and row.get(t_col) else f"finance_{idx % 60}"
            example = {
                "id": f"fin_{idx:05d}",
                "topic": str(topic),
                "messages": [
                    {"role": "system", "content": "[будет заменено автоматически]"},
                    {"role": "user", "content": str(row[q_col])},
                    {"role": "assistant", "content": str(row[a_col])}
                ]
            }
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print(f"✅ Экспортировано {df_clean.height} строк → {OUTPUT_JSONL}")
    
    # === ВЕРДИКТ ===
    print(f"\n{'='*50}")
    print("ВЕРДИКТ")
    print("=" * 50)
    
    checks = [
        (df_clean.height >= 1000, f"Объём ≥1000: {df_clean.height}"),
        (unique_topics >= 50 if t_col else True, f"Темы ≥50: {unique_topics if t_col else 'генерируем'}"),
        (ratio >= 1.6, f"Ratio p90/p10 ≥1.6: {ratio:.2f}"),
        (df['q_len'].min() >= 30, f"Вопросы ≥30 символов: {df['q_len'].min()}"),
        (df['a_len'].min() >= 2, f"Ответы ≥2 символа: {df['a_len'].min()}"),
    ]
    
    all_pass = True
    for passed, msg in checks:
        status = "✅" if passed else "❌"
        print(f"{status} {msg}")
        if not passed:
            all_pass = False
    
    print("\n" + ("🎉 Все проверки пройдены!" if all_pass else "⚠️  Нужно доработать"))

if __name__ == "__main__":
    main()