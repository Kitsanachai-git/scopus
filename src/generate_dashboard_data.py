"""
สร้างข้อมูล JSON สำหรับ Scopus/index_Scopus.html (4 กราฟ bibliometric + document drill-down)

Pipeline:
  1. โหลด dataset/cleaned_dataset.csv (ผลลัพธ์จาก clean_data.py, มี EID + subject จริง)
  2. join กลับกับ dataset/*.csv ดิบด้วย EID เพื่อดึงคอลัมน์ Year, DOI, Link กลับมา
  3. โหลด Scopus/subject_classifier.joblib แล้ว predict() subject ของทุกบทความ
     (ใช้ predicted_subject เป็นสีของกราฟ ไม่ใช่ label จริง)
  4. รวม Author Keywords + Index Keywords ต่อบทความ -> เซ็ตคำสำคัญ (dedupe/lower ในบทความเดียวกัน)
  5. คำนวณ 4 ก้อนข้อมูล: trend_heatmap, thematic_evolution, overlay_network, emerging_quadrant
     พร้อม doc_indices ต่อคำ/จุด สำหรับเปิด drill-down panel ในหน้า HTML
  6. เขียนผลลัพธ์เป็น JSON ไฟล์เดียว (Scopus/dashboard_data.json)
"""

import glob
import json
import os
from collections import Counter

import joblib
import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
CLEANED_PATH = os.path.join(DATASET_DIR, "cleaned_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "Scopus", "subject_classifier.joblib")
OUTPUT_PATH = os.path.join(BASE_DIR, "Scopus", "dashboard_data.json")

# ช่วงเวลาที่ใช้แบ่ง period สำหรับ Thematic Evolution / Emerging Quadrant
PERIODS = [
    ("≤2018", None, 2018),
    ("2019–2021", 2019, 2021),
    ("2022–2023", 2022, 2023),
    ("2024–2025", 2024, None),
]

TOP_KEYWORDS_TREND = 30       # จำนวนคำสำคัญใน Keyword Trend Heatmap
TOP_KEYWORDS_NETWORK = 70     # จำนวนคำสำคัญใน Overlay Network
TOP_KEYWORDS_QUADRANT = 90    # จำนวนคำสำคัญใน Emerging Keyword Quadrant
MIN_KEYWORD_DOCS = 5          # ตัดคำที่ปรากฏน้อยเกินไปทิ้ง (noise)


def build_text_column(df: pd.DataFrame) -> pd.Series:
    """ต้องตรงกับ src/train_classifier.py::build_text_column เพื่อให้ predict() ได้ค่าเดียวกับตอน train"""

    def combine(row):
        parts = []
        if row["Title"]:
            parts.append(f"Title: {row['Title']}")
        if row["Abstract"]:
            parts.append(f"Abstract: {row['Abstract']}")
        if row["Author Keywords"]:
            parts.append(f"Author keywords: {row['Author Keywords']}")
        if row["Index Keywords"]:
            parts.append(f"Index keywords: {row['Index Keywords']}")
        return "\n".join(parts)

    return df.fillna("").apply(combine, axis=1)


def load_meta_lookup() -> pd.DataFrame:
    """โหลด EID -> Year, DOI, Link จาก dataset/*.csv ดิบทั้งหมด"""
    frames = []
    for path in sorted(glob.glob(os.path.join(DATASET_DIR, "*.csv"))):
        if os.path.basename(path) == "cleaned_dataset.csv":
            continue
        df = pd.read_csv(path, usecols=["EID", "Year", "DOI", "Link"])
        frames.append(df)
    meta_df = pd.concat(frames, ignore_index=True)
    # EID อาจซ้ำข้าม subject (คนละไฟล์) แต่ metadata ควรตรงกัน ใช้ drop_duplicates กันซ้ำ
    return meta_df.drop_duplicates(subset=["EID"])


def split_keywords(raw: str) -> list:
    if not isinstance(raw, str) or not raw.strip():
        return []
    parts = [p.strip().lower() for p in raw.split(";")]
    return [p for p in parts if p and p not in {"nan"}]


def period_of(year: int) -> str:
    for label, lo, hi in PERIODS:
        if lo is not None and year < lo:
            continue
        if hi is not None and year > hi:
            continue
        return label
    return PERIODS[-1][0]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CLEANED_PATH)
    meta_lookup = load_meta_lookup()
    df = df.merge(meta_lookup, on="EID", how="left")
    before = len(df)
    df = df.dropna(subset=["Year"]).copy()
    df["Year"] = df["Year"].astype(int)
    print(f"บทความที่หา Year ไม่พบ (ตัดทิ้ง): {before - len(df)}")

    df["keywords"] = (
        df["Author Keywords"].fillna("").apply(split_keywords)
        .combine(df["Index Keywords"].fillna("").apply(split_keywords), lambda a, b: sorted(set(a) | set(b)))
    )
    df["period"] = df["Year"].apply(period_of)

    print("ทำนาย subject ด้วย LinearSVC pipeline ...")
    pipe = joblib.load(MODEL_PATH)
    df["text"] = build_text_column(df)
    df["predicted_subject"] = pipe.predict(df["text"])

    df = df.reset_index(drop=True)
    df["doc_idx"] = df.index

    return df


def build_documents(df: pd.DataFrame, classes: list) -> dict:
    """เอกสารดิบสำหรับ drill-down เก็บแบบ columnar (array ต่อฟิลด์) เพื่อลดขนาดไฟล์
    เทียบกับ array-of-object ที่ต้องคีย์ชื่อฟิลด์ซ้ำทุกแถว title/year/doi/link เรียงตาม doc_idx"""
    class_to_idx = {c: i for i, c in enumerate(classes)}
    titles, years, dois, links, subject_idx, periods = [], [], [], [], [], []
    for row in df.itertuples(index=False):
        titles.append(row.Title)
        years.append(int(row.Year))
        dois.append(str(row.DOI) if pd.notna(row.DOI) else "")
        links.append(str(row.Link) if pd.notna(row.Link) else "")
        subject_idx.append(class_to_idx[row.predicted_subject])
        periods.append(row.period)
    return {
        "title": titles,
        "year": years,
        "doi": dois,
        "link": links,
        "subject_idx": subject_idx,
        "period": periods,
    }


def compute_keyword_stats(df: pd.DataFrame):
    """คำนวณสถิติระดับคำสำคัญ: doc frequency รวม, ต่อ period, subject เด่น, ปีเฉลี่ย, doc_idx ที่มีคำนั้น"""
    exploded = (
        df[["keywords", "period", "Year", "predicted_subject", "doc_idx"]]
        .explode("keywords")
        .dropna(subset=["keywords"])
    )
    exploded = exploded[exploded["keywords"] != ""]

    freq_total = exploded["keywords"].value_counts()
    freq_total = freq_total[freq_total >= MIN_KEYWORD_DOCS]

    period_labels = [p[0] for p in PERIODS]
    by_period = (
        exploded[exploded["keywords"].isin(freq_total.index)]
        .groupby(["keywords", "period"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=period_labels, fill_value=0)
    )
    docs_per_period = df.groupby("period").size().reindex(period_labels, fill_value=0)

    kept = exploded[exploded["keywords"].isin(freq_total.index)]
    dominant_subject = kept.groupby("keywords")["predicted_subject"].agg(lambda s: s.value_counts().idxmax())
    avg_year = kept.groupby("keywords")["Year"].mean()
    doc_indices = kept.groupby("keywords")["doc_idx"].apply(lambda s: sorted(int(i) for i in s))

    return {
        "freq_total": freq_total,
        "by_period": by_period,
        "docs_per_period": docs_per_period,
        "dominant_subject": dominant_subject,
        "avg_year": avg_year,
        "doc_indices": doc_indices,
        "period_labels": period_labels,
    }


def build_trend_heatmap(df: pd.DataFrame, stats: dict) -> dict:
    """1. Keyword Trend Heatmap: top คำสำคัญ x ปี, ค่า cell = สัดส่วนบทความในปีนั้นที่มีคำนี้
    drill-down เมื่อคลิกเซลล์ทำได้จาก keyword_doc_index[term] กรองด้วย documents.year ที่ client (ไม่ต้องเก็บ doc_idx ต่อเซลล์ ลดขนาดไฟล์)"""
    years = sorted(df["Year"].unique().tolist())
    docs_per_year = df.groupby("Year").size()

    exploded = df[["keywords", "Year"]].explode("keywords").dropna(subset=["keywords"])
    exploded = exploded[exploded["keywords"] != ""]

    top_terms = stats["freq_total"].head(TOP_KEYWORDS_TREND).index.tolist()
    kept = exploded[exploded["keywords"].isin(top_terms)]

    counts = (
        kept.groupby(["keywords", "Year"]).size().unstack(fill_value=0)
        .reindex(columns=years, fill_value=0).reindex(index=top_terms)
    )
    matrix = counts.div(docs_per_year.reindex(years), axis=1).fillna(0.0)

    return {
        "years": years,
        "terms": top_terms,
        "matrix": [[round(v, 5) for v in row] for row in matrix.values.tolist()],
        "docs_per_year": [int(docs_per_year.get(y, 0)) for y in years],
    }


def build_thematic_evolution(df: pd.DataFrame, stats: dict) -> dict:
    """2. Thematic Evolution: alluvial data - top N คำสำคัญต่อ period + flow (overlap) ระหว่าง period ติดกัน
    drill-down ต่อ node ทำได้จาก keyword_doc_index[term] กรองด้วย documents.period ที่ client"""
    period_labels = stats["period_labels"]
    by_period = stats["by_period"]

    TOP_PER_PERIOD = 14

    nodes = []
    for period in period_labels:
        top = by_period[period].sort_values(ascending=False)
        top = top[top > 0].head(TOP_PER_PERIOD)
        for term, count in top.items():
            nodes.append({
                "id": f"{period}::{term}",
                "period": period,
                "term": term,
                "count": int(count),
                "top_subject": stats["dominant_subject"][term],
            })

    links = []
    for i in range(len(period_labels) - 1):
        p_from, p_to = period_labels[i], period_labels[i + 1]
        from_terms = {n["term"]: n for n in nodes if n["period"] == p_from}
        to_terms = {n["term"]: n for n in nodes if n["period"] == p_to}
        shared = set(from_terms) & set(to_terms)
        for term in shared:
            value = min(from_terms[term]["count"], to_terms[term]["count"])
            links.append({
                "source": f"{p_from}::{term}",
                "target": f"{p_to}::{term}",
                "term": term,
                "value": int(value),
            })

    return {"period_labels": period_labels, "nodes": nodes, "links": links}


def build_overlay_network(df: pd.DataFrame, stats: dict) -> dict:
    """3. Overlay Network: keyword co-occurrence graph, สี node ตามสาขาที่ทำนายเด่นสุด,
    ตำแหน่ง overlay (ปีเฉลี่ย) ใช้เป็นข้อมูลเสริมใน tooltip, ขนาดตาม freq
    drill-down ของ node ใช้ keyword_doc_index[term]; ของ edge = intersection ของสอง term ที่ client"""
    top_terms = stats["freq_total"].head(TOP_KEYWORDS_NETWORK).index.tolist()
    top_set = set(top_terms)

    pair_counts = Counter()
    for kws in df["keywords"]:
        present = sorted(set(kws) & top_set)
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                pair_counts[(present[i], present[j])] += 1

    nodes = [
        {
            "term": term,
            "freq": int(stats["freq_total"][term]),
            "avg_year": round(float(stats["avg_year"][term]), 2),
            "top_subject": stats["dominant_subject"][term],
        }
        for term in top_terms
    ]
    edges = [
        {"source": a, "target": b, "weight": int(w)}
        for (a, b), w in pair_counts.items()
        if w >= 2
    ]
    edges.sort(key=lambda e: -e["weight"])

    return {"nodes": nodes, "edges": edges}


def build_emerging_quadrant(df: pd.DataFrame, stats: dict) -> dict:
    """4. Emerging Keyword Quadrant: X=growth (period ล่าสุด vs ก่อนหน้า), Y=ความแพร่หลายรวม"""
    period_labels = stats["period_labels"]
    latest, prev = period_labels[-1], period_labels[-2]
    docs_per_period = stats["docs_per_period"]
    by_period = stats["by_period"]

    top_terms = stats["freq_total"].head(TOP_KEYWORDS_QUADRANT).index.tolist()

    points = []
    for term in top_terms:
        share_latest = by_period.loc[term, latest] / max(docs_per_period[latest], 1)
        share_prev = by_period.loc[term, prev] / max(docs_per_period[prev], 1)
        growth = share_latest - share_prev
        points.append({
            "term": term,
            "growth": round(float(growth), 5),
            "prevalence": int(stats["freq_total"][term]),
            "count_latest": int(by_period.loc[term, latest]),
            "count_prev": int(by_period.loc[term, prev]),
            "share_latest": round(float(share_latest), 5),
            "share_prev": round(float(share_prev), 5),
            "top_subject": stats["dominant_subject"][term],
        })

    return {
        "points": points,
        "latest_period": latest,
        "prev_period": prev,
    }


def collect_used_terms(*views) -> set:
    """เก็บชุดคำสำคัญทั้งหมดที่ปรากฏใน 4 กราฟ เพื่อสร้าง keyword_doc_index เฉพาะคำที่ใช้จริง (ลดขนาดไฟล์)"""
    used = set()
    trend_heatmap, thematic_evolution, overlay_network, emerging_quadrant = views
    used.update(trend_heatmap["terms"])
    used.update(n["term"] for n in thematic_evolution["nodes"])
    used.update(n["term"] for n in overlay_network["nodes"])
    used.update(p["term"] for p in emerging_quadrant["points"])
    return used


def main() -> None:
    df = load_data()
    print(f"บทความทั้งหมดที่ใช้: {len(df)}")

    stats = compute_keyword_stats(df)
    print(f"คำสำคัญที่ผ่านเกณฑ์ (>= {MIN_KEYWORD_DOCS} เอกสาร): {len(stats['freq_total'])}")

    classes = sorted(df["predicted_subject"].unique().tolist())

    trend_heatmap = build_trend_heatmap(df, stats)
    thematic_evolution = build_thematic_evolution(df, stats)
    overlay_network = build_overlay_network(df, stats)
    emerging_quadrant = build_emerging_quadrant(df, stats)

    used_terms = collect_used_terms(trend_heatmap, thematic_evolution, overlay_network, emerging_quadrant)
    keyword_doc_index = {term: stats["doc_indices"][term] for term in used_terms}

    data = {
        "classes": classes,
        "n_articles": int(len(df)),
        "year_min": int(df["Year"].min()),
        "year_max": int(df["Year"].max()),
        "period_labels": stats["period_labels"],
        "documents": build_documents(df, classes),
        "keyword_doc_index": keyword_doc_index,
        "trend_heatmap": trend_heatmap,
        "thematic_evolution": thematic_evolution,
        "overlay_network": overlay_network,
        "emerging_quadrant": emerging_quadrant,
        "model_params": {
            "ngram_range": [1, 2],
            "max_features": 50000,
            "min_df": 2,
            "sublinear_tf": True,
            "algorithm": "LinearSVC (one-vs-rest)",
            "n_classes": len(classes),
            "note": "subject ของแต่ละบทความคือค่าที่โมเดล TF-IDF+LinearSVC ทำนาย ไม่ใช่ label จริงเสมอไป (test accuracy ~94%)",
        },
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))

    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"บันทึกข้อมูลที่: {OUTPUT_PATH} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
