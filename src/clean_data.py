
import glob
import os
import re

import pandas as pd

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
OUTPUT_PATH = os.path.join(DATASET_DIR, "cleaned_dataset.csv")

COLUMNS = [
    "EID",
    "Title",
    "Abstract",
    "Author Keywords",
    "Index Keywords",
]


BOILERPLATE_PATTERNS = [
    r"Copyright\s*\(c\)\s*\d{4}.*?(?:https?://\S+)?",
    r"©\s*\d{4}.*?(?:[Aa]ll [Rr]ights [Rr]eserved\.?)",
    r"https?://creativecommons\.org/licenses/\S+",
]
BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_PATTERNS), flags=re.DOTALL)


def load_all_subjects() -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(os.path.join(DATASET_DIR, "*.csv"))):
        subject = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path, usecols=COLUMNS)
        df["subject"] = subject
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def drop_cross_subject_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """ลบบทความที่ EID เดียวกันปรากฎในหลาย subject ออกจากทุก subject ที่พบ"""
    dup_eids = set(df.loc[df.duplicated("EID", keep=False), "EID"])
    return df[~df["EID"].isin(dup_eids)].copy()


def clean_abstract(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = BOILERPLATE_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def is_english_title(title: str, ascii_threshold: float = 0.9) -> bool:
    """heuristic: ถือว่าเป็นภาษาอังกฤษถ้าตัวอักษร (ไม่รวมช่องว่าง/สัญลักษณ์) เป็น ASCII อย่างน้อย 90%"""
    if not isinstance(title, str) or not title.strip():
        return False
    letters = [c for c in title if c.isalpha()]
    if not letters:
        return True  # ไม่มีตัวอักษรเลย (เช่น เลข/สัญลักษณ์อย่างเดียว) ไม่ตัดออกด้วยเหตุนี้
    ascii_letters = sum(1 for c in letters if c.isascii())
    return (ascii_letters / len(letters)) >= ascii_threshold


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_cross_subject_duplicates(df)

    df["Abstract"] = df["Abstract"].apply(clean_abstract)
    df["Author Keywords"] = df["Author Keywords"].fillna("")
    df["Index Keywords"] = df["Index Keywords"].fillna("")

    before = len(df)
    df = df[df["Title"].apply(is_english_title)]
    print(f"ลบ Title ที่ไม่ใช่ภาษาอังกฤษ: {before - len(df)} แถว")

    before = len(df)
    df = df.drop_duplicates(subset=["Title"]).reset_index(drop=True)
    print(f"ลบ Title ที่ซ้ำกัน: {before - len(df)} แถว")

    return df


def main() -> None:
    raw = load_all_subjects()
    print("จำนวนแถวทั้งหมดก่อนทำความสะอาด:", len(raw))

    cleaned = clean(raw)
    print("จำนวนแถวหลังทำความสะอาด:", len(cleaned))
    print()
    print("จำนวนบทความต่อ subject:")
    print(cleaned["subject"].value_counts())

    cleaned.to_csv(OUTPUT_PATH, index=False)
    print()
    print(f"บันทึกผลลัพธ์ที่: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
