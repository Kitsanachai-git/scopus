

import os

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
CLEANED_PATH = os.path.join(DATASET_DIR, "cleaned_dataset.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "subject_classifier.joblib")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "model_report.md")

RANDOM_STATE = 42


def build_text_column(df: pd.DataFrame) -> pd.Series:
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


def main() -> None:
    df = pd.read_csv(CLEANED_PATH)
    df["text"] = build_text_column(df)

    X = df["text"]
    y = df["subject"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=50000,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=2,
                ),
            ),
            ("clf", LinearSVC(C=1.0, random_state=RANDOM_STATE)),
        ]
    )

    pipe.fit(X_train, y_train)

    train_pred = pipe.predict(X_train)
    test_pred = pipe.predict(X_test)
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"\nTrain accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")
    print(f"Train-test gap: {train_acc - test_acc:.4f}")

    # Cross-validation บน train set เพื่อเช็คความเสถียรของผลลัพธ์
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, n_jobs=-1)
    print(f"\n5-fold CV accuracy: {cv_scores}")
    print(f"CV mean: {cv_scores.mean():.4f}  CV std: {cv_scores.std():.4f}")

    # Baseline: ทำนายด้วย majority class ตลอด เพื่อเช็ค underfitting
    baseline = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    baseline.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, baseline.predict(X_test))
    print(f"\nMajority-class baseline test accuracy: {baseline_acc:.4f}")

    report = classification_report(y_test, test_pred)
    print("\nClassification report (test set):")
    print(report)

    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, test_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    joblib.dump(pipe, MODEL_PATH)
    print(f"\nบันทึกโมเดลที่: {MODEL_PATH}")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Model Report: Scopus Subject Classifier\n\n")
        f.write(f"- Train size: {len(X_train)}\n")
        f.write(f"- Test size: {len(X_test)}\n")
        f.write(f"- Train accuracy: {train_acc:.4f}\n")
        f.write(f"- Test accuracy: {test_acc:.4f}\n")
        f.write(f"- Train-test gap: {train_acc - test_acc:.4f}\n")
        f.write(f"- 5-fold CV mean accuracy (on train set): {cv_scores.mean():.4f}\n")
        f.write(f"- 5-fold CV std: {cv_scores.std():.4f}\n")
        f.write(f"- Majority-class baseline test accuracy: {baseline_acc:.4f}\n\n")
        f.write("## Classification report (test set)\n\n```\n")
        f.write(report)
        f.write("```\n\n")
        f.write("## Confusion matrix (test set)\n\n")
        f.write(cm_df.to_markdown())
        f.write("\n")

    print(f"บันทึกรายงานที่: {REPORT_PATH}")


if __name__ == "__main__":
    main()
