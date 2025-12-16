import pandas as pd
import numpy as np

ICD_MAP = {
    "diabetes": {9: ["250"], 10: ["E10", "E11", "E13", "E14"]},
    "hypertension": {9: ["401", "402", "403", "404", "405"], 10: ["I10", "I11", "I12", "I13", "I15"]},
    "dyslipidemia": {9: ["272"], 10: ["E78"]},
    "ischemic_stroke": {9: ["433", "434", "435"], 10: ["I63", "G45"]},
    "atrial_fibrillation": {9: ["42731"], 10: ["I48"]},
    "sleep_disordered_breathing": {9: ["32723", "3272", "78057"], 10: ["G473"]},
}

def icd_matches(code: str, prefixes) -> bool:
    code = str(code).replace(".", "").strip()  # normalize if dots exist
    return any(code.startswith(p.replace(".", "")) for p in prefixes)

# Load tables
diagnoses = pd.read_csv("hosp/diagnoses_icd.csv")       # subject_id, hadm_id, icd_code, icd_version
discharge = pd.read_csv("note/discharge.csv")           # note_id, subject_id, hadm_id, text

# Normalize ICD codes once
diagnoses["icd_code_norm"] = diagnoses["icd_code"].astype(str).str.replace(".", "", regex=False).str.strip()

# Get unique admissions appearing in discharge notes (restrict to those we can actually use)
usable_hadm = set(discharge["hadm_id"].dropna().unique())
dx = diagnoses[diagnoses["hadm_id"].isin(usable_hadm)].copy()

# Build admission-level labels
cats = list(ICD_MAP.keys())
hadm_ids = sorted(dx["hadm_id"].unique())
Y = pd.DataFrame(0, index=hadm_ids, columns=cats, dtype=int)

# Efficient grouping
for hadm_id, g in dx.groupby("hadm_id"):
    for cat, mp in ICD_MAP.items():
        # filter within hadm by version
        hit = False
        for ver, prefixes in mp.items():
            gv = g[g["icd_version"] == ver]
            if gv.empty:
                continue
            if gv["icd_code_norm"].apply(lambda c: icd_matches(c, prefixes)).any():
                hit = True
                break
        Y.loc[hadm_id, cat] = 1 if hit else 0

# Attach discharge text (one row per hadm_id; if multiple notes per hadm_id, keep the longest)
discharge = discharge.dropna(subset=["hadm_id", "text"]).copy()
discharge["text_len"] = discharge["text"].astype(str).str.len()

discharge_best = (
    discharge.sort_values("text_len", ascending=False)
            .drop_duplicates(subset=["hadm_id"])
            .set_index("hadm_id")
)

# Align label matrix and notes
Y = Y.loc[Y.index.intersection(discharge_best.index)]
data = Y.join(discharge_best[["subject_id", "text"]], how="inner").reset_index().rename(columns={"index":"hadm_id"})

# Optional: remove near-empty / templated notes
MIN_CHARS = 250
data = data[data["text"].str.len() >= MIN_CHARS].reset_index(drop=True)

print("Usable admissions:", len(data))
print("Positive counts per category:\n", data[cats].sum())
