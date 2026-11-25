import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_FILE = "blink_data.csv"
MODEL_FILE = "blink_model.pkl"

FEATURES = [
    "left_ear",
    "right_ear",
    "average_ear",
    "min_ear",
    "ear_difference",
    "eye_width_ratio",
    "nose_offset_ratio",
    "blink_duration"
]

VALID_LABELS = {"SHORT_BLINK", "LONG_BLINK"}

# ============================================
# 1. Load calibration data
# ============================================

df = pd.read_csv(DATA_FILE)

print("Dataset loaded successfully.")
print("Total samples:", len(df))

df = df.dropna(how="all").copy()

df["label"] = (
    df["label"]
    .astype(str)
    .str.strip()
    .str.upper()
)

print("\nClass distribution:")
print(df["label"].value_counts())

# ============================================
# 2. Keep only SHORT/LONG blink classes
# ============================================

invalid_labels = sorted(set(df["label"].unique()) - VALID_LABELS)

if invalid_labels:
    print("\nWARNING: Ignoring invalid labels:")
    print(invalid_labels)

df = df[df["label"].isin(VALID_LABELS)].copy()

if df.empty:
    raise SystemExit(
        "\nERROR: No SHORT_BLINK or LONG_BLINK samples found."
    )

# ============================================
# 3. Check required features
# ============================================

missing_features = [f for f in FEATURES if f not in df.columns]

if missing_features:
    raise SystemExit(
        "\nERROR: Dataset is missing:\n" +
        "\n".join(missing_features)
    )

X = df[FEATURES].copy()
y = df["label"].copy()

# ============================================
# 4. Convert features to numeric
# ============================================

for feature in FEATURES:
    X[feature] = pd.to_numeric(X[feature], errors="coerce")

valid_rows = X.notna().all(axis=1)

removed_rows = int((~valid_rows).sum())

if removed_rows:
    print(
        f"\nRemoved {removed_rows} rows containing "
        "invalid numeric values."
    )

X = X.loc[valid_rows].reset_index(drop=True)
y = y.loc[valid_rows].reset_index(drop=True)

# ============================================
# 5. Check blink-duration quality
# ============================================

print("\nBlink duration statistics:")
print(
    X.assign(label=y)
    .groupby("label")["blink_duration"]
    .agg(["count", "min", "mean", "max"])
)

short_data = X.loc[y == "SHORT_BLINK", "blink_duration"]
long_data = X.loc[y == "LONG_BLINK", "blink_duration"]

if len(short_data):
    bad_short = short_data[short_data >= 0.40]
    if len(bad_short):
        print(
            f"\nWARNING: {len(bad_short)} SHORT_BLINK samples "
            "are >= 0.40 seconds."
        )
        print(
            "These samples can cause the model to predict a "
            "long blink as SHORT_BLINK."
        )

if len(long_data):
    bad_long = long_data[long_data < 0.40]
    if len(bad_long):
        print(
            f"\nWARNING: {len(bad_long)} LONG_BLINK samples "
            "are < 0.40 seconds."
        )
        print(
            "Check the calibration data before trusting the model."
        )

# ============================================
# 6. Check classes
# ============================================

class_counts = y.value_counts()

if len(class_counts) < 2:
    raise SystemExit(
        "\nERROR: Both SHORT_BLINK and LONG_BLINK classes "
        "are required."
    )

if class_counts.min() < 2:
    raise SystemExit(
        "\nERROR: Each class needs at least 2 samples."
    )

# ============================================
# 7. Split data
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ============================================
# 8. Create model
# ============================================

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    max_depth=8,
    min_samples_leaf=2,
    n_jobs=-1
)

# ============================================
# 9. Train
# ============================================

print("\nTraining model...")
model.fit(X_train, y_train)
print("Training completed.")

# ============================================
# 10. Test
# ============================================

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n============================================")
print("MODEL ACCURACY")
print("============================================")
print(f"{accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=["SHORT_BLINK", "LONG_BLINK"],
        zero_division=0
    )
)

# ============================================
# 11. Confusion matrix
# ============================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["SHORT_BLINK", "LONG_BLINK"]
)

print("\nConfusion Matrix")
print("Rows = actual, Columns = predicted")
print("                 SHORT    LONG")
print(f"SHORT_BLINK      {cm[0][0]:5d}  {cm[0][1]:5d}")
print(f"LONG_BLINK       {cm[1][0]:5d}  {cm[1][1]:5d}")

# ============================================
# 12. Feature importance
# ============================================

importance = pd.Series(
    model.feature_importances_,
    index=FEATURES
).sort_values(ascending=False)

print("\nFeature importance:")
for feature, value in importance.items():
    print(f"{feature:22s}: {value:.4f}")

# ============================================
# 13. Save model
# ============================================

joblib.dump(model, MODEL_FILE)

print("\n============================================")
print(f"Model saved as: {MODEL_FILE}")
print("============================================")

print("\nIMPORTANT:")
print(
    "If a 1+ second blink is still predicted as SHORT_BLINK, "
    "the next fix must be made in eye_detection.py."
)
