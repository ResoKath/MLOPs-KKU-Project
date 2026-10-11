"""Explicit raw schema and train-only, serializable preprocessing."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "Time_taken (min)"
CATEGORIES = {
    "Weather_conditions": ["Fog", "Stormy", "Cloudy", "Sandstorms", "Windy", "Sunny", "Unknown"],
    "Road_traffic_density": ["Low", "Medium", "High", "Jam", "Unknown"],
    "Vehicle_condition": [0, 1, 2, 3],
    "Type_of_order": ["Snack", "Meal", "Drinks", "Buffet", "Unknown"],
    "Type_of_vehicle": ["motorcycle", "scooter", "electric_scooter", "bicycle", "Unknown"],
    "Festival": ["Yes", "No", "Unknown"],
    "City": ["Metropolitian", "Urban", "Semi-Urban", "Unknown"],
}
NUMERIC = ["Delivery_person_Age", "Delivery_person_Ratings"]
GPS = ["Restaurant_latitude", "Restaurant_longitude", "Delivery_location_latitude", "Delivery_location_longitude"]
REQUIRED = NUMERIC + GPS + list(CATEGORIES) + ["Time_Orderd"]
FEATURES = NUMERIC + ["distance_km"] + list(CATEGORIES) + ["order_hour", "order_minute", "order_time_missing"]


def normalize(raw):
    data = raw.copy()
    for col in data.select_dtypes(include=["object", "string"]):
        data[col] = data[col].map(lambda x: x.strip() if isinstance(x, str) else x)
        data[col] = data[col].replace(["", "NaN", "nan", "None", "null"], np.nan)
    return data


def distance(data):
    lat1, lon1, lat2, lon2 = [np.radians(data[c].astype(float)) for c in GPS]
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def parse_time(values):
    """Accept HH:MM and Excel fractions; 1 means midnight next day."""
    t = pd.to_datetime(values, format="%H:%M", errors="coerce")
    fraction = pd.to_numeric(values, errors="coerce")
    mask = fraction.between(0, 1) & fraction.notna()
    minutes = (fraction[mask] * 1440).round().astype(int) % 1440
    t.loc[mask] = pd.Timestamp("1900-01-01") + pd.to_timedelta(minutes, unit="m")
    return t


def validate_raw(raw, training=False, allow_invalid_gps=False):
    data = normalize(raw)
    missing = sorted(set(REQUIRED + ([TARGET, "Order_Date"] if training else [])) - set(data.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if data.empty:
        raise ValueError("Empty dataset")
    for col in NUMERIC + GPS + ["Vehicle_condition"] + ([TARGET] if training else []):
        parsed = pd.to_numeric(data[col], errors="coerce")
        if (data[col].notna() & parsed.isna()).any() or np.isinf(parsed).any():
            raise ValueError(f"{col}: nonnumeric or infinite value")
        data[col] = parsed
    for col, allowed in CATEGORIES.items():
        if (~data[col].isin(allowed) & data[col].notna()).any():
            raise ValueError(f"{col}: unsupported category")
    if data.Vehicle_condition.isna().any():
        raise ValueError("Vehicle_condition: missing")
    valid_gps = pd.Series(True, index=data.index)
    for col in GPS:
        valid_gps &= data[col].between(*( (6,38) if "latitude" in col else (68,98)))
    if not allow_invalid_gps and not valid_gps.all():
        raise ValueError("Invalid GPS coordinates for India service area")
    if not allow_invalid_gps and (distance(data) > 50).any():
        raise ValueError("distance_km: above service limit 50 km")
    # Null time is supported; malformed non-null time stops the pipeline/API.
    t = parse_time(data.Time_Orderd)
    if (data.Time_Orderd.notna() & t.isna()).any():
        raise ValueError("Time_Orderd: expected HH:MM, Excel fraction 0..1, or null")
    if training:
        if not (data[TARGET].notna() & data[TARGET].gt(0)).all():
            raise ValueError("Invalid positive target")
        data["Order_Date"] = pd.to_datetime(data.Order_Date, format="%d-%m-%Y", errors="coerce")
        if data.Order_Date.isna().any():
            raise ValueError("Order_Date: expected DD-MM-YYYY")
        if "ID" in data and data.ID.duplicated().any():
            raise ValueError("Duplicate order ID")
    return data, valid_gps


def engineered(data, excel_times=True):
    data = data.copy()
    for col, limits in zip(NUMERIC, [(18,65), (1,5)]):
        data.loc[~data[col].between(*limits), col] = np.nan
    for col in CATEGORIES:
        if col != "Vehicle_condition":
            data[col] = data[col].fillna("Unknown")
    data["distance_km"] = distance(data)
    t = parse_time(data.Time_Orderd) if excel_times else pd.to_datetime(data.Time_Orderd,format="%H:%M",errors="coerce")
    data["order_hour"] = t.dt.hour
    data["order_minute"] = t.dt.minute
    data["order_time_missing"] = t.isna().astype(int)
    return data[FEATURES]


class Preprocessor:
    def __init__(self, config=None):
        self.config = config

    def fit(self, raw):
        data, _ = validate_raw(raw)
        x = engineered(data)
        cols = NUMERIC + ["order_hour", "order_minute"]
        medians = x[cols].median()
        if medians.isna().any():
            raise ValueError("Cannot learn median from all-missing train column")
        self.config = {
            "schema_version": 1, "features": FEATURES, "categories": CATEGORIES,
            "medians": medians.to_dict(), "gps_bounds": {"latitude": [6,38], "longitude": [68,98]},
            "max_distance_km": 50, "outlier_policy": "age outside 18..65 or rating outside 1..5 -> train median",
            "missing_time_policy": "train median hour/minute plus missing indicator",
            "time_format": "HH:MM or Excel fraction 0..1; 1=midnight; round to minute",
            "multiple_deliveries": "excluded: not available reliably at order creation; not required for inference",
            "date_policy": "DD-MM-YYYY; whole-date temporal 70/15/15 split",
        }
        return self

    def transform(self, raw):
        if self.config is None:
            raise ValueError("Preprocessor has not been fitted")
        data, _ = validate_raw(raw)
        x = engineered(data,excel_times=self.config.get("time_mode") != "legacy_hhmm_only").fillna(self.config["medians"])
        for col, values in self.config["categories"].items():
            x[col] = pd.Categorical(x[col], categories=values)
        if x.isna().any().any():
            raise ValueError("Missing or unsupported processed value")
        return x[self.config["features"]]

    def save(self, path):
        Path(path).write_text(json.dumps(self.config, indent=2), encoding="utf8")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf8")))


def prepare_data(path):
    raw = pd.read_csv(path)
    data, gps_ok = validate_raw(raw, training=True, allow_invalid_gps=True)
    valid = gps_ok & distance(data).le(50)
    quality = {"raw_rows": len(raw), "invalid_gps_rows": int((~gps_ok).sum()),
               "distance_excluded_rows": int((gps_ok & ~valid).sum()),
               "missing": raw.isna().sum().to_dict(), "duplicates": int(raw.duplicated().sum())}
    clean = data.loc[valid].sort_values("Order_Date", kind="stable")
    counts = clean.groupby("Order_Date").size().cumsum()
    train_cut = counts[counts >= len(clean)*0.70].index[0]
    val_cut = counts[counts >= len(clean)*0.85].index[0]
    splits = [clean.loc[clean.Order_Date <= train_cut],
              clean.loc[(clean.Order_Date > train_cut) & (clean.Order_Date <= val_cut)],
              clean.loc[clean.Order_Date > val_cut]]
    if any(s.empty for s in splits):
        raise ValueError("Temporal split requires enough distinct dates")
    assert splits[0].Order_Date.max() < splits[1].Order_Date.min() < splits[2].Order_Date.min()
    for label, part in zip(["train", "validation", "test"], splits):
        quality[label] = {"rows": len(part), "start": str(part.Order_Date.min().date()),
                          "end": str(part.Order_Date.max().date())}
    return splits, quality
