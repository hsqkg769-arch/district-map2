import os
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go

# ======================================================
# Page
# ======================================================
st.set_page_config(page_title="District Population / School / Bear / Land Price Viewer", layout="wide")

from shapely.geometry import Point

def geom_to_latlon(geom):
    """Return (lat, lon). Works for Point and other geometries (uses centroid)."""
    if geom is None:
        return None
    try:
        if isinstance(geom, Point):
            return (geom.y, geom.x)
        c = geom.centroid
        return (c.y, c.x)
    except Exception:
        return None
# =========================
# Column names
# =========================
# schoolPT.shp の学校名列（候補→無ければ5列目）
SCHOOL_NAME_CANDIDATES = ["school_nam", "school_name", "NAME", "Name", "name"]


# ======================================================
# Paths (all under data/)
# ======================================================
DATA_DIR = "data"

DISTRICT_SHP = os.path.join(DATA_DIR, "project.shp")
SCHOOL_PT_SHP = os.path.join(DATA_DIR, "schoolPT.shp")
BEAR_PT_SHP = os.path.join(DATA_DIR, "bearPT.shp")

POP_CSV = os.path.join(DATA_DIR, "jinko2018_2025.csv")
POP_LATEST_CSV = os.path.join(DATA_DIR, "pop_latest.csv")
POP_BY_GAKKU_YEAR_CSV = os.path.join(DATA_DIR, "pop_by_gakku_year.csv")
SCHOOL_CSV = os.path.join(DATA_DIR, "school.csv")
BEAR_CSV = os.path.join(DATA_DIR, "bear.csv")
BEAR_RISK_BY_GAKKU_CSV = os.path.join(DATA_DIR, "bear_risk_by_gakku.csv")

# land price: prefer shp if exists
LAND_SHP = os.path.join(DATA_DIR, "Chika.shp")
LAND_CSV = os.path.join(DATA_DIR, "chika2.csv")

# Future projections and predictions
POP_2030_CSV = os.path.join(DATA_DIR, "population_trend_2030.csv")
STUDENTS_2030_CSV = os.path.join(DATA_DIR, "students_pred_2030_by_gakku.csv")
LAND_PRICE_LONG_CSV = os.path.join(DATA_DIR, "land_price_long.csv")

def pick_col_by_candidates(columns, candidates, fallback_index=None):
    cols = list(columns)
    for c in candidates:
        if c in cols:
            return c
    if fallback_index is not None and 0 <= fallback_index < len(cols):
        return cols[fallback_index]
    return cols[0] if cols else None

# ======================================================
# Columns
# ======================================================
GAKKU_COL = "gakkuNo"   # district key (English)
ID_COL = "ID"

# population columns (your csv should have these)
YEAR_COL_POP = "fiscal year"   # in jinko csv
# We will derive 4 groups if raw ages not present.
# Expect at least: pop_0_11, pop_12_18, pop_19_64, pop_65_over OR similar.
# If only some exist, we handle gracefully.

# school csv
YEAR_COL_SCHOOL = "fiscalyear"  # you changed from "fiscal year"
SCHOOL_NAME_COL_CSV = "school_name"
STUDENTS_COL = "enroll_total"
CLASSES_COL = "classes"

# point shapefile columns
SCHOOL_NAME_COL_PT = "school_nam"   # you said 5th col is school name in schoolPT
# bear points expect year column; if not, we try to parse.
BEAR_YEAR_COL_PT = "year"

# ======================================================
# Helpers
# ======================================================
def safe_read_csv(path: str):
    """Try utf-8 then shift_jis."""
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="shift_jis")

def to_epsg4326(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf is None or gdf.empty:
        return gdf
    if gdf.crs is None:
        # If CRS missing, assume WGS84 (best guess). Adjust if needed.
        gdf = gdf.set_crs(epsg=4326)
    return gdf.to_crs(epsg=4326)

def polyfit_slope(x, y):
    if len(x) < 3:
        return np.nan
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # center x for stability
    return np.polyfit(x - x.mean(), y, 1)[0]

def classify_trend(slope, thr=2.0):
    if pd.isna(slope):
        return "Flat"
    if slope > thr:
        return "Up"
    if slope < -thr:
        return "Down"
    return "Flat"

def normalize_0_100(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce")
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series([0]*len(s), index=s.index)
    return (s - mn) / (mx - mn) * 100

def fmt_int(x):
    if pd.isna(x):
        return "-"
    try:
        return f"{int(round(float(x)))}"
    except Exception:
        return str(x)

def num_or_zero(x):
    try:
        if x is None:
            return 0
        # "--" みたいなのを弾く
        if isinstance(x, str) and (x.strip() == "" or x.strip() == "--"):
            return 0
        return float(x)
    except Exception:
        return 0

def plot_population_structure(latest_row: dict):
    cats = ["0–11", "12–18", "19–64", "65+"]
    vals = [
        num_or_zero(latest_row.get("pop_0_11")),
        num_or_zero(latest_row.get("pop_12_18")),
        num_or_zero(latest_row.get("pop_19_64")),
        num_or_zero(latest_row.get("pop_65_over")),
    ]
    fig = go.Figure()
    fig.add_bar(x=cats, y=vals, text=[int(v) for v in vals], textposition="outside")
    fig.update_layout(
        title="人口構造（2025）",
        height=200,  # 小さい棒グラフ
        margin=dict(l=10, r=10, t=35, b=10),
        xaxis_title="",
        yaxis_title="人口",
    )
    fig.update_yaxes(rangemode="tozero")
    return fig


def plot_timeseries(df_ts: pd.DataFrame, xcol: str, ycol: str, title: str, height=240):
    fig = go.Figure()
    fig.add_scatter(x=df_ts[xcol], y=df_ts[ycol], mode="lines+markers")
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=10, r=10, t=45, b=10),
        xaxis_title="Year",
        yaxis_title=ycol,
    )
    fig.update_xaxes(dtick=1)
    return fig

def plot_bear_risk_bar(bear_risk_0_100: float):
    fig = go.Figure()
    fig.add_bar(x=["Bear risk"], y=[bear_risk_0_100])
    fig.update_layout(
        title="Bear risk (0–100)",
        height=180,
        margin=dict(l=10, r=10, t=45, b=10),
        yaxis_range=[0, 100],
    )
    return fig

def plot_bear_risk_by_gakku_bar(gdf_bear_risk: gpd.GeoDataFrame):
    """学区別の熊リスク指数の棒グラフ"""
    if gdf_bear_risk is None or gdf_bear_risk.empty or "bear_risk_0_100" not in gdf_bear_risk.columns:
        return None
    df_plot = gdf_bear_risk[[GAKKU_COL, "bear_risk_0_100"]].copy()
    df_plot = df_plot.sort_values("bear_risk_0_100", ascending=False).head(20)  # Top 20
    fig = go.Figure()
    fig.add_bar(
        x=df_plot[GAKKU_COL],
        y=df_plot["bear_risk_0_100"],
        marker_color="crimson"
    )
    fig.update_layout(
        title="学区別 熊リスク指数（上位20学区）",
        height=300,
        margin=dict(l=10, r=10, t=45, b=100),
        xaxis_title="学区",
        yaxis_title="熊リスク指数 (0-100)",
        xaxis_tickangle=-45,
    )
    return fig

def plot_bear_risk_scatter(gdf_plot: gpd.GeoDataFrame):
    """散布図（人口トレンド × 熊リスク指数）"""
    if gdf_plot is None or gdf_plot.empty:
        return None
    need_cols = [GAKKU_COL, "pop_slope", "bear_risk_0_100"]
    if not all(c in gdf_plot.columns for c in need_cols):
        return None
    df_scatter = gdf_plot[need_cols].copy()
    df_scatter = df_scatter.dropna()
    if df_scatter.empty:
        return None
    fig = go.Figure()
    fig.add_scatter(
        x=df_scatter["pop_slope"],
        y=df_scatter["bear_risk_0_100"],
        mode="markers+text",
        text=df_scatter[GAKKU_COL],
        textposition="top center",
        marker=dict(size=10, color="darkred", opacity=0.7)
    )
    fig.update_layout(
        title="人口トレンド × 熊リスク指数 散布図",
        height=400,
        margin=dict(l=10, r=10, t=45, b=10),
        xaxis_title="人口トレンド（傾き: 0-11歳）",
        yaxis_title="熊リスク指数 (0-100)",
    )
    return fig

def plot_land_price_series(df_land_long: pd.DataFrame, title="Official land price trend"):
    # df_land_long columns: year(int), price(float)
    fig = go.Figure()
    fig.add_scatter(x=df_land_long["year"], y=df_land_long["price"], mode="lines+markers", name="JPY/m²")

    # Era shading (example)
    eras = [
        ("Bubble", 1983, 1991),
        ("Post-bubble", 1992, 2002),
        ("Recovery", 2003, 2012),
        ("Recent", 2013, 2025),
    ]
    for name, a, b in eras:
        fig.add_vrect(x0=a, x1=b, opacity=0.07, layer="below", line_width=0)
        # small label near top
        fig.add_annotation(x=(a+b)/2, y=1.02, xref="x", yref="paper", text=name, showarrow=False, font=dict(size=10))

    fig.update_layout(
        title=title,
        height=260,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="Year",
        yaxis_title="JPY / m²",
    )
    fig.update_xaxes(dtick=5)
    return fig

# ======================================================
# Load & build features
# ======================================================
@st.cache_data(show_spinner=False)
def load_all():
    # --- polygons
    gdf = gpd.read_file(DISTRICT_SHP)
    gdf = to_epsg4326(gdf)

    # --- population
    df_pop = safe_read_csv(POP_CSV)
    df_pop.columns = [c.strip() for c in df_pop.columns]

    # rename year col if needed
    if YEAR_COL_POP not in df_pop.columns:
        # try common variants
        for cand in ["year", "fiscalyear", "fiscal_year"]:
            if cand in df_pop.columns:
                df_pop = df_pop.rename(columns={cand: YEAR_COL_POP})
                break

    df_pop[YEAR_COL_POP] = pd.to_numeric(df_pop[YEAR_COL_POP], errors="coerce")

    # ------------------------------------------------------
    # Population preprocessing (by gakku × year)
    # ------------------------------------------------------

    # 想定する年齢区分列
    needed = [
        "pop_0_11",
        "pop_12_18",
        "pop_19_64",
        "pop_65_over",
        "pop_total",
    ]

    # 数値型に変換
    for col in needed:
        if col in df_pop.columns:
            df_pop[col] = pd.to_numeric(df_pop[col], errors="coerce")

    # 必要な列だけ残す
    use_cols = [GAKKU_COL, YEAR_COL_POP] + [c for c in needed if c in df_pop.columns]

    df_pop_use = df_pop[use_cols].dropna(subset=[GAKKU_COL, YEAR_COL_POP])

    # 学区 × 年で集計（重複対策）
    grp_pop = (
        df_pop_use
        .groupby([GAKKU_COL, YEAR_COL_POP], as_index=False)
        .sum()
    )

    # --- pop trend (0-11)
    trend_pop = []
    for gakku, sub in grp_pop.groupby(GAKKU_COL):
        sub = sub.sort_values(YEAR_COL_POP)
        if "pop_0_11" not in sub.columns:
            slope = np.nan
        else:
            slope = polyfit_slope(sub[YEAR_COL_POP].values, sub["pop_0_11"].values)
        trend_pop.append({"gakku": gakku, "pop_slope": slope, "pop_trend": classify_trend(slope, thr=2.0)})
    df_pop_trend = pd.DataFrame(trend_pop)

    # latest year summary for pop structure
    latest_year_pop = int(grp_pop[YEAR_COL_POP].max())
    df_pop_latest = grp_pop[grp_pop[YEAR_COL_POP] == latest_year_pop].copy()
    
    # --- load pop_latest.csv if exists (may have pop_0_11 and pop_total)
    if os.path.exists(POP_LATEST_CSV):
        try:
            df_pop_latest_file = safe_read_csv(POP_LATEST_CSV)
            df_pop_latest_file.columns = [c.strip() for c in df_pop_latest_file.columns]
            # 数値型に変換
            for col in ["pop_total", "pop_0_11", "pop_12_18", "pop_19_64", "pop_65_over"]:
                if col in df_pop_latest_file.columns:
                    df_pop_latest_file[col] = pd.to_numeric(df_pop_latest_file[col], errors="coerce")
            # GAKKU_COLが存在する場合、マージする
            if GAKKU_COL in df_pop_latest_file.columns:
                # pop_latest.csvのデータで上書き（より正確なデータの可能性があるため）
                merge_cols = [GAKKU_COL] + [c for c in ["pop_total", "pop_0_11", "pop_12_18", "pop_19_64", "pop_65_over"] if c in df_pop_latest_file.columns]
                df_pop_latest = df_pop_latest.merge(
                    df_pop_latest_file[merge_cols],
                    on=GAKKU_COL,
                    how="outer",
                    suffixes=("", "_latest")
                )
                # _latestサフィックスのあるカラムを優先（pop_latest.csvのデータ）
                for col in ["pop_total", "pop_0_11", "pop_12_18", "pop_19_64", "pop_65_over"]:
                    latest_col = f"{col}_latest"
                    if latest_col in df_pop_latest.columns:
                        # _latestの値で上書き（NULLでない場合）
                        df_pop_latest[col] = df_pop_latest[latest_col].fillna(df_pop_latest[col])
                        df_pop_latest = df_pop_latest.drop(columns=[latest_col])
        except Exception as e:
            # pop_latest.csvの読み込みに失敗しても続行
            pass

    # --- school csv
    df_school = safe_read_csv(SCHOOL_CSV)
    df_school.columns = [c.strip() for c in df_school.columns]

    if YEAR_COL_SCHOOL not in df_school.columns:
        for cand in ["fiscal year", "fiscal_year", "year"]:
            if cand in df_school.columns:
                df_school = df_school.rename(columns={cand: YEAR_COL_SCHOOL})
                break

    df_school[YEAR_COL_SCHOOL] = pd.to_numeric(df_school[YEAR_COL_SCHOOL], errors="coerce")
    df_school[STUDENTS_COL] = pd.to_numeric(df_school[STUDENTS_COL], errors="coerce")
    df_school[CLASSES_COL] = pd.to_numeric(df_school[CLASSES_COL], errors="coerce")

    # gakku-year students total (district view)
    grp_students = (
        df_school
        .dropna(subset=[GAKKU_COL, YEAR_COL_SCHOOL, STUDENTS_COL])
        .groupby([GAKKU_COL, YEAR_COL_SCHOOL], as_index=False)[STUDENTS_COL].sum()
        .sort_values([GAKKU_COL, YEAR_COL_SCHOOL])
    )

    # students trend slope per year
    trend_students = []
    for gakku, sub in grp_students.groupby(GAKKU_COL):
        sub = sub.sort_values(YEAR_COL_SCHOOL)
        slope = polyfit_slope(sub[YEAR_COL_SCHOOL].values, sub[STUDENTS_COL].values)
        trend_students.append({"gakku": gakku, "students_slope": slope, "students_trend": classify_trend(slope, thr=2.0)})
    df_students_trend = pd.DataFrame(trend_students)

    # latest year school risk (for point coloring, per school)
    latest_year_school = int(df_school[YEAR_COL_SCHOOL].max())
    df_school_latest = df_school[df_school[YEAR_COL_SCHOOL] == latest_year_school].copy()

    def student_risk(n):
        if pd.isna(n):
            return "Watch"
        if n < 100:
            return "High"
        if n < 200:
            return "Watch"
        return "Low"

    def class_risk(c):
        if pd.isna(c):
            return "Watch"
        if c <= 6:
            return "High"
        if c <= 11:
            return "Watch"
        return "Low"

    df_school_latest["student_risk"] = df_school_latest[STUDENTS_COL].apply(student_risk)
    df_school_latest["class_risk"] = df_school_latest[CLASSES_COL].apply(class_risk)
    df_school_latest["integration_type"] = df_school_latest["student_risk"] + " / " + df_school_latest["class_risk"]

    # --- load student 2030 predictions (before merging to school_pt)
    df_students_2030 = None
    if os.path.exists(STUDENTS_2030_CSV):
        try:
            df_students_2030 = safe_read_csv(STUDENTS_2030_CSV)
            df_students_2030.columns = [c.strip() for c in df_students_2030.columns]
            if "students_pred_2030" in df_students_2030.columns:
                df_students_2030["students_pred_2030"] = pd.to_numeric(df_students_2030["students_pred_2030"], errors="coerce")
            if "students_risk_2030" in df_students_2030.columns:
                pass  # keep as string
        except Exception:
            df_students_2030 = None

    # merge student 2030 predictions to school_latest (before merging to school_pt)
    if df_students_2030 is not None and GAKKU_COL in df_students_2030.columns:
        merge_cols_2030 = [GAKKU_COL] + [c for c in ["students_pred_2030", "students_risk_2030", "students_change_to_2030"] if c in df_students_2030.columns]
        df_school_latest = df_school_latest.merge(
            df_students_2030[merge_cols_2030].drop_duplicates(subset=[GAKKU_COL]),
            on=GAKKU_COL,
            how="left"
        )

    # --- school points
    school_pt = gpd.read_file(SCHOOL_PT_SHP)
    school_pt = to_epsg4326(school_pt)

    # Get school name column from school_pt
    SCHOOL_NAME_COL_PT = pick_col_by_candidates(school_pt.columns, SCHOOL_NAME_CANDIDATES, fallback_index=4)

    # join risk to school points by (ID) if possible, else by school name
    # Include 2030 predictions if available
    merge_cols = [ID_COL, SCHOOL_NAME_COL_CSV, GAKKU_COL, STUDENTS_COL, CLASSES_COL, "student_risk", "class_risk", "integration_type"]
    if df_students_2030 is not None and GAKKU_COL in df_school_latest.columns:
        merge_cols.extend([c for c in ["students_pred_2030", "students_risk_2030", "students_change_to_2030"] if c in df_school_latest.columns])
    
    if ID_COL in school_pt.columns and ID_COL in df_school_latest.columns:
        school_pt = school_pt.merge(
            df_school_latest[merge_cols],
            on=ID_COL,
            how="left",
            suffixes=("", "_csv")
        )
    else:
        # fallback: join by school name
        if SCHOOL_NAME_COL_PT and SCHOOL_NAME_COL_PT in school_pt.columns and SCHOOL_NAME_COL_CSV in df_school_latest.columns:
            # Remove ID_COL from merge cols for name-based merge
            name_merge_cols = [c for c in merge_cols if c != ID_COL]
            school_pt = school_pt.merge(
                df_school_latest[name_merge_cols],
                left_on=SCHOOL_NAME_COL_PT,
                right_on=SCHOOL_NAME_COL_CSV,
                how="left"
            )

    # --- bear csv
    df_bear = safe_read_csv(BEAR_CSV)
    df_bear.columns = [c.strip() for c in df_bear.columns]
    if "year" not in df_bear.columns:
        # try to infer
        for cand in ["fiscal year", "fiscalyear", "Year"]:
            if cand in df_bear.columns:
                df_bear = df_bear.rename(columns={cand: "year"})
                break
    df_bear["year"] = pd.to_numeric(df_bear["year"], errors="coerce").astype("Int64")

    # we need gakku + year per record; assume 1 row per incident
    df_bear_inc = df_bear.dropna(subset=[GAKKU_COL, "year"]).copy()
    bear_counts = (
        df_bear_inc
        .groupby([GAKKU_COL, "year"], as_index=False)
        .size()
        .rename(columns={"size": "bear_cnt"})
    )

    # 3-year total (2023-2025)
    bear_3y = (
        bear_counts[bear_counts["year"].isin([2023, 2024, 2025])]
        .groupby(GAKKU_COL, as_index=False)["bear_cnt"].sum()
        .rename(columns={"bear_cnt": "bear_3y"})
    )

    # --- load bear_risk_by_gakku.csv if exists
    bear_risk = None
    if os.path.exists(BEAR_RISK_BY_GAKKU_CSV):
        try:
            bear_risk = safe_read_csv(BEAR_RISK_BY_GAKKU_CSV)
            bear_risk.columns = [c.strip() for c in bear_risk.columns]
            # Ensure numeric columns
            for col in ["bear_3y", "pop_0_11", "bear_per_1000_children", "bear_risk_0_100"]:
                if col in bear_risk.columns:
                    bear_risk[col] = pd.to_numeric(bear_risk[col], errors="coerce")
            # Rename bear_per_1000_children to bear_per_1000_kids for consistency
            if "bear_per_1000_children" in bear_risk.columns:
                bear_risk = bear_risk.rename(columns={"bear_per_1000_children": "bear_per_1000_kids"})
        except Exception:
            bear_risk = None
    
    # Fallback: compute bear risk from bear.csv if bear_risk_by_gakku.csv not available
    if bear_risk is None:
        # compute bear risk per 1000 kids (use latest pop_0_11)
        if "pop_0_11" in df_pop_latest.columns:
            tmp = df_pop_latest[[GAKKU_COL, "pop_0_11"]].copy()
        else:
            tmp = df_pop_latest[[GAKKU_COL]].copy()
            tmp["pop_0_11"] = np.nan

        bear_risk = tmp.merge(bear_3y, on=GAKKU_COL, how="left")
        bear_risk["bear_3y"] = bear_risk["bear_3y"].fillna(0)
        bear_risk["bear_per_1000_kids"] = np.where(
            bear_risk["pop_0_11"] > 0,
            bear_risk["bear_3y"] / bear_risk["pop_0_11"] * 1000.0,
            0
        )
        bear_risk["bear_risk_0_100"] = normalize_0_100(bear_risk["bear_per_1000_kids"])

    # --- bear points
    bear_pt = gpd.read_file(BEAR_PT_SHP)
    bear_pt = to_epsg4326(bear_pt)
    # ensure year exists
    if BEAR_YEAR_COL_PT not in bear_pt.columns:
        for cand in ["Year", "fiscalyear", "fiscal year"]:
            if cand in bear_pt.columns:
                bear_pt = bear_pt.rename(columns={cand: BEAR_YEAR_COL_PT})
                break

    # --- land price points
    land_pt = None
    df_land_long_map = {}  # ChikaID -> long df

    if os.path.exists(LAND_SHP):
        land_pt = gpd.read_file(LAND_SHP)
        land_pt = to_epsg4326(land_pt)
        # Expect year columns 1983..2025
        year_cols = [c for c in land_pt.columns if str(c).isdigit()]
        for _, r in land_pt.iterrows():
            key = r.get("ChikaID", r.get("chikaid", r.get("ID", None)))
            if key is None:
                continue
            vals = []
            for yc in year_cols:
                v = pd.to_numeric(r[yc], errors="coerce")
                if pd.notna(v):
                    vals.append((int(yc), float(v)))
            if vals:
                df_land_long_map[key] = pd.DataFrame(vals, columns=["year", "price"]).sort_values("year")
    elif os.path.exists(LAND_CSV):
        df_land = safe_read_csv(LAND_CSV)
        df_land.columns = [c.strip() for c in df_land.columns]
        # If csv has lon/lat or x/y, we can map. If not, map cannot show.
        lon_col = None
        lat_col = None
        for a, b in [("lon", "lat"), ("longitude", "latitude"), ("x", "y")]:
            if a in df_land.columns and b in df_land.columns:
                lon_col, lat_col = a, b
                break

        year_cols = [c for c in df_land.columns if str(c).isdigit()]
        if lon_col and lat_col:
            land_pt = gpd.GeoDataFrame(
                df_land,
                geometry=gpd.points_from_xy(df_land[lon_col], df_land[lat_col]),
                crs="EPSG:4326"
            )
        else:
            land_pt = None

        # long series map (if has ChikaID)
        key_col = "ChikaID" if "ChikaID" in df_land.columns else ("chikaid" if "chikaid" in df_land.columns else None)
        if key_col:
            for _, r in df_land.iterrows():
                key = r.get(key_col)
                vals = []
                for yc in year_cols:
                    v = pd.to_numeric(r[yc], errors="coerce")
                    if pd.notna(v):
                        vals.append((int(yc), float(v)))
                if vals:
                    df_land_long_map[key] = pd.DataFrame(vals, columns=["year", "price"]).sort_values("year")

    # --- merge into polygon gdf
    gdf2 = gdf.copy()

    # add latest pop columns
    if not df_pop_latest.empty and GAKKU_COL in df_pop_latest.columns and GAKKU_COL in gdf2.columns:
        pop_cols = [c for c in ["pop_total", "pop_0_11", "pop_12_18", "pop_19_64", "pop_65_over"] if c in df_pop_latest.columns]
        if pop_cols:
            gdf2 = gdf2.merge(
                df_pop_latest[[GAKKU_COL] + pop_cols],
                on=GAKKU_COL,
                how="left"
            )

    # add trends
    if GAKKU_COL in gdf2.columns:
        if not df_pop_trend.empty and GAKKU_COL in df_pop_trend.columns:
            gdf2 = gdf2.merge(df_pop_trend, on=GAKKU_COL, how="left")
        if not df_students_trend.empty and GAKKU_COL in df_students_trend.columns:
            gdf2 = gdf2.merge(df_students_trend, on=GAKKU_COL, how="left")
        if not bear_risk.empty and GAKKU_COL in bear_risk.columns:
            bear_cols = [c for c in ["bear_risk_0_100", "bear_per_1000_kids", "bear_3y"] if c in bear_risk.columns]
            if bear_cols:
                gdf2 = gdf2.merge(bear_risk[[GAKKU_COL] + bear_cols], on=GAKKU_COL, how="left")

    # fill numeric
    for c in ["bear_risk_0_100", "bear_per_1000_kids", "bear_3y"]:
        if c in gdf2.columns:
            gdf2[c] = pd.to_numeric(gdf2[c], errors="coerce").fillna(0)

    # --- load future projections (2030)
    df_pop_2030 = None
    if os.path.exists(POP_2030_CSV):
        try:
            df_pop_2030 = safe_read_csv(POP_2030_CSV)
            df_pop_2030.columns = [c.strip() for c in df_pop_2030.columns]
            # merge district name to gakku if needed
            if "district" in df_pop_2030.columns and GAKKU_COL not in df_pop_2030.columns:
                df_pop_2030[GAKKU_COL] = df_pop_2030["district"]
            if "pop_2030" in df_pop_2030.columns:
                df_pop_2030["pop_2030"] = pd.to_numeric(df_pop_2030["pop_2030"], errors="coerce")
            if "pop_2025" in df_pop_2030.columns:
                df_pop_2030["pop_2025"] = pd.to_numeric(df_pop_2030["pop_2025"], errors="coerce")
            # 0-11人口用のカラムも数値型に変換
            for c in ["pop_0_11_2025", "pop_0_11_2030", "pop_0_11"]:
                if c in df_pop_2030.columns:
                    df_pop_2030[c] = pd.to_numeric(df_pop_2030[c], errors="coerce")
        except Exception:
            df_pop_2030 = None

    # merge 2030 projection to gdf2
    if df_pop_2030 is not None and not df_pop_2030.empty and GAKKU_COL in df_pop_2030.columns and GAKKU_COL in gdf2.columns:
        merge_cols = [GAKKU_COL]
        # 総人口と0-11人口の両方をマージ
        for c in ["pop_2030", "pop_2025", "pop_0_11_2025", "pop_0_11_2030", "change_2025_2030", "trend_slope"]:
            if c in df_pop_2030.columns:
                merge_cols.append(c)
        if len(merge_cols) > 1:  # GAKKU_COL以外に列がある場合のみマージ
            gdf2 = gdf2.merge(df_pop_2030[merge_cols], on=GAKKU_COL, how="left")

    # --- load land price long format if exists
    if os.path.exists(LAND_PRICE_LONG_CSV) and not df_land_long_map:
        try:
            df_land_long = safe_read_csv(LAND_PRICE_LONG_CSV)
            df_land_long.columns = [c.strip() for c in df_land_long.columns]
            if "ChikaID" in df_land_long.columns and "year" in df_land_long.columns and "price" in df_land_long.columns:
                for key, group in df_land_long.groupby("ChikaID"):
                    df_land_long_map[key] = group[["year", "price"]].sort_values("year")
        except Exception:
            pass

    # --- load pop_by_gakku_year.csv for bar charts
    df_pop_by_gakku_year = None
    if os.path.exists(POP_BY_GAKKU_YEAR_CSV):
        try:
            df_pop_by_gakku_year = safe_read_csv(POP_BY_GAKKU_YEAR_CSV)
            df_pop_by_gakku_year.columns = [c.strip() for c in df_pop_by_gakku_year.columns]
            # Ensure year column is correct
            year_col = None
            for cand in ["fiscalyear", "fiscal_year", "fiscal year", "year"]:
                if cand in df_pop_by_gakku_year.columns:
                    year_col = cand
                    break
            if year_col:
                df_pop_by_gakku_year[year_col] = pd.to_numeric(df_pop_by_gakku_year[year_col], errors="coerce")
            # Convert population columns to numeric
            for col in ["pop_total", "pop_0_11", "pop_12_18", "pop_19_64", "pop_65_over"]:
                if col in df_pop_by_gakku_year.columns:
                    df_pop_by_gakku_year[col] = pd.to_numeric(df_pop_by_gakku_year[col], errors="coerce")
        except Exception:
            df_pop_by_gakku_year = None

    return gdf2, grp_pop, grp_students, bear_counts, school_pt, bear_pt, land_pt, df_land_long_map, latest_year_pop, latest_year_school, df_pop_2030, df_students_2030, df_pop_by_gakku_year

# ======================================================
# Load data (must be before sidebar to check available data)
# ======================================================
gdf, grp_pop, grp_students, bear_counts, school_pt, bear_pt, land_pt, df_land_long_map, latest_year_pop, latest_year_school, df_pop_2030, df_students_2030, df_pop_by_gakku_year = load_all()

# ======================================================
# UI: sidebar
# ======================================================
st.title("District Population / School / Bear / Land Price Viewer")

with st.sidebar:
    st.header("Layers")
    show_district = st.checkbox("District polygons", value=True)
    show_school = st.checkbox("Schools (square)", value=True)
    show_bear = st.checkbox("Bears (circle)", value=False)
    show_land = st.checkbox("Land price points (triangle)", value=False)

    st.header("人口polygon表示モード")
    color_by_options = [
        "総人口（2025）",
    ]
    # Add 2030 projection option if available
    if df_pop_2030 is not None and "pop_2030" in df_pop_2030.columns:
        color_by_options.append("将来推計（2030 0–11）")
    color_by_options.append("熊risk×人口")
    
    color_by = st.radio(
        "表示モード",
        color_by_options,
        index=1
    )

    st.header("学校point表示")
    school_display_mode = st.radio(
        "表示方法",
        ["児童数×学級数（点の色）", "児童数×学級数（下に表）"],
        index=0
    )

# ======================================================
# Layout: Left (charts) / Right (map)
# ======================================================
left, right = st.columns([0.44, 0.56], gap="large")

# ======================================================
# State: selected gakku (by click only)
# ======================================================
if GAKKU_COL in gdf.columns and not gdf.empty:
    all_gakku = sorted([g for g in gdf[GAKKU_COL].dropna().unique().tolist()])
else:
    all_gakku = []
    st.error(f"Error: '{GAKKU_COL}' column not found in district shapefile. Available columns: {list(gdf.columns) if not gdf.empty else 'No data'}")
selected_gakku = all_gakku[0] if all_gakku else None  # Initialize with first gakku

with left:
    st.subheader("Attributes / Charts")

# ======================================================
# Map setup
# ======================================================
# Center
centroid = gdf.geometry.centroid
center = [float(centroid.y.mean()), float(centroid.x.mean())]

m = folium.Map(location=center, zoom_start=11, tiles=None)

# Aerial ONLY
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery",
    name="Aerial",
    overlay=False,
    control=False,  # no need to switch
).add_to(m)

# --- polygon style
# Up=RED, Down=BLUE, Flat=YELLOW
trend_color = {"Up": "#e74c3c", "Down": "#3498db", "Flat": "#f1c40f"}

# Default style_polygon function (will be overridden if needed)
def style_polygon(feature):
    prop = feature["properties"]
    # default fill
    fill = "#cccccc"
    
    if color_by == "熊risk×人口":
        # gradient buckets for bear risk
        v = float(prop.get("bear_risk_0_100", 0) or 0)
        if v >= 80:
            fill = "#7f0000"
        elif v >= 60:
            fill = "#b30000"
        elif v >= 40:
            fill = "#e34a33"
        elif v >= 20:
            fill = "#fdbb84"
        else:
            fill = "#fee8c8"

    return {
        "fillColor": fill,
        "color": "#999999",     # outline only light gray
        "weight": 1,
        "fillOpacity": 0.75,  # より見やすくするために不透明度を上げる
    }

# For numeric color-by (pop), build choropleth-like bins
if color_by in ["総人口（2025）", "将来推計（2030 0–11）"]:
    col = None
    if color_by == "総人口（2025）":
        # pop_2025を優先、なければpop_totalを試す
        if "pop_2025" in gdf.columns:
            col = "pop_2025"
        elif "pop_total" in gdf.columns:
            col = "pop_total"
        else:
            col = None
    elif color_by == "将来推計（2030 0–11）":
        col = "pop_2030" if "pop_2030" in gdf.columns else None


    if col and col in gdf.columns:
        vals = pd.to_numeric(gdf[col], errors="coerce").fillna(0)
        # bins
        qs = np.quantile(vals, [0, 0.2, 0.4, 0.6, 0.8, 1.0]).tolist()
        palette = ["#de2d26", "#fc9272", "#fcbba1", "#fee0d2", "#fff5f0"]  # dark -> light (多い順に濃い色から薄い色へ)
        
        # すべての値が同じ場合の処理
        if qs[0] == qs[-1]:
            # すべて同じ値の場合は中間色を使用
            default_color = palette[len(palette) // 2]
        else:
            default_color = None

        def style_polygon(feature):
            props = feature.get("properties", {})
            # カラム名のバリエーションを試す
            v = None
            if col in props:
                v = props[col]
            else:
                # 大文字小文字を無視して検索
                for key in props.keys():
                    if key.lower() == col.lower():
                        v = props[key]
                        break
            
            if v is None:
                v = 0
            else:
                try:
                    v = float(v)
                    if pd.isna(v):
                        v = 0
                except (ValueError, TypeError):
                    v = 0
            
            # すべての値が同じ場合は中間色を返す
            if default_color is not None:
                return {
                    "fillColor": default_color,
                    "color": "#999999",
                    "weight": 1,
                    "fillOpacity": 0.75,
                }
            
            # find bin (多い値ほど濃い色になるように)
            idx = len(palette) - 1  # default to lightest
            for i in range(len(qs) - 1, 0, -1):  # 大きい値から順にチェック (i = 5, 4, 3, 2, 1)
                if v >= qs[i]:
                    # qs[5] (max) → palette[0] (darkest)
                    # qs[4] (80%) → palette[1]
                    # qs[3] (60%) → palette[2]
                    # qs[2] (40%) → palette[3]
                    # qs[1] (20%) → palette[4] (lightest)
                    idx = len(qs) - 1 - i
                    break
            idx = min(max(0, idx), len(palette)-1)
            return {
                "fillColor": palette[idx],
                "color": "#999999",
                "weight": 1,
                "fillOpacity": 0.75,  # より見やすくするために不透明度を上げる
            }

# --- tooltip (keep simple, English)
tooltip_fields = []
tooltip_aliases = []
if GAKKU_COL in gdf.columns:
    tooltip_fields.append(GAKKU_COL)
    tooltip_aliases.append("gakku")
for f, a in [
    ("pop_total", "pop_total"),
    ("pop_0_11", "pop_0_11"),
    ("pop_65_over", "pop_65_over"),
    ("pop_trend", "pop_trend"),
    ("students_trend", "students_trend"),
    ("bear_risk_0_100", "bear_risk_0_100"),
    ("pop_2030", "pop_2030"),
]:
    if f in gdf.columns:
        tooltip_fields.append(f)
        tooltip_aliases.append(a)

# tooltip_fieldsが空でない場合のみtooltipを作成
if tooltip_fields:
    tooltip = folium.GeoJsonTooltip(fields=tooltip_fields, aliases=tooltip_aliases)
else:
    tooltip = None

# --- add district polygons (NO highlight_function -> avoid "selection rectangle" feel)
if show_district:
    folium_kwargs = {
        "data": gdf,
        "name": "districts",
        "style_function": style_polygon,
        "highlight_function": None,  # important
    }
    if tooltip is not None:
        folium_kwargs["tooltip"] = tooltip
    folium.GeoJson(**folium_kwargs).add_to(m)

# ======================================================
# Points: schools (square), bears (small circle), land (triangle)
# ======================================================
# School square color by student risk (High red, Low blue, else green)
risk_to_color = {"High": "#e74c3c", "Low": "#3498db", "Watch": "#2ecc71"}
integration_type_colors = {
    "High / High": "#8b0000",      # 統廃合候補（最優先）- 暗赤
    "High / Low": "#ff6600",       # 配置の歪み - オレンジ
    "Watch / Watch": "#ffd700",    # 要モニタリング - 黄色
    "Low / Low": "#2ecc71",        # 当面問題なし - 緑
}
# 2030年予測リスク色: High=赤 / Watch=緑 / Low=青
risk_2030_colors = {
    "High": "#e74c3c",  # 高リスク（統廃合検討レベル）< 100人 - 赤
    "Watch": "#2ecc71",  # 要注意 100-199人 - 緑
    "Low": "#3498db",   # 安定 200人以上 - 青
}

if show_school and (school_pt is not None) and (not school_pt.empty):
    for _, r in school_pt.iterrows():
        latlon = geom_to_latlon(r.geometry)
        if latlon is None:
            continue

        name = r.get("school_name", r.get("school_nam", "school"))
        gakku = r.get("gakku", "")
        sid = r.get("ID", "")
        students = r.get(STUDENTS_COL, None)
        classes = r.get(CLASSES_COL, None)
        student_risk_val = r.get("student_risk", "Watch")
        class_risk_val = r.get("class_risk", "Watch")
        integration_type = r.get("integration_type", "Watch / Watch")
        students_pred_2030 = r.get("students_pred_2030", None)
        students_risk_2030 = r.get("students_risk_2030", "Watch")

        # Determine color based on display mode
        if school_display_mode == "児童数×学級数（点の色）":
            # Color by risk level (combine student and class risk)
            if student_risk_val == "High" or class_risk_val == "High":
                school_color = "#e74c3c"
            elif student_risk_val == "Low" and class_risk_val == "Low":
                school_color = "#3498db"
            else:
                school_color = "#2ecc71"
            popup_text = f"<b>{name}</b><br>gakku: {gakku}<br>ID: {sid}<br>児童数: {fmt_int(students)}<br>学級数: {fmt_int(classes)}<br>児童数リスク: {student_risk_val}<br>学級数リスク: {class_risk_val}<br>統廃合タイプ: {integration_type}"
        else:  # 児童数×学級数（下に表）
            # Use neutral color when showing table
            school_color = "#808080"  # Gray
            popup_text = f"<b>{name}</b><br>gakku: {gakku}<br>ID: {sid}<br>児童数: {fmt_int(students)}<br>学級数: {fmt_int(classes)}<br>統廃合タイプ: {integration_type}"

        folium.RegularPolygonMarker(
            location=list(latlon),
            number_of_sides=4,  # square
            radius=6,
            color=school_color,
            weight=1,
            fill=True,
            fill_color=school_color,
            fill_opacity=0.95,
            popup=folium.Popup(
                popup_text,
                max_width=320,
            ),
        ).add_to(m)


if show_bear and (bear_pt is not None) and (not bear_pt.empty):
    bear_color_map = {2023: "orange", 2024: "red", 2025: "purple"}

    for _, r in bear_pt.iterrows():
        latlon = geom_to_latlon(r.geometry)
        if latlon is None:
            continue

        try:
            year = int(r.get("year", r.get(BEAR_YEAR_COL_PT, 0)))
        except Exception:
            year = 0

        color = bear_color_map.get(year, "black")
        memo = r.get("memo1", r.get("memo", ""))
        gakku = r.get("gakku", "")
        
        # Get additional attributes if available
        attributes = []
        for attr_col in ["date", "time", "location", "type", "size"]:
            if attr_col in r.index and pd.notna(r.get(attr_col)):
                attributes.append(f"<b>{attr_col}:</b> {r.get(attr_col)}")
        
        attr_text = "<br>".join(attributes) if attributes else ""
        if attr_text:
            attr_text = "<br>" + attr_text

        folium.CircleMarker(
            location=list(latlon),
            radius=3,              # ←小さく
            color=color,
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(
                f"<b>熊情報</b><br><b>年:</b> {year}<br><b>学区:</b> {gakku}<br><b>メモ:</b> {memo}{attr_text}",
                max_width=320,
            ),
        ).add_to(m)


# ------------------------------------------------------
# 地価公示ポイント（三角）
if show_land and (land_pt is not None) and (not land_pt.empty):
    # choose key for series
    key_col = None
    for cand in ["ChikaID", "chikaid", "ID"]:
        if cand in land_pt.columns:
            key_col = cand
            break

    for _, r in land_pt.iterrows():
        latlon = geom_to_latlon(r.geometry)
        if latlon is None:
            continue

        key = r.get(key_col, "") if key_col else ""
        gakku = r.get("gakku", "")

        # Get latest price from series for reference price calculation
        latest_price_per_m2 = None
        if key and key in df_land_long_map:
            latest_price_data = df_land_long_map[key]
            if not latest_price_data.empty:
                latest_price_per_m2 = latest_price_data["price"].iloc[-1]

        # Calculate reference prices (rough estimates)
        # 戸建て: 100m² × 地価/m² × 1.5 (建物価格含む)
        # マンション: 70m² × 地価/m² × 1.3 (建物価格含む)
        ref_price_text = ""
        if latest_price_per_m2 and pd.notna(latest_price_per_m2) and latest_price_per_m2 > 0:
            kodate_price = latest_price_per_m2 * 100 * 1.5  # 戸建て目安
            mansion_price = latest_price_per_m2 * 70 * 1.3  # マンション目安
            ref_price_text = f"<br>参考価格（目安）:<br>戸建て（100m²）: 約{int(kodate_price/10000)}万円<br>マンション（70m²）: 約{int(mansion_price/10000)}万円<br>地価/m²: {int(latest_price_per_m2)}円"

        folium.RegularPolygonMarker(
            location=list(latlon),
            number_of_sides=3,   # triangle
            radius=6,
            color="#f1c40f",
            weight=1,
            fill=True,
            fill_color="#f1c40f",
            fill_opacity=0.95,
            popup=folium.Popup(
                f"<b>地価公示ポイント</b><br>gakku: {gakku}<br>ID: {key}{ref_price_text}",
                max_width=320,
            ),
        ).add_to(m)



# ======================================================
# Render map
# ======================================================
with right:
    st.subheader("Map")
    map_data = st_folium(m, width=None, height=620)

# ======================================================
# Click handling -> update selected_gakku (if polygon clicked)
# ======================================================
clicked_props = None
if map_data and isinstance(map_data, dict):
    clicked_props = map_data.get("last_active_drawing")
    # Depending on streamlit-folium version, you may have:
    # - "last_active_drawing" for GeoJson feature
    # - "last_object_clicked" for markers
    if clicked_props and isinstance(clicked_props, dict):
        props = clicked_props.get("properties", None)
        if props and props.get(GAKKU_COL) in all_gakku:
            selected_gakku = props.get(GAKKU_COL)

# ======================================================
# Left panel: charts & ranking (separate scroll)
# ======================================================
with left:
    # independent scroll area
    with st.container(height=760):
        st.markdown(f"### Selected gakku: `{selected_gakku}`")

        # latest pop row
        if GAKKU_COL in gdf.columns and selected_gakku:
            pop_latest_row = (
                gdf[gdf[GAKKU_COL] == selected_gakku]
                .drop(columns="geometry")
                .iloc[0]
                .to_dict()
                if (gdf[gdf[GAKKU_COL] == selected_gakku].shape[0] > 0)
                else {}
            )
        else:
            pop_latest_row = {}

        # Show key numbers - use pop_by_gakku_year.csv if available
        display_data = {}
        if df_pop_by_gakku_year is not None and GAKKU_COL in df_pop_by_gakku_year.columns:
            year_col_pop = "fiscal year" if "fiscal year" in df_pop_by_gakku_year.columns else ("year" if "year" in df_pop_by_gakku_year.columns else None)
            if year_col_pop:
                latest_year_data = df_pop_by_gakku_year[df_pop_by_gakku_year[GAKKU_COL] == selected_gakku].copy()
                if not latest_year_data.empty:
                    latest_year_data = latest_year_data.sort_values(year_col_pop, ascending=False)
                    latest_row_year = latest_year_data.iloc[0]
                    display_data = {
                        "pop_total": fmt_int(latest_row_year.get("pop_total", np.nan)),
                        "pop_0_11": fmt_int(latest_row_year.get("pop_0_11", np.nan)),
                        "pop_65_over": fmt_int(latest_row_year.get("pop_65_over", np.nan)),
                        "pop_trend": pop_latest_row.get("pop_trend", "-"),
                        "students_trend": pop_latest_row.get("students_trend", "-"),
                        "bear_risk_0_100": round(float(pop_latest_row.get("bear_risk_0_100", 0) or 0), 1),
                    }
        if not display_data:
            display_data = {
                "pop_total": fmt_int(pop_latest_row.get("pop_total", np.nan)),
                "pop_0_11": fmt_int(pop_latest_row.get("pop_0_11", np.nan)),
                "pop_65_over": fmt_int(pop_latest_row.get("pop_65_over", np.nan)),
                "pop_trend": pop_latest_row.get("pop_trend", "-"),
                "students_trend": pop_latest_row.get("students_trend", "-"),
                "bear_risk_0_100": round(float(pop_latest_row.get("bear_risk_0_100", 0) or 0), 1),
            }
        st.write(display_data)

        # Population structure bar chart using pop_by_gakku_year.csv
        if df_pop_by_gakku_year is not None and GAKKU_COL in df_pop_by_gakku_year.columns:
            year_col_pop = "fiscal year" if "fiscal year" in df_pop_by_gakku_year.columns else ("year" if "year" in df_pop_by_gakku_year.columns else None)
            if year_col_pop:
                latest_year_data = df_pop_by_gakku_year[df_pop_by_gakku_year[GAKKU_COL] == selected_gakku].copy()
                if not latest_year_data.empty:
                    latest_year_data = latest_year_data.sort_values(year_col_pop, ascending=False)
                    latest_row_year = latest_year_data.iloc[0]
                    # Convert to dict and ensure numeric values
                    latest_row_dict = latest_row_year.to_dict()
                    # Ensure all population values are properly converted
                    for key in ["pop_0_11", "pop_12_18", "pop_19_64", "pop_65_over"]:
                        if key in latest_row_dict:
                            latest_row_dict[key] = num_or_zero(latest_row_dict[key])
                    st.plotly_chart(plot_population_structure(latest_row_dict), use_container_width=True)
                else:
                    # Fallback to pop_latest_row
                    st.plotly_chart(plot_population_structure(pop_latest_row), use_container_width=True)
            else:
                st.plotly_chart(plot_population_structure(pop_latest_row), use_container_width=True)
        else:
            st.plotly_chart(plot_population_structure(pop_latest_row), use_container_width=True)

        # pop 0-11 trend by year using pop_by_gakku_year.csv
        if df_pop_by_gakku_year is not None and GAKKU_COL in df_pop_by_gakku_year.columns:
            year_col_pop = "fiscal year" if "fiscal year" in df_pop_by_gakku_year.columns else ("year" if "year" in df_pop_by_gakku_year.columns else None)
            if year_col_pop and "pop_0_11" in df_pop_by_gakku_year.columns:
                ts_pop = df_pop_by_gakku_year[df_pop_by_gakku_year[GAKKU_COL] == selected_gakku].sort_values(year_col_pop)
                if not ts_pop.empty:
                    st.plotly_chart(
                        plot_timeseries(ts_pop, year_col_pop, "pop_0_11", "pop_0_11 trend by year"),
                        use_container_width=True
                    )
        elif "pop_0_11" in grp_pop.columns:
            ts_pop = grp_pop[grp_pop[GAKKU_COL] == selected_gakku].sort_values(YEAR_COL_POP)
            if not ts_pop.empty:
                st.plotly_chart(
                    plot_timeseries(ts_pop, YEAR_COL_POP, "pop_0_11", "pop_0_11 trend by year"),
                    use_container_width=True
                )

        # students trend by year (district aggregated)
        ts_stu = grp_students[grp_students[GAKKU_COL] == selected_gakku].sort_values(YEAR_COL_SCHOOL)
        if not ts_stu.empty:
            st.plotly_chart(
                plot_timeseries(ts_stu, YEAR_COL_SCHOOL, STUDENTS_COL, "students (enroll_total) trend by year"),
                use_container_width=True
            )


        # land price chart (if a land point is clicked, show its series)
        land_clicked_key = None
        if map_data and isinstance(map_data, dict):
            obj = map_data.get("last_object_clicked_popup")
            # popup string isn't reliable for key extraction; best is to click land points and later add custom callback.
        # So we show “example” series: pick first point in same gakku if available
        if show_land and df_land_long_map:
            # try get a key in same gakku
            key_candidates = []
            if land_pt is not None and not land_pt.empty and "gakku" in land_pt.columns:
                sub = land_pt[land_pt["gakku"] == selected_gakku]
                if not sub.empty:
                    for cand in ["ChikaID", "chikaid", "ID"]:
                        if cand in sub.columns:
                            key_candidates = sub[cand].dropna().tolist()
                            break
            if key_candidates:
                k = key_candidates[0]
                if k in df_land_long_map:
                    st.plotly_chart(plot_land_price_series(df_land_long_map[k], title="Official land price trend (example point)"), use_container_width=True)
                else:
                    st.info("Land price series map exists, but key not found for this point.")
            else:
                st.info("No land price point for this gakku (or land data not available).")

        # School table (if "下に表" mode is selected)
        if school_display_mode == "児童数×学級数（下に表）" and show_school and (school_pt is not None) and (not school_pt.empty):
            st.markdown("### 学校一覧表（児童数×学級数）")
            school_table_cols = ["school_name" if "school_name" in school_pt.columns else "school_nam",
                                GAKKU_COL, STUDENTS_COL, CLASSES_COL, "student_risk", "class_risk", "integration_type"]
            available_cols = [c for c in school_table_cols if c in school_pt.columns]
            if available_cols:
                df_school_table = school_pt[available_cols].copy()
                # Sort by gakku and students
                df_school_table = df_school_table.sort_values(by=[GAKKU_COL, STUDENTS_COL], ascending=[True, False])
                st.dataframe(df_school_table, use_container_width=True, height=400)

        # District ranking (under charts)
        st.markdown("### District Ranking (top / bottom)")
        # ランキング用のカラム名を決定（色付けと同じロジックを使用）
        rank_metric = None
        if color_by == "総人口（2025）":
            if "pop_2025" in gdf.columns:
                rank_metric = "pop_2025"
            elif "pop_total" in gdf.columns:
                rank_metric = "pop_total"
        elif color_by == "将来推計（2030 0–11）":
            if "pop_2030" in gdf.columns:
                rank_metric = "pop_2030"
        elif color_by == "熊リスク×人口":
            if "bear_risk_0_100" in gdf.columns:
                rank_metric = "bear_risk_0_100"

        if rank_metric and rank_metric in gdf.columns:
            dfr = gdf[[GAKKU_COL, rank_metric]].copy()
            dfr[rank_metric] = pd.to_numeric(dfr[rank_metric], errors="coerce").fillna(0)
            top10 = dfr.sort_values(rank_metric, ascending=False).head(10)
            bot10 = dfr.sort_values(rank_metric, ascending=True).head(10)
            st.write("Top 10")
            st.dataframe(top10, use_container_width=True, height=240)
            st.write("Bottom 10")
            st.dataframe(bot10, use_container_width=True, height=240)
        else:
            st.caption("Ranking is shown for numeric coloring modes (pop / bear risk).")

