# app.py
import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import altair as alt

# -----------------------------
# Page
# -----------------------------
st.set_page_config(page_title="District Risk Viewer", layout="wide")
st.title("District Population / School / Bear / Land Price Viewer")

# -----------------------------
# Paths (EDIT HERE if needed)
# -----------------------------
DISTRICT_SHP = "data/project.shp"          # 学区ポリゴン
SCHOOL_SHP = "data/schoolPT.shp"           # 学校ポイント（任意）
BEAR_SHP = "data/bearPT.shp"               # 熊ポイント（任意）

POP_CSV = "data/jinko2018_2025.csv"        # 人口（0-11 等が入っている想定）
SCHOOL_CSV = "data/school.csv"             # 児童数・学級数
BEAR_CSV = "data/bear.csv"                 # 熊（gakku列がある想定）
LANDPRICE_CSV = "data/chika2.csv"          # 公示価格ポイント（任意）

# -----------------------------
# Constants
# -----------------------------
UP_COLOR = "#e74c3c"     # red
DOWN_COLOR = "#2e6bd6"   # blue
FLAT_COLOR = "#b0b0b0"   # gray

RISK_HIGH = "#e74c3c"
RISK_LOW = "#2e6bd6"
RISK_MID = "#2ecc71"

BEAR_COLOR_MAP = {2023: "orange", 2024: "red", 2025: "purple"}

import re
import numpy as np
import pandas as pd

# -----------------------------
# Helpers
# -----------------------------
def safe_read_csv(path: str):
    for enc in ["utf-8-sig", "utf-8", "shift_jis", "cp932"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)

def trend_class_from_slope(slope: float, thr: float = 2.0):
    if pd.isna(slope):
        return "Flat"
    if slope > thr:
        return "Up"
    if slope < -thr:
        return "Down"
    return "Flat"

def trend_color(trend: str):
    if trend == "Up":
        return UP_COLOR
    if trend == "Down":
        return DOWN_COLOR
    return FLAT_COLOR

def student_risk(n):
    # 暫定ルール（あなたが確認していたものと同じ）
    # 2030予測ではなく「最新実績」ベースで色分け（簡易）
    if n < 100:
        return "High"
    elif n < 200:
        return "Watch"
    return "Low"

def class_risk(c):
    if c <= 6:
        return "High"
    elif c <= 11:
        return "Watch"
    return "Low"

def school_color(student_r, class_r):
    # どちらかがHighならHigh扱い、両方LowならLow、それ以外はMiddle
    if student_r == "High" or class_r == "High":
        return RISK_HIGH
    if student_r == "Low" and class_r == "Low":
        return RISK_LOW
    return RISK_MID

def normalize_0_1(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce")
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series([0.0]*len(s), index=s.index)
    return (s - mn) / (mx - mn)

def shaded_price_chart(df_point: pd.DataFrame, year_col="year", value_col="price"):
    """
    Land price chart with background periods.
    Periods are "example"—adjust to your presentation definition.
    """
    dfp = df_point[[year_col, value_col]].dropna().copy()
    dfp[year_col] = pd.to_numeric(dfp[year_col], errors="coerce")
    dfp[value_col] = pd.to_numeric(dfp[value_col], errors="coerce")
    dfp = dfp.dropna().sort_values(year_col)

    if dfp.empty:
        st.info("No price data for this point.")
        return

    # Periods (edit if you want)
    periods = [
        {"label": "Bubble", "start": 1986, "end": 1991},
        {"label": "Post-bubble", "start": 1992, "end": 2002},
        {"label": "Recent", "start": 2013, "end": 2025},
    ]
    miny, maxy = dfp[value_col].min(), dfp[value_col].max()
    pad = (maxy - miny) * 0.05 if maxy > miny else 1.0

    base = alt.Chart(dfp).encode(
        x=alt.X(f"{year_col}:Q", title="Year", axis=alt.Axis(format="d")),
    )

    # background bands
    bands = alt.Chart(pd.DataFrame(periods)).mark_rect(opacity=0.08, color="gray").encode(
        x=alt.X("start:Q"),
        x2="end:Q",
        y=alt.value(miny - pad),
        y2=alt.value(maxy + pad),
        tooltip=["label:N", "start:Q", "end:Q"],
    )

    line = base.mark_line().encode(
        y=alt.Y(f"{value_col}:Q", title="Land price"),
        tooltip=[alt.Tooltip(f"{year_col}:Q", format="d"), alt.Tooltip(f"{value_col}:Q")],
    )

    points = base.mark_point().encode(
        y=alt.Y(f"{value_col}:Q"),
    )

    text_labels = alt.Chart(pd.DataFrame(periods)).mark_text(align="left", dy=-8, color="gray").encode(
        x="start:Q",
        y=alt.value(maxy + pad),
        text="label:N",
    )

    chart = (bands + line + points + text_labels).properties(height=260)
    st.altair_chart(chart, use_container_width=True)

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data(show_spinner=False)
def load_all():
    # District polygons
    gdf = gpd.read_file(DISTRICT_SHP)
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    else:
        gdf = gdf.to_crs(epsg=4326)

    # population
    df_pop = safe_read_csv(POP_CSV)
    # expected cols: gakku, fiscal year, pop_0_11, pop_65_over etc.
    # (あなたの最新データに合わせて)
    for c in ["fiscal year", "pop_0_11", "pop_65_over", "pop_total"]:
        if c in df_pop.columns:
            df_pop[c] = pd.to_numeric(df_pop[c], errors="coerce")
    if "gakku" not in df_pop.columns and "district" in df_pop.columns:
        df_pop["gakku"] = df_pop["district"]

    # school (students)
    df_school = safe_read_csv(SCHOOL_CSV)
    for c in ["fiscal year", "enroll_total", "classes", "teachers"]:
        if c in df_school.columns:
            df_school[c] = pd.to_numeric(df_school[c], errors="coerce")
    if "gakku" not in df_school.columns and "district" in df_school.columns:
        df_school["gakku"] = df_school["district"]

    # bear csv
    df_bear = None
    try:
        df_bear = safe_read_csv(BEAR_CSV)
        if "year" in df_bear.columns:
            df_bear["year"] = pd.to_numeric(df_bear["year"], errors="coerce").astype("Int64")
        if "gakku" not in df_bear.columns and "district" in df_bear.columns:
            df_bear["gakku"] = df_bear["district"]
    except Exception:
        df_bear = pd.DataFrame()

    # points shapefiles (optional)
    school_pt = gpd.GeoDataFrame()
    bear_pt = gpd.GeoDataFrame()
    try:
        school_pt = gpd.read_file(SCHOOL_SHP).to_crs(epsg=4326)
    except Exception:
        school_pt = gpd.GeoDataFrame()

    try:
        bear_pt = gpd.read_file(BEAR_SHP).to_crs(epsg=4326)
    except Exception:
        bear_pt = gpd.GeoDataFrame()

    # land price csv (optional)
    df_land = None
    try:
        df_land = safe_read_csv(LANDPRICE_CSV)
    except Exception:
        df_land = pd.DataFrame()

    return gdf, df_pop, df_school, df_bear, school_pt, bear_pt, df_land

gdf, df_pop, df_school, df_bear, school_pt, bear_pt, df_land = load_all()

import re
import unicodedata
import numpy as np
import pandas as pd

# -----------------------------
# Build metrics per district (gakku)
# -----------------------------
def _normalize_label(s: str) -> str:
    """列名の表記ゆれ吸収用（全角→半角っぽく、記号ゆれを統一）"""
    if s is None:
        return ""
    s = str(s).strip()

    # 全角数字を半角に寄せる（最低限）
    trans = str.maketrans({
        "０":"0","１":"1","２":"2","３":"3","４":"4","５":"5","６":"6","７":"7","８":"8","９":"9",
        "－":"-","ー":"-","―":"-","‐":"-","–":"-","—":"-",
        "〜":"-","～":"-",
        "歳":"", "才":"", "人":"", "人口":""
    })
    s = s.translate(trans)

    # 記号ゆれ（スペース、括弧など）を削除
    s = re.sub(r"[()\[\]{}　\s]", "", s)
    s = s.replace("_", "-")
    return s.lower()

def _find_age_col(df: pd.DataFrame, age_key: str) -> str | None:
    """
    age_key 例: "0-11", "12-18", "19-64", "65-"
    df内の列からそれっぽい列名を探す
    """
    target = _normalize_label(age_key)

    # 代表的な候補パターン
    patterns = []
    if target == "0-11":
        patterns = [r"0-11", r"0-11.*", r".*0-11", r".*0-11.*", r"^0-11$"]
    elif target == "12-18":
        patterns = [r"12-18", r".*12-18.*"]
    elif target == "19-64":
        patterns = [r"19-64", r".*19-64.*"]
    elif target == "65-":
        patterns = [r"65-", r"65-.*", r".*65-.*", r"^65-$", r"^65\+$", r"65\+"]
    else:
        patterns = [re.escape(target)]

    norm_map = {c: _normalize_label(c) for c in df.columns}

    # 1) 正規化して「完全一致 or 含む」で探す（まずは強め）
    for c, n in norm_map.items():
        if target in n:
            return c

    # 2) パターンで探す
    for c, n in norm_map.items():
        for p in patterns:
            if re.search(p, n):
                return c

    return None

def compute_pop_trend(df_pop: pd.DataFrame, age_group: str = "0-11") -> pd.DataFrame:
    """
    学区ごとの人口トレンド（傾き）を計算
    age_group: "0-11", "12-18", "19-64", "65-" など
    """
    # --- 必須列チェック ---
    required_base = ["gakku", "fiscal year"]
    miss = [c for c in required_base if c not in df_pop.columns]
    if miss:
        raise ValueError(f"df_popに必須列がありません: {miss} / いまの列: {df_pop.columns.tolist()}")

    # --- 年齢区分列を自動発見 ---
    pop_col = _find_age_col(df_pop, age_group)
    if pop_col is None:
        raise ValueError(
            f"{age_group}人口の列が見つかりません（df_popの列名を確認してください）\n"
            f"いまの列: {df_pop.columns.tolist()}\n"
            f"ヒント: 列名が '0-11' や 'pop_0_11' や '0～11歳' など表記ゆれの可能性があります"
        )

    # --- slope by gakku using polyfit ---
    tmp = df_pop.dropna(subset=["gakku", "fiscal year", pop_col]).copy()
    tmp = tmp.sort_values(["gakku", "fiscal year"])

    out = []
    for g, sub in tmp.groupby("gakku"):
        x = sub["fiscal year"].values
        y = sub[pop_col].values
        if len(sub) < 3:
            continue
        slope = np.polyfit(x - x.mean(), y, 1)[0]
        out.append({
            "gakku": str(g),
            f"pop_slope_{age_group}": slope,
            f"pop_trend_{age_group}": trend_class_from_slope(slope, thr=2.0)
        })

    return pd.DataFrame(out)



def compute_bear_risk(df_bear):
    if df_bear is None or df_bear.empty:
        return pd.DataFrame(columns=["gakku", "bear_risk_raw", "bear_risk"])
    tmp = df_bear.dropna(subset=["gakku"]).copy()
    # allow either wide columns bear_2023.. or long (year per row)
    if all(c in tmp.columns for c in ["bear_2023","bear_2024","bear_2025"]):
        tmp["bear_risk_raw"] = tmp[["bear_2023","bear_2024","bear_2025"]].sum(axis=1)
        agg = tmp.groupby("gakku", as_index=False)["bear_risk_raw"].sum()
    elif "year" in tmp.columns:
        # long format
        if "count" in tmp.columns:
            agg = tmp.groupby("gakku", as_index=False)["count"].sum().rename(columns={"count":"bear_risk_raw"})
        else:
            agg = tmp.groupby("gakku", as_index=False).size().rename(columns={"size":"bear_risk_raw"})
    else:
        agg = tmp.groupby("gakku", as_index=False).size().rename(columns={"size":"bear_risk_raw"})

    agg["bear_risk"] = normalize_0_1(agg["bear_risk_raw"])
    return agg

df_pop_trend = compute_pop_trend(df_pop, age_group="0-11")
# df_pop_trend = compute_pop_trend(df_pop, age_group="12-18")
# df_pop_trend = compute_pop_trend(df_pop, age_group="19-64")
# df_pop_trend = compute_pop_trend(df_pop, age_group="65-")

df_students_trend = compute_students_trend(df_school)
df_bear_risk = compute_bear_risk(df_bear)

# Merge for polygon coloring
df_merge = (
    df_pop_trend.merge(df_students_trend, on="gakku", how="outer")
               .merge(df_bear_risk, on="gakku", how="left")
)

# Latest population total for choropleth (optional)
df_pop_latest = df_pop.dropna(subset=["gakku","fiscal year"]).copy()
if not df_pop_latest.empty:
    latest_y = int(df_pop_latest["fiscal year"].max())
    # try total: use pop_total if present; else sum old_*
    if "pop_total" in df_pop_latest.columns:
        pop_latest = df_pop_latest[df_pop_latest["fiscal year"]==latest_y][["gakku","pop_total"]]
    else:
        old_cols = [c for c in df_pop_latest.columns if c.startswith("old_")]
        tmp = df_pop_latest[df_pop_latest["fiscal year"]==latest_y].copy()
        if old_cols:
            tmp["pop_total"] = tmp[old_cols].sum(axis=1)
        pop_latest = tmp[["gakku","pop_total"]]
    df_merge = df_merge.merge(pop_latest, on="gakku", how="left")
else:
    latest_y = None

# Attach merged attributes to gdf (join key: try "gakku" then "district")
JOIN_KEY = None
for k in ["gakku", "district", "gakkuNo", "school_nam"]:
    if k in gdf.columns:
        # prefer gakku / district
        if k in ["gakku","district"]:
            JOIN_KEY = k
            break
        JOIN_KEY = k

# If your polygon has English district name in "district", use that.
# We'll create 'gakku_join' column in gdf for robust join:
gdf = gdf.copy()
if "gakku" in gdf.columns:
    gdf["gakku_join"] = gdf["gakku"].astype(str)
elif "district" in gdf.columns:
    gdf["gakku_join"] = gdf["district"].astype(str)
else:
    # fallback: use a likely name column; edit here if needed
    gdf["gakku_join"] = gdf.iloc[:,0].astype(str)

df_merge["gakku"] = df_merge["gakku"].astype(str)
gdf = gdf.merge(df_merge, left_on="gakku_join", right_on="gakku", how="left")

# Combined risk example: bear risk high + pop down => higher
gdf["combined_risk"] = (gdf["bear_risk"].fillna(0) + (gdf["pop_trend"]=="Down").astype(int)*0.5).clip(0, 1.5)

# -----------------------------
# Sidebar: controls
# -----------------------------
with st.sidebar:
    st.subheader("Layers")
    show_district = st.checkbox("District polygons", True)
    show_school = st.checkbox("Schools (squares)", True)
    show_bear = st.checkbox("Bears (circles)", True)
    show_land = st.checkbox("Land price (triangles)", True)

    st.subheader("Polygon coloring")
    mode = st.selectbox(
        "Color by",
        [
            "Population trend (0–11) Up/Down/Flat",
            "Students trend (enroll_total) Up/Down/Flat",
            "Bear × Population combined risk (0–1)",
            "Population total (latest year)",
        ],
        index=0
    )

    st.subheader("Select district for charts")
    district_options = sorted(gdf["gakku_join"].dropna().unique())
    sel_district = st.selectbox("District (gakku)", district_options)

    st.subheader("Land price point (for chart)")
    # Provide dropdown for land price point if data exists
    land_point_id = None
    if df_land is not None and not df_land.empty:
        # Try to guess id/point name columns
        cand_cols = [c for c in ["ID","id","point_id","name","地点","地点名","標準地番号"] if c in df_land.columns]
        id_col = cand_cols[0] if cand_cols else df_land.columns[0]
        land_ids = df_land[id_col].dropna().unique().tolist()
        land_point_id = st.selectbox(f"Land point ({id_col})", land_ids)
    else:
        st.caption("No land price csv loaded (chika2.csv).")

# -----------------------------
# Layout: left info + right map (scroll separated)
# -----------------------------
left, right = st.columns([0.42, 0.58], gap="large")

# -----------------------------
# LEFT: attributes + charts
# -----------------------------
with left:
    st.subheader("Selected district")
    g_row = gdf[gdf["gakku_join"] == sel_district].head(1)
    if g_row.empty:
        st.warning("District not found in polygon data.")
    else:
        r = g_row.iloc[0]
        cols_show = [
            ("District", "gakku_join"),
            ("Pop trend", "pop_trend"),
            ("Pop slope", "pop_slope"),
            ("Students trend", "students_trend"),
            ("Students slope", "students_slope"),
            ("Bear risk raw", "bear_risk_raw"),
            ("Bear risk (0-1)", "bear_risk"),
            ("Combined risk", "combined_risk"),
            ("Population total (latest)", "pop_total"),
        ]
        info = {}
        for label, key in cols_show:
            if key in g_row.columns:
                info[label] = r.get(key, None)
        st.json(info)

    # Population time series (0-11, 65+) for selected district
    st.subheader("Population time series")
    dp = df_pop[df_pop["gakku"].astype(str) == str(sel_district)].copy()
    if dp.empty:
        st.caption("No population rows for this district.")
    else:
        # build age bins if missing
        if "pop_0_11" in dp.columns:
            chart_df = dp[["fiscal year","pop_0_11"]].dropna().sort_values("fiscal year")
            st.line_chart(chart_df.set_index("fiscal year"))
        if "pop_65_over" in dp.columns:
            chart_df2 = dp[["fiscal year","pop_65_over"]].dropna().sort_values("fiscal year")
            st.line_chart(chart_df2.set_index("fiscal year"))

    # Students time series for selected district (sum of schools)
    st.subheader("Students time series (sum by district)")
    ds = df_school[df_school["gakku"].astype(str) == str(sel_district)].copy()
    if ds.empty:
        st.caption("No school rows for this district.")
    else:
        ts = ds.groupby("fiscal year", as_index=False)["enroll_total"].sum().dropna().sort_values("fiscal year")
        st.line_chart(ts.set_index("fiscal year"))

    # Land price chart (selected point)
    st.subheader("Land price time series (selected point)")
    if df_land is not None and (not df_land.empty) and land_point_id is not None:
        # guess columns
        # year col candidates
        year_candidates = [c for c in ["year","fiscal_year","年度","y"] if c in df_land.columns]
        value_candidates = [c for c in ["price","land_price","公示価格","value","p"] if c in df_land.columns]
        id_candidates = [c for c in ["ID","id","point_id","name","地点","地点名","標準地番号"] if c in df_land.columns]
        id_col = id_candidates[0] if id_candidates else df_land.columns[0]
        year_col = year_candidates[0] if year_candidates else None
        value_col = value_candidates[0] if value_candidates else None

        if year_col is None or value_col is None:
            st.info("chika2.csv needs year column and price column. Please rename/confirm columns.")
        else:
            dfp = df_land[df_land[id_col] == land_point_id].copy()
            shaded_price_chart(dfp, year_col=year_col, value_col=value_col)
            st.caption("Tip: You can add 'house / condo reference price' next to this chart later.")
    else:
        st.caption("No land price point selected.")

# -----------------------------
# RIGHT: map
# -----------------------------
with right:
    st.subheader("Map")

    centroid = gdf.geometry.centroid
    center = [float(centroid.y.mean()), float(centroid.x.mean())]

    m = folium.Map(location=center, zoom_start=11, tiles=None)

    # Aerial base only (keep OSM optional if you want)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Aerial",
        overlay=False,
        control=False,
    ).add_to(m)

    # ------------------------------------------------------
    # Polygon styling
    # ------------------------------------------------------
    def style_func(feature):
        props = feature["properties"]
        # border only light gray
        border = "#d0d0d0"

        if mode.startswith("Population trend"):
            tr = props.get("pop_trend", "Flat")
            fill = trend_color(tr)
            return {"fillColor": fill, "color": border, "weight": 1, "fillOpacity": 0.55}

        if mode.startswith("Students trend"):
            tr = props.get("students_trend", "Flat")
            fill = trend_color(tr)
            return {"fillColor": fill, "color": border, "weight": 1, "fillOpacity": 0.55}

        if mode.startswith("Bear ×"):
            # grayscale by risk (0-1) -> simple ramp (light to dark)
            v = props.get("combined_risk", 0)
            try:
                v = float(v)
            except:
                v = 0.0
            # map v to gray intensity
            # v in [0,1.5] -> intensity [240..80]
            v = max(0.0, min(1.5, v))
            intensity = int(240 - (160 * (v / 1.5)))
            fill = f"#{intensity:02x}{intensity:02x}{intensity:02x}"
            return {"fillColor": fill, "color": border, "weight": 1, "fillOpacity": 0.60}

        if mode.startswith("Population total"):
            v = props.get("pop_total", None)
            try:
                v = float(v)
            except:
                v = None
            if v is None or np.isnan(v):
                fill = "#cccccc"
            else:
                # simple quantile-ish manual scaling using global
                vals = pd.to_numeric(gdf["pop_total"], errors="coerce")
                vmax = float(vals.max()) if vals.notna().any() else 1.0
                ratio = min(1.0, v / vmax) if vmax > 0 else 0
                intensity = int(240 - 160 * ratio)  # larger -> darker
                fill = f"#{intensity:02x}{intensity:02x}{intensity:02x}"
            return {"fillColor": fill, "color": border, "weight": 1, "fillOpacity": 0.60}

        return {"fillColor": FLAT_COLOR, "color": border, "weight": 1, "fillOpacity": 0.55}

    tooltip_fields = []
    tooltip_aliases = []
    # show basic fields if exist
    for f, a in [
        ("gakku_join", "District"),
        ("pop_trend", "Pop trend"),
        ("pop_slope", "Pop slope"),
        ("students_trend", "Students trend"),
        ("students_slope", "Students slope"),
        ("bear_risk_raw", "Bear raw"),
        ("combined_risk", "Combined risk"),
        ("pop_total", "Pop total"),
    ]:
        if f in gdf.columns:
            tooltip_fields.append(f)
            tooltip_aliases.append(a)

    tooltip = folium.GeoJsonTooltip(fields=tooltip_fields, aliases=tooltip_aliases, sticky=True)

    if show_district:
        folium.GeoJson(
            gdf,
            name="districts",
            style_function=style_func,
            tooltip=tooltip,
        ).add_to(m)

    # ------------------------------------------------------
    # School points: squares, color by (students/classes) risk using latest year in csv
    # ------------------------------------------------------
    if show_school:
        # Use school point shapefile if available; otherwise fallback to no points
        # We color by latest-year enroll_total/classes from school.csv by school_name
        if not school_pt.empty:
            # latest year table per school
            latest_year_school = int(df_school["fiscal year"].max()) if df_school["fiscal year"].notna().any() else None
            school_latest = df_school[df_school["fiscal year"] == latest_year_school].copy() if latest_year_school else df_school.copy()

            # join key candidates
            name_col = None
            for c in ["school_name","name","NAME","学校名","school_nam"]:
                if c in school_pt.columns:
                    name_col = c
                    break

            for _, row in school_pt.iterrows():
                lat, lon = float(row.geometry.y), float(row.geometry.x)
                sname = str(row.get(name_col, "School")) if name_col else "School"
                gakku = row.get("gakku", row.get("district", ""))

                # find latest metrics from csv
                rec = school_latest[school_latest["school_name"] == sname]
                if rec.empty:
                    # fallback: no data -> green
                    color = RISK_MID
                    s_r, c_r = "Watch", "Watch"
                    enroll = None
                    classes = None
                else:
                    enroll = float(rec["enroll_total"].iloc[0]) if "enroll_total" in rec.columns else None
                    classes = float(rec["classes"].iloc[0]) if "classes" in rec.columns else None
                    s_r = student_risk(enroll) if enroll is not None else "Watch"
                    c_r = class_risk(classes) if classes is not None else "Watch"
                    color = school_color(s_r, c_r)

                folium.RegularPolygonMarker(
                    location=[lat, lon],
                    number_of_sides=4,  # square
                    radius=6,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.85,
                    popup=folium.Popup(
                        f"<b>{sname}</b><br>District: {gakku}<br>"
                        f"Student risk: {s_r}<br>Class risk: {c_r}"
                        f"<br>Enroll: {enroll if enroll is not None else '-'}"
                        f"<br>Classes: {classes if classes is not None else '-'}",
                        max_width=300
                    )
                ).add_to(m)
        else:
            st.caption("schoolPT.shp not loaded (or empty).")

    # ------------------------------------------------------
    # Bear points: circles, small
    # ------------------------------------------------------
    if show_bear and not bear_pt.empty:
        # try to find year column
        year_col = None
        for c in ["year","Year","年度"]:
            if c in bear_pt.columns:
                year_col = c
                break

        for _, row in bear_pt.iterrows():
            lat, lon = float(row.geometry.y), float(row.geometry.x)
            y = row.get(year_col, None) if year_col else None
            try:
                y = int(y)
            except:
                y = None
            color = BEAR_COLOR_MAP.get(y, "black")
            gakku = row.get("gakku", row.get("district",""))
            memo = row.get("memo1", row.get("memo",""))

            folium.CircleMarker(
                location=[lat, lon],
                radius=3,  # smaller
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                popup=folium.Popup(
                    f"<b>Bear</b><br>Year: {y}<br>District: {gakku}<br>{memo}",
                    max_width=280
                )
            ).add_to(m)

    # ------------------------------------------------------
    # Land price points: triangles (CSV needs lat/lon)
    # ------------------------------------------------------
    if show_land and df_land is not None and not df_land.empty:
        lat_col = None
        lon_col = None
        for c in ["lat","latitude","緯度","LAT"]:
            if c in df_land.columns:
                lat_col = c
                break
        for c in ["lon","lng","longitude","経度","LON"]:
            if c in df_land.columns:
                lon_col = c
                break

        id_candidates = [c for c in ["ID","id","point_id","name","地点","地点名","標準地番号"] if c in df_land.columns]
        id_col = id_candidates[0] if id_candidates else df_land.columns[0]

        if lat_col and lon_col:
            pts = df_land.dropna(subset=[lat_col, lon_col]).copy()
            for _, r in pts.iterrows():
                lat, lon = float(r[lat_col]), float(r[lon_col])
                pid = r.get(id_col, "")
                folium.RegularPolygonMarker(
                    location=[lat, lon],
                    number_of_sides=3,  # triangle
                    radius=6,
                    color="#8e6b3a",     # brown-ish
                    fill=True,
                    fill_color="#8e6b3a",
                    fill_opacity=0.85,
                    popup=folium.Popup(f"<b>Land price</b><br>ID: {pid}", max_width=200)
                ).add_to(m)
        else:
            st.caption("chika2.csv has no lat/lon columns (lat/lon). Triangles not shown on map.")

    folium.LayerControl(collapsed=True).add_to(m)

    map_data = st_folium(m, width=None, height=640)

    st.caption("Tip: Use left panel for charts; map is for spatial context & click-to-see basic attributes.")
