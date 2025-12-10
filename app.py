import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from statsmodels.tsa.statespace.structural import UnobservedComponents
import altair as alt
from shapely.geometry import Point

# ------------------------------------------------------
# ページ設定
# ------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="District Population & Students Viewer"
)

# ------------------------------------------------------
# カラムごとに別スクロールにするための CSS
# ------------------------------------------------------
st.markdown(
    """
    <style>
    /* 最初の columns の左カラム */
    div[data-testid="column"]:nth-of-type(1) > div {
        max-height: 900px;
        overflow-y: auto;
    }
    /* 最初の columns の右カラム */
    div[data-testid="column"]:nth-of-type(2) > div {
        max-height: 900px;
        overflow-y: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------
# 学校リスク判定のヘルパー
# ------------------------------------------------------
def categorize_school_risk(students_2030: float) -> str:
    """
    2030年の予測児童数から簡易にリスク判定する。
    - < 100人 : 高リスク（統廃合検討レベル）
    - 100〜199人 : 要注意
    - >= 200人 : 安定
    """
    if students_2030 is None:
        return "データ不足"
    try:
        v = float(students_2030)
    except Exception:
        return "データ不足"

    if v < 100:
        return "高リスク（統廃合検討レベル）"
    elif v < 200:
        return "要注意"
    else:
        return "安定"

# ------------------------------------------------------
# データ読み込み
# ------------------------------------------------------
@st.cache_data
def load_data():
    # 学区ポリゴン
    gdf = gpd.read_file("gakku_trend.geojson")
    try:
        gdf = gdf.to_crs(epsg=4326)
    except Exception:
        pass

    # 人口データ
    df_pop = pd.read_csv("jinko2018_2025.csv", encoding="shift_jis")
    df_pop.columns = [c.strip() for c in df_pop.columns]

    if "pop_0_11" not in df_pop.columns:
        age_cols = [f"old_{i}" for i in range(12) if f"old_{i}" in df_pop.columns]
        df_pop["pop_0_11"] = df_pop[age_cols].sum(axis=1)

    df_pop["ID"] = df_pop["ID"].astype(str)

    # 生徒数データ
    df_school = pd.read_csv("school.csv", encoding="utf-8-sig")
    df_school.columns = [c.strip() for c in df_school.columns]
    df_school = df_school.rename(columns={"enroll_total": "students"})
    df_school["ID"] = df_school["ID"].astype(str)
    df_school["fiscal year"] = df_school["fiscal year"].astype(int)

    # 学校ポイント
    school_pt = gpd.read_file("schoolPT.shp")
    school_pt = school_pt.to_crs(epsg=4326)

    # 公示価格（モデル・属性用に学区平均だけ作る）
    try:
        df_chika = pd.read_csv("chika2.csv", encoding="utf-8-sig")
        df_chika.columns = [c.strip() for c in df_chika.columns]
        df_chika["ID"] = df_chika["ID"].astype(str)

        year_cols = ["2023", "2024", "2025"]
        for y in year_cols:
            df_chika[y] = pd.to_numeric(df_chika[y], errors="coerce")

        chika_group = (
            df_chika.groupby("ID")[year_cols]
            .mean()
            .rename(columns={
                "2023": "landprice_2023",
                "2024": "landprice_2024",
                "2025": "landprice_2025",
            })
            .reset_index()
        )

        gdf["ID"] = gdf["ID"].astype(str)
        gdf = gdf.merge(chika_group, on="ID", how="left")
    except Exception:
        # 地価が無くてもアプリが落ちないように
        pass

    # 熊ポイント
    try:
        bear_pt = gpd.read_file("bearPT.shp")
        bear_pt = bear_pt.to_crs(epsg=4326)
        # year を int に
        bear_pt["year"] = bear_pt["year"].astype(int)
        bear_pt["ID"] = bear_pt["ID"].astype(str)
    except Exception:
        bear_pt = gpd.GeoDataFrame(columns=["memo1", "year", "area", "ID", "gakku", "geometry"], geometry="geometry")

    return gdf, df_pop, df_school, school_pt, bear_pt


gdf, df_pop, df_school, school_pt, bear_pt = load_data()

# 学校名の列名（shape によって違う可能性があるので自動判定）
if "school_name" in school_pt.columns:
    SCHOOL_NAME_COL = "school_name"
elif "school_nam" in school_pt.columns:
    SCHOOL_NAME_COL = "school_nam"
else:
    SCHOOL_NAME_COL = school_pt.columns[0]

# ------------------------------------------------------
# サイドバー：レイヤー表示切り替え
# ------------------------------------------------------
st.sidebar.header("Layer visibility")
show_district = st.sidebar.checkbox("人口・学区ポリゴン", value=True)
show_school   = st.sidebar.checkbox("学校ポイント", value=True)
show_bear     = st.sidebar.checkbox("熊出没ポイント", value=True)
# 今回は地価はモデル側で利用が主なので、地図レイヤーとしては後回し

# ------------------------------------------------------
# 地図の作成（Folium）
# ------------------------------------------------------
centroid = gdf.geometry.centroid
center = [centroid.y.mean(), centroid.x.mean()]

# タイルは自前で追加するので tiles=None
m = folium.Map(location=center, zoom_start=11, tiles=None)

# 航空写真タイル（デフォルト）
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery",
    name="Aerial",
    overlay=False,
    control=True,
).add_to(m)

# 通常のOSMも選べるように
folium.TileLayer(
    tiles="OpenStreetMap",
    name="OSM",
    overlay=False,
    control=True,
).add_to(m)

color_map = {"Up": "#2ecc71", "Down": "#e74c3c", "Flat": "#f1c40f"}


def style_func(feature):
    trend = feature["properties"].get("trend_class", "Flat")
    return {
        "fillColor": color_map.get(trend, "#cccccc"),
        "color": "black",
        "fillOpacity": 0.6,
        "weight": 1,
    }


tooltip = folium.GeoJsonTooltip(
    fields=["school_nam", "district", "trend_class", "pop_2025", "pop_2030"],
    aliases=["District (JP)", "District (EN)", "Trend", "Pop 2025", "Pop 2030"],
)

# 学区ポリゴン
if show_district:
    folium.GeoJson(
        gdf,
        name="districts",
        style_function=style_func,
        tooltip=tooltip,
    ).add_to(m)

# 学校ポイント：ピクトグラム風（緑の学校マーク）
if show_school:
    for _, row in school_pt.iterrows():
        name = row[SCHOOL_NAME_COL]
        gakku = row.get("gakku", "")
        sid = row.get("ID", "")

        # 「附属（other）」など区別したい場合はここで条件分岐
        icon = folium.Icon(
            icon="graduation-cap",  # 学校のピクトグラム風
            prefix="fa",
            color="green",          # ピンの色
            icon_color="white",     # アイコン色
        )

        folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            icon=icon,
            popup=folium.Popup(
                f"<b>{name}</b><br>ID: {sid}<br>Gakku: {gakku}",
                max_width=250,
            ),
        ).add_to(m)

# 熊ポイント：年ごとに色を変える
if show_bear and not bear_pt.empty:
    bear_color_map = {
        2023: "orange",
        2024: "red",
        2025: "purple",
    }
    for _, row in bear_pt.iterrows():
        year = int(row["year"])
        color = bear_color_map.get(year, "black")
        memo = row.get("memo1", "")
        gakku = row.get("gakku", "")
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=folium.Popup(
                f"<b>Year:</b> {year}<br><b>Gakku:</b> {gakku}<br><b>Memo:</b> {memo}",
                max_width=300,
            ),
        ).add_to(m)

# レイヤーコントロール
folium.LayerControl().add_to(m)

# ------------------------------------------------------
# レイアウト：左＝属性＆グラフ／右＝地図＋ランキング
# ------------------------------------------------------
st.title("District Population & Students Viewer")

col_left, col_right = st.columns([1, 1])

# 右カラム：地図＋District Ranking
with col_right:
    st.subheader("Map")
    map_data = st_folium(
        m,
        width=800,
        height=500,
        returned_objects=["last_clicked"],
    )

    # ---------------- 学区ランキング（地図の下） ----------------
    st.markdown("---")
    st.subheader("District Ranking (0–11 Population Change 2025→2030)")

    rank_cols = [
        "school_nam",       # JP name
        "district",         # EN name
        "trend_class",
        "pop_2025",
        "pop_2030",
        "change_2025_2030",
    ]
    df_rank = gdf[rank_cols].copy()
    df_rank = df_rank.rename(
        columns={
            "school_nam": "District_JP",
            "district": "District_EN",
            "trend_class": "Trend",
            "pop_2025": "Pop_2025",
            "pop_2030": "Pop_2030",
            "change_2025_2030": "Change_25_30",
        }
    )

    df_rank["Pop_2030"] = df_rank["Pop_2030"].round(1)
    df_rank["Change_25_30"] = df_rank["Change_25_30"].round(1)

    df_top = df_rank.sort_values("Change_25_30", ascending=False).head(10)
    df_bottom = df_rank.sort_values("Change_25_30", ascending=True).head(10)

    col_top, col_bottom = st.columns(2)

    with col_top:
        st.markdown("**📈 Top 10 Increasing Districts**")
        st.dataframe(
            df_top[
                [
                    "District_JP",
                    "District_EN",
                    "Trend",
                    "Pop_2025",
                    "Pop_2030",
                    "Change_25_30",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    with col_bottom:
        st.markdown("**📉 Top 10 Decreasing Districts**")
        st.dataframe(
            df_bottom[
                [
                    "District_JP",
                    "District_EN",
                    "Trend",
                    "Pop_2025",
                    "Pop_2030",
                    "Change_25_30",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

# ------------------------------------------------------
# クリック位置から「学区」と「学校」を判定
# ------------------------------------------------------
selected_district_id = None
selected_school_id = None

clicked = map_data.get("last_clicked") if map_data else None

if clicked and ("lat" in clicked and "lng" in clicked):
    pt = Point(clicked["lng"], clicked["lat"])

    # ① 最寄りの学校ポイント（約300m以内なら学校クリックとみなす）
    if show_school and not school_pt.empty:
        dists = school_pt.geometry.distance(pt)
        idx_min = dists.idxmin()
        min_dist = dists.loc[idx_min]

        if min_dist < 0.003:  # ~ 300m
            selected_school_id = str(school_pt.loc[idx_min, "ID"])

    # ② ポリゴンに含まれていれば学区
    if show_district:
        candidates = gdf[gdf.geometry.contains(pt)]
        if len(candidates) > 0:
            selected_district_id = str(candidates.iloc[0]["ID"])

# ------------------------------------------------------
# 左カラム：属性・グラフ類
# ------------------------------------------------------
with col_left:
    # ---------------- 学区情報 ----------------
    st.subheader("Selected District Info")

    if selected_district_id is not None:
        row = gdf[gdf["ID"].astype(str) == selected_district_id]
        if len(row) > 0:
            row = row.iloc[0]
            st.write(f"**District (JP)**: {row['school_nam']}")
            st.write(f"**District (EN)**: {row['district']}")
            st.write(f"**Trend**: {row['trend_class']}")
            st.write(f"**Pop 0–11 (2025)**: {row['pop_2025']}")
            st.write(f"**Pop 0–11 (2030)**: {row['pop_2030']:.1f}")
            st.write(f"**Change 2025→2030**: {row['change_2025_2030']:.1f}")
            # 地価が入っていれば表示
            if "landprice_2025" in row.index:
                lp = row["landprice_2025"]
                if pd.notnull(lp):
                    st.write(f"**Land Price 2025 (avg)**: {lp:.0f} 円/㎡")
        else:
            st.warning("選択された ID に対応する学区が見つかりませんでした。")
    else:
        st.info("地図の学区（ポリゴン）をクリックすると情報が表示されます。")

    # ---------------- 学区の時系列 ----------------
    st.subheader("District Population & Students Time Series")

    if selected_district_id is not None:
        row_dist = gdf[gdf["ID"].astype(str) == selected_district_id]

        if len(row_dist) > 0:
            name_en = row_dist.iloc[0]["district"]

            df_pop_d = df_pop[df_pop["gakku"] == name_en].sort_values("fiscal year")
            df_sch_d = df_school[
                (df_school["gakku"] == name_en) & (df_school["gakku"] != "other")
            ].sort_values("fiscal year")

            if len(df_pop_d) == 0 and len(df_sch_d) == 0:
                st.warning("この学区の人口・生徒数データが見つかりませんでした。")
            else:
                rec = []

                # 人口
                if len(df_pop_d) > 0:
                    target_pop = df_pop_d.set_index("fiscal year")["pop_0_11"]

                    model_pop = UnobservedComponents(
                        target_pop, level="local linear trend"
                    )
                    result_pop = model_pop.fit(disp=False)

                    steps = 5
                    pred_pop = result_pop.get_forecast(steps=steps)
                    pred_pop_mean = pred_pop.predicted_mean
                    last_year_pop = int(target_pop.index.max())
                    future_years_pop = [
                        last_year_pop + i for i in range(1, steps + 1)
                    ]

                    for y, v in target_pop.items():
                        rec.append(
                            {"year": int(y), "Series": "Population Observed", "value": v}
                        )
                    for y, v in zip(future_years_pop, pred_pop_mean.values):
                        rec.append(
                            {"year": int(y), "Series": "Population Forecast", "value": v}
                        )

                # 生徒（学区単位）
                if len(df_sch_d) > 0:
                    target_std = df_sch_d.set_index("fiscal year")["students"]

                    if len(target_std) >= 2:
                        model_std = UnobservedComponents(
                            target_std, level="local linear trend"
                        )
                        result_std = model_std.fit(disp=False)

                        steps = 5
                        pred_std = result_std.get_forecast(steps=steps)
                        pred_std_mean = pred_std.predicted_mean
                        last_year_std = int(target_std.index.max())
                        future_years_std = [
                            last_year_std + i for i in range(1, steps + 1)
                        ]

                        for y, v in target_std.items():
                            rec.append(
                                {"year": int(y), "Series": "Students Observed", "value": v}
                            )
                        for y, v in zip(future_years_std, pred_std_mean.values):
                            rec.append(
                                {"year": int(y), "Series": "Students Forecast", "value": v}
                            )
                    else:
                        for y, v in target_std.items():
                            rec.append(
                                {"year": int(y), "Series": "Students Observed", "value": v}
                            )

                if rec:
                    df_long = pd.DataFrame(rec)
                    chart = (
                        alt.Chart(df_long)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("year:Q", axis=alt.Axis(title="Year", format=".0f")),
                            y=alt.Y(
                                "value:Q",
                                title="Population / Students",
                            ),
                            color=alt.Color("Series:N", title=""),
                        )
                        .properties(height=220)
                    )
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.warning("表示できる時系列データがありません。")
        else:
            st.warning("選択された ID に対応する学区が見つかりませんでした。")
    else:
        st.info("上のグラフには、選択した学区の人口と生徒数の時系列が表示されます。")

    # ---------------- 学校情報 ----------------
    st.markdown("---")
    st.subheader("Selected School Info")

    if selected_school_id is not None:
        row_pt = school_pt[school_pt["ID"].astype(str) == selected_school_id]
        row_sch = df_school[df_school["ID"].astype(str) == selected_school_id]

        if len(row_pt) > 0 and len(row_sch) > 0:
            row_pt = row_pt.iloc[0]
            name = row_pt[SCHOOL_NAME_COL]
            gakku = row_pt.get("gakku", "N/A")

            df_s = row_sch.sort_values("fiscal year")
            target_std = df_s.set_index("fiscal year")["students"]

            # 生徒数予測（学校単位）
            students_2030 = None
            if len(target_std) >= 2:
                model_s = UnobservedComponents(
                    target_std, level="local linear trend"
                )
                result_s = model_s.fit(disp=False)
                pred_s = result_s.get_forecast(steps=5)
                students_2030 = pred_s.predicted_mean.iloc[-1]

            students_2025 = (
                target_std.loc[2025] if 2025 in target_std.index else None
            )
            change = (
                students_2030 - students_2025
                if (students_2030 is not None and students_2025 is not None)
                else None
            )
            change_rate = (
                (students_2030 / students_2025 - 1) * 100
                if (students_2030 is not None and students_2025 not in (None, 0))
                else None
            )

            # 統廃合リスク判定
            risk_label = categorize_school_risk(students_2030)

            st.write(f"**School**: {name}")
            st.write(f"**Gakku**: {gakku}")
            st.write(f"**School ID**: {selected_school_id}")

            if students_2025 is not None:
                st.write(f"**Students 2025**: {students_2025:.0f}")
            if students_2030 is not None:
                st.write(f"**Students 2030 (forecast)**: {students_2030:.1f}")
            if change is not None:
                st.write(f"**Change 2025→2030**: {change:.1f}")
            if change_rate is not None:
                st.write(f"**Change rate**: {change_rate:.1f}%")

            st.write(f"**統廃合リスク判定**: {risk_label}")
            st.caption(
                "※ 簡易ルール：2030年の予測児童数が 200人未満で『要注意』、"
                "100人未満で『高リスク（統廃合検討レベル）』としています（暫定基準）。"
            )

            # 教員数・学級数（2025）
            df_s_idx = df_s.set_index("fiscal year")
            teachers_2025 = (
                df_s_idx["teachers"].loc[2025]
                if "teachers" in df_s_idx.columns and 2025 in df_s_idx.index
                else None
            )
            classes_2025 = (
                df_s_idx["classes"].loc[2025]
                if "classes" in df_s_idx.columns and 2025 in df_s_idx.index
                else None
            )

            if teachers_2025 is not None:
                st.write(f"**Teachers 2025**: {teachers_2025:.0f}")
            if classes_2025 is not None:
                st.write(f"**Classes 2025**: {classes_2025:.0f}")
        else:
            st.warning("選択された学校の情報が見つかりませんでした。")
    else:
        st.info("学校ポイントをクリックすると、ここに情報と指標が表示されます。")

    # ---------------- 学校タイムシリーズ（上下2段） ----------------
    st.subheader("School Time Series (Students / Teachers / Classes)")

    if selected_school_id is not None:
        row_sch = df_school[df_school["ID"].astype(str) == selected_school_id]

        if len(row_sch) > 0:
            df_s = row_sch.sort_values("fiscal year").set_index("fiscal year")

            # 生徒（観測＋予測）
            rec_students = []
            for y, v in df_s["students"].items():
                rec_students.append(
                    {"year": int(y), "Series": "Students", "value": v}
                )

            target_std = df_s["students"]
            if len(target_std) >= 2:
                model_s = UnobservedComponents(
                    target_std, level="local linear trend"
                )
                result_s = model_s.fit(disp=False)
                steps = 5
                pred_s = result_s.get_forecast(steps=steps)
                pred_s_mean = pred_s.predicted_mean
                last_year = int(target_std.index.max())
                future_years = [last_year + i for i in range(1, steps + 1)]

                for y, v in zip(future_years, pred_s_mean.values):
                    rec_students.append(
                        {
                            "year": int(y),
                            "Series": "Students Forecast",
                            "value": v,
                        }
                    )

            df_students_long = pd.DataFrame(rec_students)

            # 教員・学級
            rec_staff = []
            if "teachers" in df_s.columns:
                for y, v in df_s["teachers"].items():
                    rec_staff.append(
                        {"year": int(y), "Series": "Teachers", "value": v}
                    )
            if "classes" in df_s.columns:
                for y, v in df_s["classes"].items():
                    rec_staff.append(
                        {"year": int(y), "Series": "Classes", "value": v}
                    )

            df_staff_long = pd.DataFrame(rec_staff)

            charts = []

            # 上：生徒
            if not df_students_long.empty:
                students_chart = (
                    alt.Chart(df_students_long)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X(
                            "year:Q",
                            axis=alt.Axis(title="Year", format=".0f"),
                        ),
                        y=alt.Y(
                            "value:Q",
                            title="Students",
                        ),
                        color=alt.Color("Series:N", title=""),
                    )
                    .properties(height=160)
                )
                charts.append(students_chart)

            # 下：教員＋学級
            if not df_staff_long.empty:
                staff_chart = (
                    alt.Chart(df_staff_long)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X(
                            "year:Q",
                            axis=alt.Axis(title="Year", format=".0f"),
                        ),
                        y=alt.Y(
                            "value:Q",
                            title="Teachers / Classes",
                        ),
                        color=alt.Color("Series:N", title=""),
                    )
                    .properties(height=160)
                )
                charts.append(staff_chart)

            if charts:
                combined = alt.vconcat(*charts).resolve_scale(y="independent")
                st.altair_chart(combined, use_container_width=True)
            else:
                st.warning("表示できる学校時系列データがありません。")
        else:
            st.warning("選択された学校のデータが見つかりませんでした。")
    else:
        st.info("学校ポイントをクリックすると、下のグラフに時系列が表示されます。")
