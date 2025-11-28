import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Norway Macro Dashboard", layout="wide")
st.title("Norway Macro Dashboard")
st.caption("Live data from Statistics Norway (SSB) and Norges Bank APIs")

def ssb_to_timestamp(x: str) -> pd.Timestamp:
    y, m = x.split("M")
    return pd.to_datetime(f"{y}-{m}-01")

def set_default_window(fig, dates, years=1):
    """
    Keep full history in the DF, but default the chart view to the last `years` years.
    Also forces y-axes to autorange based on the visible x-window.
    """
    dates = pd.to_datetime(pd.Series(dates)).dropna()
    if dates.empty:
        return fig

    end = dates.max()
    start = end - pd.DateOffset(years=years)

    fig.update_xaxes(range=[start, end], type="date")
    fig.update_yaxes(autorange=True)
    return fig

def add_time_toggles(fig, dates, default_years=5):
    dates = pd.to_datetime(pd.Series(dates)).dropna()
    if dates.empty:
        return fig

    last_date = dates.max()
    first_date = dates.min()

    start = max(first_date, last_date - pd.DateOffset(years=default_years))

    fig.update_layout(
        xaxis=dict(
            range=[start, last_date],
            rangeselector=dict(buttons=RANGE_BUTTONS),
            rangeslider=dict(visible=False),
            type="date",
        )
    )
    fig.update_yaxes(autorange=True)
    return fig


# Make the NIIP / external settlement time axes “real dates” instead of "2024K1"
def quarter_str_to_timestamp(q: str) -> pd.Timestamp:
    """
    Convert an SSB quarter key like '2024K1' into a pandas Timestamp
    representing the end of that quarter.

    Mapping:
      'YYYYK1' -> YYYY-03-31
      'YYYYK2' -> YYYY-06-30
      'YYYYK3' -> YYYY-09-30
      'YYYYK4' -> YYYY-12-31
    """
    year_str, q_str = q.split("K")
    year = int(year_str)
    q_num = int(q_str)

    month_map = {1: 3, 2: 6, 3: 9, 4: 12}
    day_map   = {3: 31, 6: 30, 9: 30, 12: 31}

    month = month_map[q_num]
    day   = day_map[month]

    return pd.Timestamp(year=year, month=month, day=day)




#Color Contstraints
NB_DARK_BLUE   = "#003B5C"
NB_MED_BLUE    = "#4F87B2"
NB_LIGHT_BLUE  = "#A7C6E5"
NB_GREEN       = "#2A8C55"
NB_RED         = "#B03A2E"
NB_GREY        = "#7F7F7F"
NB_LIGHT_GREY  = "#E5E5E5"
ACC_ORANGE     = "#E08E0B"
ACC_TEAL       = "#008C95"
ACC_PURPLE     = "#5A4E8C"

#Range Buttons
RANGE_BUTTONS = [
    dict(count=3,  label="3M",  step="month", stepmode="backward"),
    dict(count=12, label="1Y",  step="month", stepmode="backward"),
    dict(count=5,  label="5Y",  step="year",  stepmode="backward"),
    dict(step="all", label="All"),
]


@st.cache_data(ttl=3600, show_spinner=True)
def dashboard_1():
    # ========================================================
    # 1) CPI: headline + core (YoY)
    # ========================================================
    url_core = (
        "https://data.ssb.no/api/pxwebapi/v2/tables/05327/data?lang=en"
        "&valueCodes[Konsumgrp]=JAE_TOTAL"
        "&valueCodes[ContentsCode]=Tolvmanedersendring"
        "&valueCodes[Tid]=top(12)"
        "&outputformat=json-px"
    )
    resp_core = requests.get(url_core)
    resp_core.raise_for_status()
    j_core = resp_core.json()

    core_rows = []
    for row in j_core["data"]:
        tid = row["key"][-1]   # ["JAE_TOTAL", "2025M01"] -> "2025M01"
        raw_val = row["values"][0]
        try:
            value = float(raw_val)
        except ValueError:
            continue
        core_rows.append({"Tid": tid, "core_yoy": value})

    df_core = pd.DataFrame(core_rows)
    df_core["date"] = df_core["Tid"].apply(ssb_to_timestamp)
    df_core = df_core[["date", "core_yoy"]].sort_values("date")

    url_headline = (
        "https://data.ssb.no/api/pxwebapi/v2/tables/03013/data?lang=en"
        "&valueCodes[Konsumgrp]=TOTAL"
        "&valueCodes[ContentsCode]=Tolvmanedersendring"
        "&valueCodes[Tid]=top(12)"
        "&outputformat=json-px"
    )
    resp_headline = requests.get(url_headline)
    resp_headline.raise_for_status()
    j_headline = resp_headline.json()

    headline_rows = []
    for row in j_headline["data"]:
        tid = row["key"][-1]
        raw_val = row["values"][0]
        try:
            value = float(raw_val)
        except ValueError:
            continue
        headline_rows.append({"Tid": tid, "headline_yoy": value})

    df_headline = pd.DataFrame(headline_rows)
    df_headline["date"] = df_headline["Tid"].apply(ssb_to_timestamp)
    df_headline = df_headline[["date", "headline_yoy"]].sort_values("date")

    df_cpi = (
        pd.merge(df_headline, df_core, on="date", how="inner")
        .sort_values("date")
    )

    # ========================================================
    # 2) Labour market: unemployment (SA, 15–74)
    # ========================================================
    url_u = (
        "https://data.ssb.no/api/pxwebapi/v2/tables/13760/data?lang=en"
        "&valueCodes[ContentsCode]=ArbledProsArbstyrk"
        "&valueCodes[Kjonn]=0"
        "&valueCodes[Alder]=15-74"
        "&valueCodes[Justering]=S"
        "&valueCodes[Tid]=top(12)"
        "&outputformat=json-px"
    )
    resp_u = requests.get(url_u)
    resp_u.raise_for_status()
    j_u = resp_u.json()

    cols_u = [c["code"] for c in j_u["columns"]]
    rows_u = []
    for row in j_u["data"]:
        key_map = dict(zip(cols_u, row["key"]))
        tid = key_map["Tid"]
        raw_val = row["values"][0]
        try:
            value = float(raw_val)
        except ValueError:
            continue
        rows_u.append({"Tid": tid, "unemployment_sa": value})

    df_u = pd.DataFrame(rows_u)
    df_u["date"] = df_u["Tid"].apply(ssb_to_timestamp)
    df_u = df_u[["date", "unemployment_sa"]].sort_values("date")

    # ========================================================
    # 3) Monetary aggregates: M1, M2, M3
    # ========================================================
    url_m = (
        "https://data.ssb.no/api/pxwebapi/v2/tables/10945/data?lang=en"
        "&valueCodes[ContentsCode]=PengmengdBehM1,PengmengdBehM2,PengmengdBehM3"
        "&valueCodes[Tid]=top(120)"
        "&outputformat=json-px"
    )
    resp_m = requests.get(url_m)
    resp_m.raise_for_status()
    j_m = resp_m.json()

    rows_m = []
    for row in j_m["data"]:
        tid = row["key"][0]
        values = row["values"]
        rows_m.append({
            "Tid": tid,
            "M1": float(values[0]),
            "M2": float(values[1]),
            "M3": float(values[2]),
        })

    df_m = pd.DataFrame(rows_m)
    df_m["date"] = df_m["Tid"].apply(ssb_to_timestamp)
    df_m = df_m[["date", "M1", "M2", "M3"]].sort_values("date")

    # ========================================================
    # 4) External settlements: incoming / outgoing / net
    # ========================================================
    url_ext = (
        "https://data.ssb.no/api/pxwebapi/v2/tables/08785/data?lang=en"
        "&valueCodes[Inngaaende2]=01,02"
        "&valueCodes[NACE2007]=A,B,C,D,E,F,G,H,I,J,L,M,N,O,P,Q,R,S,U,Z"
        "&valueCodes[ContentsCode]=Betaling"
        "&valueCodes[Tid]=top(40)"
        "&outputformat=json-px"
    )
    resp_ext = requests.get(url_ext)
    resp_ext.raise_for_status()
    j_ext = resp_ext.json()

    cols_ext = [c["code"] for c in j_ext["columns"]]
    records_ext = []
    for row in j_ext["data"]:
        key_map = dict(zip(cols_ext, row["key"]))
        raw_val = row["values"][0]
        try:
            value = float(raw_val)
        except ValueError:
            continue
        records_ext.append({
            "industry": key_map["NACE2007"],
            "period": key_map["Tid"],
            "flow": key_map["Inngaaende2"],
            "value": value,
        })

    df_ext = pd.DataFrame(records_ext)
    df_in = df_ext[df_ext["flow"] == "01"].pivot(index="industry", columns="period", values="value")
    df_out = df_ext[df_ext["flow"] == "02"].pivot(index="industry", columns="period", values="value")
    df_in = df_in.add_prefix("incoming_")
    df_out = df_out.add_prefix("outgoing_")
    df_ext_final = pd.concat([df_in, df_out], axis=1).reset_index()

    industry_map = {
        "A": "A Agriculture, forestry and fishing",
        "B": "B Mining and quarrying",
        "C": "C Manufacturing",
        "D": "D Electricity, gas and steam",
        "E": "E Water supply, sewerage, waste",
        "F": "F Construction",
        "G": "G Wholesale and retail trade: repair of motor vehicles and motorcycles",
        "H": "H Transportation and storage",
        "I": "I Accommodation and food service activities",
        "J": "J Information and communication",
        "L": "L Real estate activities",
        "M": "M Professional, scientific and technical activities",
        "N": "N Administrative and support service activities",
        "O": "O Public administration and defence",
        "P": "P Education",
        "Q": "Q Human health and social work activities",
        "R": "R Arts, entertainment and recreation",
        "S": "S Other service activities",
        "U": "U Activities of extraterritorial organisations and bodies",
        "Z": "Z Unknown industry",
    }

    df_ext_final["industry"] = df_ext_final["industry"].map(industry_map)
    df_ext_final = df_ext_final.rename(columns={"industry": "Industry"})
    df_ext_final = df_ext_final[["Industry"] + [c for c in df_ext_final.columns if c != "Industry"]]

    incoming_cols = [c for c in df_ext_final.columns if c.startswith("incoming_")]
    df_in_long = df_ext_final.melt(
        id_vars="Industry",
        value_vars=incoming_cols,
        var_name="Quarter",
        value_name="Value",
    )
    df_in_long["Quarter"] = df_in_long["Quarter"].str.replace("incoming_", "", regex=False)

    outgoing_cols = [c for c in df_ext_final.columns if c.startswith("outgoing_")]
    df_out_long = df_ext_final.melt(
        id_vars="Industry",
        value_vars=outgoing_cols,
        var_name="Quarter",
        value_name="Value",
    )
    df_out_long["Quarter"] = df_out_long["Quarter"].str.replace("outgoing_", "", regex=False)

    in_sum = df_in_long.groupby("Quarter")["Value"].sum()
    out_sum = df_out_long.groupby("Quarter")["Value"].sum()

    out_aligned = out_sum.reindex(in_sum.index).fillna(0.0)

    df_net = pd.DataFrame({
        "Quarter": in_sum.index,
        "incoming_total": in_sum.values,
        "outgoing_total": out_aligned.values,
    })
    df_net["net"] = df_net["incoming_total"] - df_net["outgoing_total"]
    df_net = df_net.sort_values("Quarter")

    # Convert quarter strings to actual dates for plotting
    df_in_long["Quarter_date"]  = df_in_long["Quarter"].apply(quarter_str_to_timestamp)
    df_out_long["Quarter_date"] = df_out_long["Quarter"].apply(quarter_str_to_timestamp)
    df_net["Quarter_date"]      = df_net["Quarter"].apply(quarter_str_to_timestamp)


    # ========================================================
    # 5) NIIP STOCKS
    # ========================================================
    url_niip_stock = (
        "https://data.ssb.no/api/pxwebapi/v2/tables/10644/data?lang=en"
        "&valueCodes[FordringGjeld]=F,G"
        "&valueCodes[FunksjObjSektor]=3.1,3.2,3.3,3.4,3.5"
        "&valueCodes[ContentsCode]=Beholdning"
        "&valueCodes[Tid]=top(40)"
        "&outputformat=json-px"
    )
    resp_ns = requests.get(url_niip_stock)
    resp_ns.raise_for_status()
    j_ns = resp_ns.json()

    cols_ns = [c["code"] for c in j_ns["columns"]]
    records_ns = []
    for row in j_ns["data"]:
        key_map = dict(zip(cols_ns, row["key"]))
        side = key_map["FordringGjeld"]
        func = key_map["FunksjObjSektor"]
        quarter = key_map["Tid"]

        raw_val = row["values"][0]
        if raw_val in (".", ".."):
            value = 0.0
        else:
            value = float(raw_val)

        records_ns.append({
            "quarter": quarter,
            "function_code": func,
            "side": side,
            "value": value,
        })

    df_ns = pd.DataFrame(records_ns)
    pivot_ns = df_ns.pivot_table(
        index=["quarter", "function_code"],
        columns="side",
        values="value",
        aggfunc="first"
    ).reset_index()

    pivot_ns = pivot_ns.rename(columns={"F": "assets", "G": "liabilities"})
    pivot_ns[["assets", "liabilities"]] = pivot_ns[["assets", "liabilities"]].fillna(0.0)
    pivot_ns["net_position"] = pivot_ns["assets"] - pivot_ns["liabilities"]

    func_labels = {
        "3.1": "Direct investment",
        "3.2": "Portfolio investment",
        "3.3": "Financial derivatives",
        "3.4": "Other investments",
        "3.5": "Reserve assets",
    }
    pivot_ns["function"] = pivot_ns["function_code"].map(func_labels)

    df_niip_stock = pivot_ns.pivot_table(
        index="quarter",
        columns="function",
        values="net_position",
        aggfunc="first"
    ).sort_index()

    df_niip_stock["Total NIIP"] = df_niip_stock.sum(axis=1)

    # Drop quarters where all functions are zero (these are basically missing data)
    func_cols_stock = [
        "Direct investment",
        "Portfolio investment",
        "Other investments",
        "Financial derivatives",
        "Reserve assets",
    ]
    mask_nonzero = df_niip_stock[func_cols_stock].abs().sum(axis=1) != 0
    df_niip_stock = df_niip_stock[mask_nonzero]


    # Bring quarter out of the index and create a real date for plotting
    df_niip_stock = df_niip_stock.reset_index()  # 'quarter' becomes a column
    df_niip_stock["quarter_date"] = df_niip_stock["quarter"].apply(quarter_str_to_timestamp)


    # ========================================================
    # 6) NIIP FLOWS
    # ========================================================
    url_niip_flow = (
        "https://data.ssb.no/api/pxwebapi/v2/tables/10644/data?lang=en"
        "&valueCodes[FordringGjeld]=F,G"
        "&valueCodes[FunksjObjSektor]=3.1,3.2,3.3,3.4,3.5"
        "&valueCodes[ContentsCode]=Transaksj"
        "&valueCodes[Tid]=top(40)"
        "&outputformat=json-px"
    )
    resp_nf = requests.get(url_niip_flow)
    resp_nf.raise_for_status()
    j_nf = resp_nf.json()

    cols_nf = [c["code"] for c in j_nf["columns"]]
    records_nf = []
    for row in j_nf["data"]:
        key_map = dict(zip(cols_nf, row["key"]))
        side = key_map["FordringGjeld"]
        func = key_map["FunksjObjSektor"]
        quarter = key_map["Tid"]

        val_raw = row["values"][0]
        if val_raw in (".", ".."):
            value = 0.0
        else:
            value = float(val_raw)

        records_nf.append({
            "quarter": quarter,
            "function_code": func,
            "side": side,
            "value": value,
        })


    df_nf = pd.DataFrame(records_nf)
    pivot_nf = df_nf.pivot_table(
        index=["quarter", "function_code"],
        columns="side",
        values="value",
        aggfunc="first"
    ).reset_index()

    pivot_nf = pivot_nf.rename(columns={"F": "assets", "G": "liabilities"})
    pivot_nf = pivot_nf.fillna(0.0)
    pivot_nf["net"] = pivot_nf["assets"] - pivot_nf["liabilities"]
    pivot_nf["function"] = pivot_nf["function_code"].map(func_labels)

    df_niip_flow = pivot_nf.pivot_table(
        index="quarter",
        columns="function",
        values="net",
        aggfunc="first"
    ).sort_index()

    func_cols = [
        "Direct investment",
        "Portfolio investment",
        "Other investments",
        "Financial derivatives",
        "Reserve assets",
    ]

    df_niip_flow["Total net flow"] = df_niip_flow[func_cols].sum(axis=1)

    # Drop quarters where all flows are zero
    mask_nonzero_flow = df_niip_flow[func_cols].abs().sum(axis=1) != 0
    df_niip_flow = df_niip_flow[mask_nonzero_flow]


    # Reset index so 'quarter' is a column, and create a proper Timestamp for plotting
    df_niip_flow = df_niip_flow.reset_index()        # 'quarter' column (e.g. '2015K1')
    df_niip_flow["quarter_date"] = df_niip_flow["quarter"].apply(quarter_str_to_timestamp)

    # ========================================================
    # 7) BUILD SEPARATE FIGURES
    # ========================================================

    # --- CPI ---
    fig_cpi = go.Figure()
    fig_cpi.add_trace(go.Scatter(
        x=df_cpi["date"], y=df_cpi["headline_yoy"],
        mode="lines+markers", name="Headline CPI YoY",
        line=dict(color=NB_DARK_BLUE, width=2),
        marker=dict(color=NB_DARK_BLUE, size=6),
    ))
    fig_cpi.add_trace(go.Scatter(
        x=df_cpi["date"], y=df_cpi["core_yoy"],
        mode="lines+markers", name="Core CPI-ATE YoY",
        line=dict(color=NB_MED_BLUE, width=2, dash="dot"),
        marker=dict(color=NB_MED_BLUE, size=6),
    ))
    fig_cpi.update_layout(
        title="CPI Inflation (Headline vs Core)",
        yaxis_title="YoY %",
        paper_bgcolor="white", plot_bgcolor="white",
        
        legend=dict(
            orientation="h",
            x=0.5, xanchor="center",
            y=-0.25, yanchor="top",
            font=dict(size=10),
            itemsizing="constant",
        ),
        margin=dict(t=70, b=90),
    )


    # --- Unemployment ---
    fig_u = go.Figure()
    fig_u.add_trace(go.Scatter(
        x=df_u["date"], y=df_u["unemployment_sa"],
        mode="lines+markers", name="Unemployment SA",
        line=dict(color=NB_DARK_BLUE, width=2),
        marker=dict(color=NB_DARK_BLUE, size=6),
    ))
    fig_u.update_layout(
        title="Unemployment (SA, 15–74)",
        yaxis_title="% of labour force",
        paper_bgcolor="white", plot_bgcolor="white",
            
        legend=dict(
            orientation="h",
            x=0.5, xanchor="center",
            y=-0.25, yanchor="top",
            font=dict(size=10),
            itemsizing="constant",
        ),
        margin=dict(t=70, b=90),
    )


    # --- Monetary aggregates ---
    fig_m = go.Figure()
    fig_m.add_trace(go.Scatter(
        x=df_m["date"], y=df_m["M1"],
        mode="lines+markers", name="M1",
        line=dict(color=NB_DARK_BLUE, width=2),
        marker=dict(color=NB_DARK_BLUE, size=5),
    ))
    fig_m.add_trace(go.Scatter(
        x=df_m["date"], y=df_m["M2"],
        mode="lines+markers", name="M2",
        line=dict(color=NB_MED_BLUE, width=2),
        marker=dict(color=NB_MED_BLUE, size=5),
    ))
    fig_m.add_trace(go.Scatter(
        x=df_m["date"], y=df_m["M3"],
        mode="lines+markers", name="M3",
        line=dict(color=NB_LIGHT_BLUE, width=2),
        marker=dict(color=NB_LIGHT_BLUE, size=5),
    ))
    fig_m.update_layout(
        title="Monetary Aggregates (M1 / M2 / M3)",
        yaxis_title="Index / NOK bn (level)",
        paper_bgcolor="white", plot_bgcolor="white",

        legend=dict(
            orientation="h",
            x=0.5, xanchor="center",
            y=-0.25, yanchor="top",
            font=dict(size=10),
            itemsizing="constant",
        ),
    margin=dict(t=70, b=95),
    )


    # --- External settlements ---
# --- External settlements ---
    fig_ext = go.Figure()
    for industry in df_in_long["Industry"].unique():
        df_ind = df_in_long[df_in_long["Industry"] == industry]
        fig_ext.add_trace(go.Bar(
            x=df_ind["Quarter_date"], y=df_ind["Value"],
            name=f"Incoming – {industry}",
            marker=dict(
                line=dict(color="white", width=0.2),
            ),
            hovertemplate=(
                "Quarter=%{x|%Y-Q%q}<br>"
                "Industry=%{customdata}<br>"
                "Incoming=%{y:,.0f} NOKm<extra></extra>"
            ),
            customdata=df_ind["Industry"],
        ))

    for industry in df_out_long["Industry"].unique():
        df_ind = df_out_long[df_out_long["Industry"] == industry]
        fig_ext.add_trace(go.Bar(
            x=df_ind["Quarter_date"], y=-df_ind["Value"],
            name=f"Outgoing – {industry}",
            marker=dict(
                line=dict(color="white", width=0.2),
            ),
            customdata=np.column_stack([df_ind["Quarter"], df_ind["Industry"]]),
            hovertemplate=(
                "Quarter=%{customdata[0]}<br>"
                "Industry=%{customdata[1]}<br>"
                "Outgoing=%{y:,.0f} NOKm<extra></extra>"
            ),
        ))

    fig_ext.add_trace(go.Scatter(
        x=df_net["Quarter_date"], y=df_net["net"],
        mode="lines+markers", name="Net (Incoming - Outgoing)",
        line=dict(width=2, color=NB_DARK_BLUE),
        marker=dict(color=NB_DARK_BLUE, size=5),
        hovertemplate="Quarter=%{x|%Y-Q%q}<br>Net flow=%{y:,.0f} NOKm<extra></extra>",
    ))

    fig_ext.update_layout(
        title="External Settlements by Industry",
        barmode="relative",
        yaxis_title="NOK million",
        paper_bgcolor="white", plot_bgcolor="white",
        showlegend=False,
        margin=dict(t=70, b=160),
    )

    add_time_toggles(fig_ext, df_net["Quarter_date"], default_years=10)

    # --- NIIP stocks ---
    colors_stock = {
        "Direct investment":     NB_DARK_BLUE,
        "Portfolio investment":  ACC_ORANGE,
        "Other investments":     ACC_TEAL,
        "Financial derivatives": NB_GREEN,
        "Reserve assets":        NB_GREY,
    }

    fig_niip_stock = go.Figure()

    for func in ["Direct investment", "Portfolio investment",
                 "Other investments", "Financial derivatives", "Reserve assets"]:
        fig_niip_stock.add_trace(go.Bar(
            x=df_niip_stock["quarter_date"],
            y=df_niip_stock[func],
            name=func,
            marker=dict(
                color=colors_stock[func],
                line=dict(color="white", width=0.2),
            ),
        ))

    fig_niip_stock.add_trace(go.Scatter(
        x=df_niip_stock["quarter_date"],
        y=df_niip_stock["Total NIIP"],
        mode="lines+markers",
        name="Total NIIP",
        line=dict(color=NB_DARK_BLUE, width=2),
        marker=dict(color=NB_DARK_BLUE, size=5),
    ))

    fig_niip_stock.update_layout(
        title="NIIP by Function (Stocks)",
        barmode="relative",
        yaxis_title="Net position (NOK million)",
        paper_bgcolor="white",
        plot_bgcolor="white",

        legend=dict(
            orientation="h",
            x=0.5, xanchor="center",
            y=-0.35, yanchor="top",
            font=dict(size=10),
            itemsizing="constant",
        ),
        margin=dict(t=70, b=140),
    )

    add_time_toggles(fig_niip_stock, df_niip_stock["quarter_date"], default_years=10)



    # --- NIIP flows ---
    colors_flow = {
        "Direct investment":     NB_DARK_BLUE,
        "Portfolio investment":  ACC_ORANGE,
        "Other investments":     ACC_TEAL,
        "Financial derivatives": NB_GREEN,
        "Reserve assets":        NB_GREY,
    }
    fig_niip_flow = go.Figure()
    for func in func_cols:
        fig_niip_flow.add_trace(go.Bar(
            x=df_niip_flow["quarter_date"],
            y=df_niip_flow[func],
            name=func,
            marker=dict(color=colors_flow[func],
                        line=dict(color="white", width=0.2)),
        ))

    fig_niip_flow.add_trace(go.Scatter(
        x=df_niip_flow["quarter_date"],
        y=df_niip_flow["Total net flow"],
        mode="lines+markers",
        name="Net NIIP flow",
        line=dict(color=NB_DARK_BLUE, width=2),
        marker=dict(color=NB_DARK_BLUE, size=5),
    ))

    fig_niip_flow.update_layout(
        title="NIIP – Financial Account Flows",
        barmode="relative",
        yaxis_title="Net flow (NOK million)",
        paper_bgcolor="white", plot_bgcolor="white",

        legend=dict(
            orientation="h",
            x=0.5, xanchor="center",
            y=-0.35, yanchor="top",
            font=dict(size=10),
            itemsizing="constant",
        ),
        margin=dict(t=70, b=140),
    )

    add_time_toggles(fig_niip_flow, df_niip_flow["quarter_date"], default_years=10)

    return fig_cpi, fig_u, fig_m, fig_ext, fig_niip_stock, fig_niip_flow

#Dashboard 2 plottt copying it then fixing it from there
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

NB_DARK_BLUE  = "#003B5C"
NB_LIGHT_BLUE = "#4F87B2"


def ssb_to_timestamp(x: str) -> pd.Timestamp:
    y, m = x.split("M")
    return pd.to_datetime(f"{y}-{m}-01")


def quarter_str_to_timestamp(q: str) -> pd.Timestamp:
    year_str, q_str = q.split("K")
    year = int(year_str)
    q_num = int(q_str)
    month_map = {1: 3, 2: 6, 3: 9, 4: 12}
    day_map   = {3: 31, 6: 30, 9: 30, 12: 31}
    month = month_map[q_num]
    day = day_map[month]
    return pd.Timestamp(year=year, month=month, day=day)


# -----------------------------
# Your original CPI loader (unchanged series codes)
# -----------------------------
def load_cpi_idx_for_seasonality():
    # Headline CPI index
    url_headline_idx = (
        "https://data.ssb.no/api/pxwebapi/v2/tables/03013/data?lang=en"
        "&valueCodes[Konsumgrp]=TOTAL"
        "&valueCodes[ContentsCode]=KpiIndMnd"
        "&valueCodes[Tid]=top(240)"
        "&outputformat=json-px"
    )
    jh = requests.get(url_headline_idx, timeout=30)
    jh.raise_for_status()
    jh = jh.json()

    headline_rows = []
    for row in jh["data"]:
        tid = row["key"][-1]
        raw_val = row["values"][0]
        try:
            value = float(raw_val)
        except ValueError:
            continue
        headline_rows.append({"Tid": tid, "headline_index": value})

    df_headline_idx = pd.DataFrame(headline_rows)
    df_headline_idx["date"] = df_headline_idx["Tid"].apply(ssb_to_timestamp)
    df_headline_idx = df_headline_idx[["date", "headline_index"]].sort_values("date")

    # Core CPI-ATE index
    url_core_idx = (
        "https://data.ssb.no/api/pxwebapi/v2/tables/05327/data?lang=en"
        "&valueCodes[Konsumgrp]=JAE_TOTAL"
        "&valueCodes[ContentsCode]=KPIJustIndMnd"
        "&valueCodes[Tid]=top(240)"
        "&outputformat=json-px"
    )
    jc = requests.get(url_core_idx, timeout=30)
    jc.raise_for_status()
    jc = jc.json()

    core_rows = []
    for row in jc["data"]:
        tid = row["key"][-1]
        raw_val = row["values"][0]
        try:
            value = float(raw_val)
        except ValueError:
            continue
        core_rows.append({"Tid": tid, "core_index": value})

    df_core_idx = pd.DataFrame(core_rows)
    df_core_idx["date"] = df_core_idx["Tid"].apply(ssb_to_timestamp)
    df_core_idx = df_core_idx[["date", "core_index"]].sort_values("date")

    df_cpi_idx = pd.merge(df_headline_idx, df_core_idx, on="date", how="inner").sort_values("date")
    df_cpi_idx["headline_mom"] = df_cpi_idx["headline_index"].pct_change() * 100
    df_cpi_idx["core_mom"]     = df_cpi_idx["core_index"].pct_change() * 100

    # important so pivots don’t include NaN row
    df_cpi_idx = df_cpi_idx.dropna(subset=["headline_mom", "core_mom"]).reset_index(drop=True)
    return df_cpi_idx


# -----------------------------
# Unemployment NSA loader with fallback (keeps your IS preference)
# -----------------------------
def load_unemployment_nsa():
    base = (
        "https://data.ssb.no/api/pxwebapi/v2/tables/13760/data?lang=en"
        "&valueCodes[ContentsCode]=ArbledProsArbstyrk"
        "&valueCodes[Kjonn]=0"
        "&valueCodes[Alder]=15-74"
        "&valueCodes[Tid]=top(200)"
        "&outputformat=json-px"
    )

    def fetch(justering_code: str) -> pd.DataFrame:
        url = base + f"&valueCodes[Justering]={justering_code}"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        j = r.json()

        cols = [c["code"] for c in j["columns"]]
        rows = []
        for row in j["data"]:
            key_map = dict(zip(cols, row["key"]))
            tid = key_map["Tid"]
            raw_val = row["values"][0]
            try:
                value = float(raw_val)
            except ValueError:
                continue
            rows.append({"Tid": tid, "unemployment_nsa": value})

        df_u = pd.DataFrame(rows)
        df_u["date"] = df_u["Tid"].apply(ssb_to_timestamp)
        df_u = df_u[["date", "unemployment_nsa"]].sort_values("date").reset_index(drop=True)
        return df_u

    for code in ["IS", "U", "S"]:
        try:
            df_try = fetch(code)
            if len(df_try) > 0:
                return df_try
        except Exception:
            pass

    raise RuntimeError("Unemployment NSA loader failed for Justering=IS/U/S (no data returned).")


# -----------------------------
# Plotly toggle helpers
# -----------------------------
def add_year_window_buttons_for_heatmap(fig: go.Figure, years, default_years=10):
    years = sorted([int(y) for y in years])
    y_min, y_max = years[0], years[-1]
    last = y_max

    def yr_range(n):
        start = max(y_min, last - (n - 1))
        return [start, last]

    buttons = [
        dict(label="5Y",  method="relayout", args=[{"yaxis.range": yr_range(5)}]),
        dict(label="10Y", method="relayout", args=[{"yaxis.range": yr_range(10)}]),
        dict(label="All", method="relayout", args=[{"yaxis.autorange": True}]),
    ]

    if default_years is not None:
        fig.update_yaxes(range=yr_range(default_years))

    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0, y=-0.18,
            xanchor="left", yanchor="top",
            buttons=buttons
        )],
        margin=dict(t=80, b=90),
    )
    return fig


def add_quarter_window_buttons_for_box(fig: go.Figure, df_clean: pd.DataFrame, default_years=10):
    d = df_clean[["qdate", "Q_label", "net"]].dropna().sort_values("qdate").copy()

    last = pd.to_datetime(d["qdate"].max())

    def window(n_years):
        start = last - pd.DateOffset(years=n_years)
        w = d[d["qdate"] >= start]
        return w["Q_label"].tolist(), w["net"].tolist()

    x_all, y_all = d["Q_label"].tolist(), d["net"].tolist()
    x_3, y_3       = window(3)
    x_5, y_5       = window(5)
    x_10, y_10     = window(10)

    buttons = [
        dict(label="3Y",  method="update", args=[{"x": [x_3],  "y": [y_3]},  {"title": "Norway NIIP Seasonality – Quarterly Boxplot (Last 3Y)"}]),
        dict(label="5Y",  method="update", args=[{"x": [x_5],  "y": [y_5]},  {"title": "Norway NIIP Seasonality – Quarterly Boxplot (Last 5Y)"}]),
        dict(label="10Y", method="update", args=[{"x": [x_10], "y": [y_10]}, {"title": "Norway NIIP Seasonality – Quarterly Boxplot (Last 10Y)"}]),
        dict(label="All", method="update", args=[{"x": [x_all],"y": [y_all]},{"title": "Norway NIIP Seasonality – Quarterly Boxplot (All)"}]),
    ]

    # set default
    if default_years == 3:
        fig.data[0].x, fig.data[0].y = x_3, y_3
        fig.update_layout(title="Norway NIIP Seasonality – Quarterly Boxplot (Last 3Y)")
    elif default_years == 5:
        fig.data[0].x, fig.data[0].y = x_5, y_5
        fig.update_layout(title="Norway NIIP Seasonality – Quarterly Boxplot (Last 5Y)")
    elif default_years == 10:
        fig.data[0].x, fig.data[0].y = x_10, y_10
        fig.update_layout(title="Norway NIIP Seasonality – Quarterly Boxplot (Last 10Y)")
    else:
        fig.data[0].x, fig.data[0].y = x_all, y_all
        fig.update_layout(title="Norway NIIP Seasonality – Quarterly Boxplot (All)")

    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0, y=-0.18,
            xanchor="left", yanchor="top",
            buttons=buttons
        )],
        margin=dict(t=80, b=90),
    )
    return fig

def add_year_window_buttons_for_monthly_seasonality(
    fig: go.Figure,
    df: pd.DataFrame,
    date_col: str,
    compute_fn,
    default_years=10,
    title_prefix=""
):
    """
    Adds 5Y / 10Y / All buttons that RECOMPUTE the seasonality values
    and update the figure via Plotly 'update' (x/y or table cells).

    compute_fn(df_filtered) must return either:
      - (x_list, y_lists_dict) for bar charts
      - dict-of-columns for tables
    """
    df = df.sort_values(date_col).copy()
    last_date = pd.to_datetime(df[date_col].max())

    def filtered(n_years):
        if n_years is None:
            return df
        start = last_date - pd.DateOffset(years=n_years)
        return df[df[date_col] >= start]

    # Precompute states
    states = {
        "5Y":  compute_fn(filtered(5)),
        "10Y": compute_fn(filtered(10)),
        "All": compute_fn(filtered(None)),
    }

    def make_args(state_key):
        state = states[state_key]

        # BAR CHART CASE: return (x, ys, names) packaged into trace updates
        if isinstance(state, tuple):
            x, y_by_trace = state  # y_by_trace: dict {trace_index: y_list}
            data_update = {"x": [], "y": []}
            # We update each trace's x/y in order
            for i in range(len(fig.data)):
                data_update["x"].append(x)
                data_update["y"].append(y_by_trace[i])
            layout_update = {"title": f"{title_prefix} (Last {state_key})" if state_key != "All" else f"{title_prefix} (All)"}
            return [data_update, layout_update]

        # TABLE CASE: dict-of-columns
        if isinstance(state, dict):
            data_update = {"cells": [{"values": [state[k] for k in state.keys()]}]}
            layout_update = {"title": f"{title_prefix} (Last {state_key})" if state_key != "All" else f"{title_prefix} (All)"}
            return [data_update, layout_update]

        raise ValueError("compute_fn returned unsupported type.")

    buttons = [
        dict(label="5Y",  method="update", args=make_args("5Y")),
        dict(label="10Y", method="update", args=make_args("10Y")),
        dict(label="All", method="update", args=make_args("All")),
    ]

    # Default state
    default_key = "All" if default_years is None else ("5Y" if default_years == 5 else "10Y")
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0, y=-0.18,
            xanchor="left", yanchor="top",
            buttons=buttons
        )],
        margin=dict(t=80,b=90),
    )

    # Apply default values immediately
    args0 = make_args(default_key)
    # args0[0] is a data update dict
    if "x" in args0[0] and "y" in args0[0]:
        # bar chart update
        for i in range(len(fig.data)):
            fig.data[i].x = args0[0]["x"][i]
            fig.data[i].y = args0[0]["y"][i]
    elif "cells" in args0[0]:
        # table update
        fig.data[0].cells.values = args0[0]["cells"][0]["values"]

    fig.update_layout(**args0[1])
    return fig

def add_niip_table_toggles(fig, df_clean, default_years=10):
    # df_clean must have: qdate (Timestamp), Q (1..4), net
    last = pd.to_datetime(df_clean["qdate"].max())

    def values_for(n_years):
        d = df_clean.copy()
        if n_years is not None:
            start = last - pd.DateOffset(years=n_years)
            d = d[d["qdate"] >= start]

        s = d.groupby("Q")["net"].agg(["mean", "std", "count"]).sort_index()
        s = (s.rename(columns={"mean": "Mean net flow (NOKm)", "std": "Std dev (NOKm)", "count": "Obs"})
              .reset_index()
              .rename(columns={"Q": "Quarter"}))
        return [s[c] for c in s.columns]

    v3, v5, v10 = values_for(3), values_for(5), values_for(10)

    # default = 10Y
    fig.data[0].cells.values = v10
    fig.update_layout(title="NIIP Net Flow Seasonality Table – Last 10Y")

    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0, y=-0.18,
            xanchor="left", yanchor="top",
            buttons=[
                dict(label="3Y",  method="update", args=[{"cells.values": [v3]},  {"title": "NIIP Net Flow Seasonality Table – Last 3Y"}]),
                dict(label="5Y",  method="update", args=[{"cells.values": [v5]},  {"title": "NIIP Net Flow Seasonality Table – Last 5Y"}]),
                dict(label="10Y", method="update", args=[{"cells.values": [v10]}, {"title": "NIIP Net Flow Seasonality Table – Last 10Y"}]),
            ],
        )],
        margin=dict(t=80,b=90),
    )
    return fig


def add_cpi_table_toggles(fig, df_cpi_idx, default_years=10):
    last = pd.to_datetime(df_cpi_idx["date"].max())

    def compute_colors(vals):
        vv = [v for v in vals if pd.notna(v)]
        if not vv:
            return ["white"] * len(vals)
        vmin, vmax = float(min(vv)), float(max(vv))

        out = []
        for v in vals:
            if pd.isna(v):
                out.append("white")
            else:
                z = 0.5 if vmax == vmin else (float(v) - vmin) / (vmax - vmin)
                r, g, b, _ = plt.get_cmap("coolwarm")(z)
                out.append(f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})")
        return out

    def payload(n_years):
        d = df_cpi_idx.copy()
        if n_years is not None:
            start = last - pd.DateOffset(years=n_years)
            d = d[d["date"] >= start]

        d["month"] = d["date"].dt.month
        s = d.groupby("month")[["headline_mom", "core_mom"]].mean().reindex(range(1, 13))

        months = list(range(1, 13))
        h = s["headline_mom"].round(2).tolist()
        c = s["core_mom"].round(2).tolist()

        h_col = compute_colors(h)
        c_col = compute_colors(c)

        cell_colors = [
            ["white"] * 12,
            h_col,
            c_col,
        ]
        return [months, h, c], cell_colors

    v3,  col3  = payload(3)
    v5,  col5  = payload(5)
    v10, col10 = payload(10)

    # default = 10Y
    fig.data[0].cells.values = v10
    fig.data[0].cells.fill.color = col10
    fig.update_layout(title="Seasonal CPI MoM Factors – Last 10Y")

    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0, y=-0.18,
            xanchor="left", yanchor="top",
            buttons=[
                dict(label="3Y",  method="update",
                     args=[{"cells.values": [v3],  "cells.fill.color": [col3]},  {"title": "Seasonal CPI MoM Factors – Last 3Y"}]),
                dict(label="5Y",  method="update",
                     args=[{"cells.values": [v5],  "cells.fill.color": [col5]},  {"title": "Seasonal CPI MoM Factors – Last 5Y"}]),
                dict(label="10Y", method="update",
                     args=[{"cells.values": [v10], "cells.fill.color": [col10]}, {"title": "Seasonal CPI MoM Factors – Last 10Y"}]),
            ],
        )],
        margin=dict(t=80, b=90),
    )
    return fig


def add_unemp_table_toggles(fig, df_u_nsa, default_years=10):
    last = pd.to_datetime(df_u_nsa["date"].max())

    def compute_colors(vals):
        vv = [v for v in vals if pd.notna(v)]
        if not vv:
            return ["white"] * len(vals)
        vmin, vmax = float(min(vv)), float(max(vv))

        out = []
        for v in vals:
            if pd.isna(v):
                out.append("white")
            else:
                z = 0.5 if vmax == vmin else (float(v) - vmin) / (vmax - vmin)
                r, g, b, _ = plt.get_cmap("coolwarm")(z)
                out.append(f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})")
        return out

    def payload(n_years):
        d = df_u_nsa.copy()
        if n_years is not None:
            start = last - pd.DateOffset(years=n_years)
            d = d[d["date"] >= start]

        d["month"] = d["date"].dt.month
        s = d.groupby("month")["unemployment_nsa"].mean().reindex(range(1, 13))

        months = list(range(1, 13))
        u = s.round(2).tolist()
        u_col = compute_colors(u)

        cell_colors = [
            ["white"] * 12,
            u_col,
        ]
        return [months, u], cell_colors

    v3,  col3  = payload(3)
    v5,  col5  = payload(5)
    v10, col10 = payload(10)

    # default = 10Y
    fig.data[0].cells.values = v10
    fig.data[0].cells.fill.color = col10
    fig.update_layout(title="Seasonal Unemployment Pattern by Month (NSA) – Last 10Y")

    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0, y=-0.18,
            xanchor="left", yanchor="top",
            buttons=[
                dict(label="3Y",  method="update",
                     args=[{"cells.values": [v3],  "cells.fill.color": [col3]},  {"title": "Seasonal Unemployment Pattern by Month (NSA) – Last 3Y"}]),
                dict(label="5Y",  method="update",
                     args=[{"cells.values": [v5],  "cells.fill.color": [col5]},  {"title": "Seasonal Unemployment Pattern by Month (NSA) – Last 5Y"}]),
                dict(label="10Y", method="update",
                     args=[{"cells.values": [v10], "cells.fill.color": [col10]}, {"title": "Seasonal Unemployment Pattern by Month (NSA) – Last 10Y"}]),
            ],
        )],
        margin=dict(t=80, b=90),
    )
    return fig


# -----------------------------
# Panel 2: Plotly-only (vertical figures)
# -----------------------------
@st.cache_data(ttl=3600, show_spinner=True)
def dashboard_2_plotly():
    # 1) NIIP Transactions
    url = (
        "https://data.ssb.no/api/pxwebapi/v2/tables/10644/data?lang=en"
        "&valueCodes[FordringGjeld]=F,G"
        "&valueCodes[FunksjObjSektor]=3.1,3.2,3.3,3.4,3.5"
        "&valueCodes[ContentsCode]=Transaksj"
        "&valueCodes[Tid]=top(80)"
        "&outputformat=json-px"
    )
    j = requests.get(url, timeout=30)
    j.raise_for_status()
    j = j.json()

    cols = [c["code"] for c in j["columns"]]
    records = []
    for row in j["data"]:
        key_map = dict(zip(cols, row["key"]))
        side = key_map["FordringGjeld"]
        quarter = key_map["Tid"]
        raw_val = row["values"][0]
        try:
            value = float(raw_val)
        except Exception:
            value = 0.0
        records.append({"quarter": quarter, "side": side, "value": value})

    df = pd.DataFrame(records)

    df_clean = (
        df.groupby(["quarter", "side"])["value"]
          .sum()
          .unstack("side")
          .fillna(0)
          .rename(columns={"F": "assets", "G": "liabilities"})
    )
    df_clean["net"] = df_clean["assets"] - df_clean["liabilities"]

    # your safety filters
    df_clean = df_clean[df_clean["liabilities"] != 0]
    df_clean["assets"] = df_clean["assets"].where(df_clean["assets"] > 100, 0)
    df_clean["liabilities"] = df_clean["liabilities"].where(df_clean["liabilities"] > 100, 0)
    df_clean["net"] = df_clean["assets"] - df_clean["liabilities"]

    df_clean = df_clean.reset_index()
    df_clean["year"] = df_clean["quarter"].str[:4].astype(int)
    df_clean["Q"] = df_clean["quarter"].str[-1].astype(int)
    df_clean["qdate"] = df_clean["quarter"].apply(quarter_str_to_timestamp)
    df_clean = df_clean.sort_values("qdate").reset_index(drop=True)

    # 2) NIIP seasonality table
    seasonality = df_clean.groupby("Q")["net"].agg(["mean", "std", "count"]).sort_index()
    seasonality_display = (
        seasonality.rename(columns={"mean": "Mean net flow (NOKm)", "std": "Std dev (NOKm)", "count": "Obs"})
                   .reset_index()
                   .rename(columns={"Q": "Quarter"})
    )

    # 3) CPI seasonality objects
    df_cpi_idx = load_cpi_idx_for_seasonality().copy()
    df_cpi_idx["year"] = df_cpi_idx["date"].dt.year
    df_cpi_idx["month"] = df_cpi_idx["date"].dt.month
    heat_cpi_mom = df_cpi_idx.pivot_table(index="year", columns="month", values="headline_mom")

    seasonal_cpi = (
        df_cpi_idx.groupby("month")[["headline_mom", "core_mom"]]
                 .mean()
                 .rename(columns={"headline_mom": "headline_mom_avg", "core_mom": "core_mom_avg"})
    )
    seasonal_cpi_rounded = seasonal_cpi.round(2)
    cpi_months = seasonal_cpi_rounded.index.tolist()
    headline_vals = seasonal_cpi_rounded["headline_mom_avg"].tolist()
    core_vals = seasonal_cpi_rounded["core_mom_avg"].tolist()

    cpi_all_vals = headline_vals + core_vals
    cpi_vmin, cpi_vmax = min(cpi_all_vals), max(cpi_all_vals)

    def cpi_value_to_color(val):
        z = 0.5 if cpi_vmax == cpi_vmin else (val - cpi_vmin) / (cpi_vmax - cpi_vmin)
        r, g, b, _ = plt.get_cmap("coolwarm")(z)
        return f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"

    headline_colors = [cpi_value_to_color(v) for v in headline_vals]
    core_colors = [cpi_value_to_color(v) for v in core_vals]
    cpi_cell_colors = [["white"] * len(cpi_months), headline_colors, core_colors]

    # 4) Unemployment seasonality objects
    df_u_nsa = load_unemployment_nsa().copy()
    df_u_nsa["year"] = df_u_nsa["date"].dt.year
    df_u_nsa["month"] = df_u_nsa["date"].dt.month
    heat_u = df_u_nsa.pivot_table(index="year", columns="month", values="unemployment_nsa")

    seasonal_u = df_u_nsa.groupby("month")["unemployment_nsa"].mean().rename("unemployment_avg")
    u_rounded = seasonal_u.round(2)
    u_months = u_rounded.index.tolist()
    u_vals = u_rounded.values.tolist()

    u_vmin, u_vmax = min(u_vals), max(u_vals)

    def u_value_to_color(val):
        z = 0.5 if u_vmax == u_vmin else (val - u_vmin) / (u_vmax - u_vmin)
        r, g, b, _ = plt.get_cmap("coolwarm")(z)
        return f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"

    u_colors = [u_value_to_color(v) for v in u_vals]
    u_cell_colors = [["white"] * len(u_months), u_colors]

    # ---- FIGURES (vertical) ----

    # CPI heatmap
    fig_cpi_heat = go.Figure(go.Heatmap(
        x=heat_cpi_mom.columns, y=heat_cpi_mom.index, z=heat_cpi_mom.values,
        colorscale="RdBu", reversescale=True, zmid=0, showscale=False
    ))
    fig_cpi_heat.update_layout(title="Headline CPI MoM – Seasonality Heatmap", paper_bgcolor="white", plot_bgcolor="white")
    fig_cpi_heat.update_xaxes(title_text="Month", type="category")
    fig_cpi_heat.update_yaxes(title_text="Year")
    add_year_window_buttons_for_heatmap(fig_cpi_heat, heat_cpi_mom.index, default_years=10)

    # Unemployment heatmap
    u_zmid = float(heat_u.stack().mean()) if heat_u.size else 0.0
    fig_u_heat = go.Figure(go.Heatmap(
        x=heat_u.columns, y=heat_u.index, z=heat_u.values,
        colorscale="RdBu", reversescale=True, zmid=u_zmid, showscale=False
    ))
    fig_u_heat.update_layout(title="Unemployment (NSA) – Seasonality Heatmap", paper_bgcolor="white", plot_bgcolor="white")
    fig_u_heat.update_xaxes(title_text="Month", type="category")
    fig_u_heat.update_yaxes(title_text="Year")
    add_year_window_buttons_for_heatmap(fig_u_heat, heat_u.index, default_years=10)

    # -----------------------------
    # NIIP boxplot (FIXED order + white background)
    # -----------------------------

    # 1) Create labels so x-axis shows Q1..Q4 (and not 1..4)
    df_clean["Q_label"] = "Q" + df_clean["Q"].astype(int).astype(str)

    # 2) Build the boxplot using Q_label (NOT df_clean["Q"])
    fig_niip_box = go.Figure(go.Box(
        x=df_clean["Q_label"],
        y=df_clean["net"],
        name="Net NIIP Flow",
        marker_color=NB_LIGHT_BLUE,
        line_color=NB_DARK_BLUE,
        boxpoints="outliers",
    ))

    # 3) Force the category order explicitly (prevents 3,4,1,2 ordering)
    fig_niip_box.update_xaxes(
        title_text="Quarter",
        type="category",
        categoryorder="array",
        categoryarray=["Q1", "Q2", "Q3", "Q4"],
    )

    fig_niip_box.update_yaxes(title_text="Net NIIP Flow (NOKm)")

    # 4) Force white background (prevents the blue/grey theme)
    fig_niip_box.update_layout(
        template="simple_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    # 5) Keep your time toggle buttons (BUT you must update that function too—see below)
    add_quarter_window_buttons_for_box(fig_niip_box, df_clean, default_years=10)


    # CPI seasonal bars
    fig_cpi_bar = go.Figure()
    fig_cpi_bar.add_trace(go.Bar(x=seasonal_cpi.index, y=seasonal_cpi["headline_mom_avg"], name="Headline MoM avg", marker_color=NB_DARK_BLUE))
    fig_cpi_bar.add_trace(go.Bar(x=seasonal_cpi.index, y=seasonal_cpi["core_mom_avg"], name="Core MoM avg", marker_color=NB_LIGHT_BLUE))
    fig_cpi_bar.update_layout(title="Average MoM Inflation by Month (Headline vs Core)", barmode="group", paper_bgcolor="white", plot_bgcolor="white")
    fig_cpi_bar.update_xaxes(title_text="Month", type="category")
    fig_cpi_bar.update_yaxes(title_text="Average MoM %")
    # CPI seasonal bars (with 1Y/3Y/5Y/10Y/All toggles)
    fig_cpi_bar = go.Figure()
    fig_cpi_bar.add_trace(go.Bar(
        x=seasonal_cpi.index,
        y=seasonal_cpi["headline_mom_avg"],
        name="Headline MoM avg",
        marker_color=NB_DARK_BLUE
    ))
    fig_cpi_bar.add_trace(go.Bar(
        x=seasonal_cpi.index,
        y=seasonal_cpi["core_mom_avg"],
        name="Core MoM avg",
        marker_color=NB_LIGHT_BLUE
    ))
    fig_cpi_bar.update_layout(
        title="Average MoM Inflation by Month (Headline vs Core)",
        barmode="group",
        paper_bgcolor="white",
        plot_bgcolor="white"
    )
    fig_cpi_bar.update_xaxes(title_text="Month", type="category")
    fig_cpi_bar.update_yaxes(title_text="Average MoM %")

    # -----------------------------
    # Time toggles (1Y/3Y/5Y/10Y/All)
    # -----------------------------
    _last = df_cpi_idx["date"].max()

    def _cpi_monthly_means(df_):
        tmp = df_.copy()
        tmp["month"] = tmp["date"].dt.month
        s = tmp.groupby("month")[["headline_mom", "core_mom"]].mean()
        s = s.reindex(range(1, 13))  # ensure months 1..12
        return s["headline_mom"].tolist(), s["core_mom"].tolist()

    def _df_last_n_years(df_, n_years):
        if n_years is None:
            return df_
        start = _last - pd.DateOffset(years=n_years)
        return df_[df_["date"] >= start]

    h_1y,  c_1y  = _cpi_monthly_means(_df_last_n_years(df_cpi_idx, 1))
    h_3y,  c_3y  = _cpi_monthly_means(_df_last_n_years(df_cpi_idx, 3))
    h_5y,  c_5y  = _cpi_monthly_means(_df_last_n_years(df_cpi_idx, 5))
    h_10y, c_10y = _cpi_monthly_means(_df_last_n_years(df_cpi_idx, 10))
    h_all, c_all = _cpi_monthly_means(_df_last_n_years(df_cpi_idx, None))

    months_1_12 = list(range(1, 13))

    # default view = 10Y (change if you prefer)
    fig_cpi_bar.data[0].x = months_1_12
    fig_cpi_bar.data[0].y = h_10y
    fig_cpi_bar.data[1].x = months_1_12
    fig_cpi_bar.data[1].y = c_10y
    fig_cpi_bar.update_layout(title="Average MoM Inflation by Month (Headline vs Core) – Last 10Y")

    fig_cpi_bar.update_layout(
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0, y=-0.18,
            xanchor="left", yanchor="top",
            buttons=[
                dict(label="1Y",  method="update",
                    args=[{"x": [months_1_12, months_1_12], "y": [h_1y,  c_1y]},
                          {"title": "Average MoM Inflation by Month (Headline vs Core) – Last 1Y"}]),
                dict(label="3Y",  method="update",
                    args=[{"x": [months_1_12, months_1_12], "y": [h_3y,  c_3y]},
                          {"title": "Average MoM Inflation by Month (Headline vs Core) – Last 3Y"}]),
                dict(label="5Y",  method="update",
                    args=[{"x": [months_1_12, months_1_12], "y": [h_5y,  c_5y]},
                          {"title": "Average MoM Inflation by Month (Headline vs Core) – Last 5Y"}]),
                dict(label="10Y", method="update",
                    args=[{"x": [months_1_12, months_1_12], "y": [h_10y, c_10y]},
                          {"title": "Average MoM Inflation by Month (Headline vs Core) – Last 10Y"}]),
                dict(label="All", method="update",
                    args=[{"x": [months_1_12, months_1_12], "y": [h_all, c_all]},
                          {"title": "Average MoM Inflation by Month (Headline vs Core) – All"}]),
            ]
        )],
        margin=dict(t=80, b=90),
    )

    # Unemployment seasonal bars
    fig_u_bar = go.Figure(go.Bar(x=seasonal_u.index, y=seasonal_u.values, name="Avg unemployment (NSA)", marker_color=NB_DARK_BLUE))
    fig_u_bar.update_layout(title="Average Unemployment Rate by Month (NSA)", paper_bgcolor="white", plot_bgcolor="white")
    fig_u_bar.update_xaxes(title_text="Month", type="category")
    fig_u_bar.update_yaxes(title_text="Unemployment %")

    # Unemployment seasonal bars (with 1Y/3Y/5Y/10Y/All toggles)
    fig_u_bar = go.Figure(go.Bar(
        x=seasonal_u.index,
        y=seasonal_u.values,
        name="Avg unemployment (NSA)",
        marker_color=NB_DARK_BLUE
    ))
    fig_u_bar.update_layout(
        title="Average Unemployment Rate by Month (NSA)",
        paper_bgcolor="white",
        plot_bgcolor="white"
    )
    fig_u_bar.update_xaxes(title_text="Month", type="category")
    fig_u_bar.update_yaxes(title_text="Unemployment %")

    # -----------------------------
    # Time toggles (1Y/3Y/5Y/10Y/All)
    # -----------------------------
    _last_u = df_u_nsa["date"].max()

    def _u_monthly_means(df_):
        tmp = df_.copy()
        tmp["month"] = tmp["date"].dt.month
        s = tmp.groupby("month")["unemployment_nsa"].mean()
        s = s.reindex(range(1, 13))
        return s.tolist()

    def _df_last_n_years_u(df_, n_years):
        if n_years is None:
            return df_
        start = _last_u - pd.DateOffset(years=n_years)
        return df_[df_["date"] >= start]

    u_1y  = _u_monthly_means(_df_last_n_years_u(df_u_nsa, 1))
    u_3y  = _u_monthly_means(_df_last_n_years_u(df_u_nsa, 3))
    u_5y  = _u_monthly_means(_df_last_n_years_u(df_u_nsa, 5))
    u_10y = _u_monthly_means(_df_last_n_years_u(df_u_nsa, 10))
    u_all = _u_monthly_means(_df_last_n_years_u(df_u_nsa, None))

    months_1_12 = list(range(1, 13))

    # default view = 10Y
    fig_u_bar.data[0].x = months_1_12
    fig_u_bar.data[0].y = u_10y
    fig_u_bar.update_layout(title="Average Unemployment Rate by Month (NSA) – Last 10Y")

    fig_u_bar.update_layout(
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0, y=-0.18,
            xanchor="left", yanchor="top",
            buttons=[
                dict(label="1Y",  method="update",
                    args=[{"x": [months_1_12], "y": [u_1y]},
                          {"title": "Average Unemployment Rate by Month (NSA) – Last 1Y"}]),
                dict(label="3Y",  method="update",
                    args=[{"x": [months_1_12], "y": [u_3y]},
                          {"title": "Average Unemployment Rate by Month (NSA) – Last 3Y"}]),
                dict(label="5Y",  method="update",
                    args=[{"x": [months_1_12], "y": [u_5y]},
                          {"title": "Average Unemployment Rate by Month (NSA) – Last 5Y"}]),
                dict(label="10Y", method="update",
                    args=[{"x": [months_1_12], "y": [u_10y]},
                          {"title": "Average Unemployment Rate by Month (NSA) – Last 10Y"}]),
                dict(label="All", method="update",
                    args=[{"x": [months_1_12], "y": [u_all]},
                          {"title": "Average Unemployment Rate by Month (NSA) – All"}]),
            ]
        )],
        margin=dict(t=80, b=90),
    )

    # NIIP seasonality table
    fig_niip_table = go.Figure(go.Table(
        header=dict(values=list(seasonality_display.columns), fill_color=NB_DARK_BLUE, align="center",
                    font=dict(color="white", size=12), height=32),
        cells=dict(values=[seasonality_display[c] for c in seasonality_display.columns], align="center",
                   fill_color="white", font=dict(color="black", size=11), height=28),
    ))
    fig_niip_table.update_layout(title="NIIP Net Flow Seasonality Table")
    add_niip_table_toggles(fig_niip_table, df_clean, default_years=10)

    # CPI conditional table
    fig_cpi_table = go.Figure(go.Table(
        header=dict(values=["Month", "Headline MoM Avg (%)", "Core MoM Avg (%)"],
                    fill_color=NB_DARK_BLUE, align="center",
                    font=dict(color="white", size=12), height=32),
        cells=dict(values=[cpi_months, headline_vals, core_vals],
                   fill_color=cpi_cell_colors, align="center", height=28, font=dict(size=12)),
    ))
    fig_cpi_table.update_layout(title="Seasonal CPI MoM Factors")
    add_cpi_table_toggles(fig_cpi_table, df_cpi_idx, default_years=10)


    # Unemployment conditional table
    fig_u_table = go.Figure(go.Table(
        header=dict(values=["Month", "Unemployment Avg (%)"],
                    fill_color=NB_DARK_BLUE, align="center",
                    font=dict(color="white", size=12), height=32),
        cells=dict(values=[u_months, u_vals],
                   fill_color=u_cell_colors, align="center", height=28, font=dict(size=12)),
    ))
    fig_u_table.update_layout(title="Seasonal Unemployment Pattern by Month (NSA)")
    add_unemp_table_toggles(fig_u_table, df_u_nsa, default_years=10)


    return [fig_cpi_heat, fig_u_heat, fig_niip_box, fig_cpi_bar, fig_u_bar, fig_niip_table, fig_cpi_table, fig_u_table]

#Dashboard 3!!!

@st.cache_data(ttl=3600, show_spinner=True)
def dashboard_3():

    # =========================================================
    # 1) Helper functions
    # =========================================================

    def tenor_to_years(t):
        """
        Convert a tenor string like '3M', '6M', '2Y' into years (float).
        """
        t = str(t).strip().upper()

        if t.endswith("M"):  # months -> years
            return int(t[:-1]) / 12.0

        if t.endswith("Y"):  # years -> years
            return float(t[:-1])

        return np.nan

    def extract_nowa_measure(measure_code):
        """
        Extracts a specific UNIT_MEASURE (T, V, BL, R, BB)
        for NOWA ON from Norges Bank SHORT_RATES dataset.

        measure_code: 'T', 'V', 'BL', 'R', 'BB'
        Returns a DataFrame with columns: ['date', measure_code]
        """
        url = (
            "https://data.norges-bank.no/api/data/"
            "SHORT_RATES/B.NOWA.ON.?format=sdmx-json&lastNObservations=360&locale=en"
        )

        resp = requests.get(url)
        resp.raise_for_status()
        j = resp.json()

        # ---- Extract observation dates ----
        obs_dims = j["data"]["structure"]["dimensions"]["observation"]
        time_vals = [d["values"] for d in obs_dims if d["id"] == "TIME_PERIOD"][0]
        dates = [v["id"] for v in time_vals]

        # ---- Extract UNIT_MEASURE dimension ----
        series_dims = j["data"]["structure"]["dimensions"]["series"]
        measure_pos = next(i for i, d in enumerate(series_dims) if d["id"] == "UNIT_MEASURE")
        measure_vals = series_dims[measure_pos]["values"]

        # index of the requested measure (e.g., 'R')
        try:
            measure_idx = next(i for i, v in enumerate(measure_vals) if v["id"] == measure_code)
        except StopIteration:
            raise ValueError(f"UNIT_MEASURE '{measure_code}' not found.")

        # ---- Find the series that matches this measure ----
        series = j["data"]["dataSets"][0]["series"]
        target_obs = None

        for key, entry in series.items():
            idxs = list(map(int, key.split(":")))
            if idxs[measure_pos] == measure_idx:
                target_obs = entry["observations"]
                break

        if target_obs is None:
            raise ValueError(f"No series found for UNIT_MEASURE = {measure_code}")

        # ---- Build tidy dataframe ----
        records = []
        for obs_idx_str, val in target_obs.items():
            idx = int(obs_idx_str)
            records.append({
                "date": pd.to_datetime(dates[idx]),
                measure_code: float(val[0]),
            })

        df = pd.DataFrame(records).sort_values("date")
        return df

    # =========================================================
    # 2) Data loaders
    # =========================================================

    def load_govt_curve(last_n=60):
        """
        Pulls TBIL + GBON generic government curves (last N observations)
        and returns a cleaned DataFrame.
        """
        url = (
            "https://data.norges-bank.no/api/data/"
            f"GOVT_GENERIC_RATES/B..TBIL+GBON.?format=csv"
            f"&lastNObservations={last_n}&locale=en"
        )
        df = pd.read_csv(url, sep=';')

        df_clean = df[["TIME_PERIOD", "TENOR", "INSTRUMENT_TYPE", "OBS_VALUE"]].copy()
        df_clean = df_clean.rename(columns={
            "TIME_PERIOD": "date",
            "TENOR": "tenor",
            "INSTRUMENT_TYPE": "instrument",
            "OBS_VALUE": "yield_pct",
        })
        df_clean["date"] = pd.to_datetime(df_clean["date"])
        df_clean["yield_pct"] = pd.to_numeric(df_clean["yield_pct"], errors="coerce")
        df_clean["maturity_years"] = df_clean["tenor"].apply(tenor_to_years)

        df_clean = df_clean.sort_values(["date", "maturity_years"])
        return df_clean

    def load_nowa():
        """
        Pulls NOWA ON series and merges:
        R (rate), T (transactions), V (volume),
        BL (banks lending), BB (banks borrowing).
        """
        df_R  = extract_nowa_measure("R")   # Rate
        df_T  = extract_nowa_measure("T")   # Transactions count
        df_V  = extract_nowa_measure("V")   # Volume
        df_BL = extract_nowa_measure("BL")  # Banks lending
        df_BB = extract_nowa_measure("BB")  # Banks borrowing

        df_nowa = (
            df_R.merge(df_T, on="date")
                .merge(df_V, on="date")
                .merge(df_BL, on="date")
                .merge(df_BB, on="date")
        )
        return df_nowa

    def load_nok_fx(years=10, fallback_last_n=2600):
        """
        Pull USDNOK, EURNOK, AUDNOK spot rates from Norges Bank EXR dataset.

        Preferred: use startPeriod to get exactly ~years of history.
        Fallback: lastNObservations (count-based) if startPeriod not accepted.
        """
        end = pd.Timestamp.today().normalize()
        start = (end - pd.DateOffset(years=years)).date().isoformat()

        base = "https://data.norges-bank.no/api/data/EXR/B.USD+EUR+AUD+GBP.NOK.SP"
        url_startperiod = f"{base}?format=sdmx-json&startPeriod={start}&locale=en"
        url_lastn = f"{base}?format=sdmx-json&lastNObservations={fallback_last_n}&locale=en"

        try:
            resp = requests.get(url_startperiod, timeout=30)
            resp.raise_for_status()
            j = resp.json()
        except Exception:
            resp = requests.get(url_lastn, timeout=30)
            resp.raise_for_status()
            j = resp.json()

        # ---- Time dimension ----
        obs_dims = j["data"]["structure"]["dimensions"]["observation"]
        time_vals = next(d["values"] for d in obs_dims if d["id"] == "TIME_PERIOD")
        dates = [pd.to_datetime(v["id"]) for v in time_vals]

        # ---- Base currency dimension (USD/EUR/AUD) ----
        series_dims = j["data"]["structure"]["dimensions"]["series"]
        base_pos = next(i for i, d in enumerate(series_dims) if d["id"] in ("BASE_CUR", "CURRENCY"))
        base_currencies = [v["id"] for v in series_dims[base_pos]["values"]]

        # ---- Observations ----
        records = []
        for key, entry in j["data"]["dataSets"][0]["series"].items():
            idxs = list(map(int, key.split(":")))
            base_cur = base_currencies[idxs[base_pos]]

            for obs_idx_str, val_arr in entry["observations"].items():
                obs_idx = int(obs_idx_str)
                records.append({"date": dates[obs_idx], "currency": base_cur, "value": float(val_arr[0])})

        df_long = pd.DataFrame(records).sort_values("date")
        df_wide = df_long.pivot(index="date", columns="currency", values="value")

        return df_long, df_wide


    # =========================================================
    # 3) Figure builders
    # =========================================================

    def build_govt_curve_fig(df_gov):
        """
        Yield curve figure for latest 4 observation dates.
        """
        unique_dates = sorted(df_gov["date"].unique())
        dates_to_plot = unique_dates[-4:]
        latest = dates_to_plot[-1]

        fig = go.Figure()

        nb_dark_blue = "#003B5C"
        palette = px.colors.sequential.Blues
        base_colors = palette[1:1 + (len(dates_to_plot) - 1)]

        for i, d in enumerate(dates_to_plot):
            curve = (
                df_gov[df_gov["date"] == d]
                .sort_values("maturity_years")
            )

            if d == latest:
                line_width = 4
                marker_size = 9
                line_color = nb_dark_blue
                name = f"{d.strftime('%Y-%m-%d')} (Latest)"
            else:
                line_width = 2
                marker_size = 7
                line_color = base_colors[i]
                name = d.strftime("%Y-%m-%d")

            fig.add_trace(
                go.Scatter(
                    x=curve["maturity_years"],
                    y=curve["yield_pct"],
                    mode="lines+markers",
                    line=dict(width=line_width, color=line_color),
                    marker=dict(size=marker_size),
                    name=name,
                )
            )

        fig.update_layout(
            title="Norwegian Govt Curve – Recent Dates (TBIL + GBON)",
            xaxis_title="Maturity (years)",
            yaxis_title="Yield (%)",
            template="simple_white",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40, r=30, t=60, b=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(0,0,0,0)",
            ),
        )

        fig.update_xaxes(showgrid=True, gridcolor="rgba(200,200,200,0.4)")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(200,200,200,0.4)")

        return fig

    def build_nowa_rate_volume_fig(df_nowa):
        """
        Single chart with two y-axes (no subplots):
          - NOWA rate (left axis, line)
          - Overnight volume (right axis, bars)
        """
        df = df_nowa.sort_values("date").copy()

        fig = go.Figure()

        # Rate (left axis)
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["R"],
                mode="lines",
                name="NOWA rate",
                line=dict(width=2, color="#003B5C"),
                yaxis="y1",
            )
        )

        # Volume (right axis)
        fig.add_trace(
            go.Bar(
                x=df["date"],
                y=df["V"],
                name="Overnight volume",
                opacity=0.6,
                marker=dict(color="#4F87B2"),
                yaxis="y2",
            )
        )

        fig.update_layout(
            title="NOWA Overnight – Rate vs Volume",
            template="simple_white",
            margin=dict(l=60, r=60, t=70, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),

            # Left axis
            yaxis=dict(title="Rate (%)", showgrid=True, gridcolor="rgba(200,200,200,0.4)"),

            # Right axis (overlay)
            yaxis2=dict(
                title="Volume (NOK mn)",
                overlaying="y",
                side="right",
                showgrid=False,
            ),

            xaxis=dict(
                title="Date",
                type="date",
                tickformat="%b\n%Y",
                dtick="M1",
                showgrid=True,
                gridcolor="rgba(200,200,200,0.4)",
            ),
        )

        return fig

    # These imports should be here or at the top of the cell
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    def build_nowa_market_health_figs(df_nowa):
      # --- Prep: focus on last 180 days & add avg deal size ---
      df = df_nowa.copy().sort_values("date")
      cutoff = df["date"].max() - pd.Timedelta(days=180)
      df = df[df["date"] >= cutoff].reset_index(drop=True)

      df["avg_deal_size"] = df["V"] / df["T"]          # NOK mn per trade
      df["V_ma5"] = df["V"].rolling(5).mean()          # 5-day vol MA

      # --- Figure with 3 stacked rows ---
      fig = make_subplots(
          rows=3,
          cols=1,
          shared_xaxes=True,
          vertical_spacing=0.05,
          row_heights=[0.45, 0.30, 0.25],
      )

      # =========================
      # Row 1: Volume + MA(5)
      # =========================
      fig.add_trace(
          go.Bar(
              x=df["date"],
              y=df["V"],
              name="Overnight volume (NOK mn)",
              marker=dict(color="#4F87B2"),
              opacity=0.6,
              hovertemplate="%{x|%d %b %Y}<br>Volume: %{y:,.0f} mn NOK<extra></extra>",
          ),
          row=1, col=1,
      )

      fig.add_trace(
          go.Scatter(
              x=df["date"],
              y=df["V_ma5"],
              name="Volume 5d MA",
              mode="lines",
              line=dict(color="#003B5C", width=2),
              hovertemplate="%{x|%d %b %Y}<br>5d MA: %{y:,.0f} mn NOK<extra></extra>",

          ),
          row=1, col=1,
      )

      # =========================
      # Row 2: T, BL, BB
      # =========================
      fig.add_trace(
          go.Scatter(
              x=df["date"],
              y=df["T"],
              name="Transactions (T)",
              mode="lines",
              line=dict(color="#003B5C", width=2),
              hovertemplate="%{x|%d %b %Y}<br>Transactions: %{y}<extra></extra>",

          ),
          row=2, col=1,
      )

      fig.add_trace(
          go.Scatter(
              x=df["date"],
              y=df["BL"],
              name="Banks lending (BL)",
              mode="lines",
              line=dict(color="#2A8C55", width=1.5, dash="dot"),
              hovertemplate="%{x|%d %b %Y}<br>Banks lending: %{y}<extra></extra>",
          ),
          row=2, col=1,
      )

      fig.add_trace(
          go.Scatter(
              x=df["date"],
              y=df["BB"],
              name="Banks borrowing (BB)",
              mode="lines",
              line=dict(color="#B03A2E", width=1.5, dash="dot"),
              hovertemplate="%{x|%d %b %Y}<br>Banks borrowing: %{y}<extra></extra>",

          ),
          row=2, col=1,
      )

      # =========================
      # Row 3: Average deal size
      # =========================
      fig.add_trace(
          go.Scatter(
              x=df["date"],
              y=df["avg_deal_size"],
              name="Avg deal size (V/T, NOK mn)",
              mode="lines+markers",
              line=dict(color="#003B5C", width=2),
              marker=dict(size=4),
              hovertemplate="%{x|%d %b %Y}<br>Avg deal size: %{y:,.1f} mn NOK<extra></extra>",

          ),
          row=3, col=1,
      )

      # --- Layout / axes ---
      fig.update_yaxes(
          title_text="Volume (NOK mn)",
          row=1, col=1,
          showgrid=True,
          gridcolor="rgba(200,200,200,0.4)",
          title_font=dict(size=11),
          tickfont=dict(size=9),
          automargin=True,
      )

      fig.update_yaxes(
          title_text="Trades / Banks",
          row=2, col=1,
          showgrid=True,
          gridcolor="rgba(220,220,220,0.4)",
          title_font=dict(size=11),
          tickfont=dict(size=9),
          automargin=True,
      )

      fig.update_yaxes(
          title_text="Avg deal size (NOK mn)",
          row=3, col=1,
          showgrid=True,
          gridcolor="rgba(220,220,220,0.4)",
          title_font=dict(size=11),
          tickfont=dict(size=9),
          automargin=True,
      )


      fig.update_xaxes(
          showgrid=True,
          gridcolor="rgba(230,230,230,0.4)",
          tickformat="%d\n%b",
      )

      fig.update_layout(
          title="NOWA Market Health",
          legend=dict(
              orientation="h",
              yanchor="top",
              y=-0.25,
              xanchor="center",
              x=0.5,
              bgcolor="rgba(0,0,0,0)",
          ),
          margin=dict(l=90, r=40, t=80, b=40),
          plot_bgcolor="rgba(0,0,0,0)",
          paper_bgcolor="rgba(0,0,0,0)",
      )

      return fig # Corrected: return the single figure object

    def build_fx_line_fig(df_fx_wide, ccy="USD"):
        if ccy not in df_fx_wide.columns:
            raise ValueError(f"{ccy} not in FX dataframe columns: {df_fx_wide.columns.tolist()}")

        df = df_fx_wide[[ccy]].dropna().reset_index().rename(columns={"index": "date"})

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["date"], y=df[ccy],
            mode="lines",
            name=f"{ccy}NOK",
            line=dict(width=2, color="#003B5C"),
            hovertemplate="%{x|%d %b %Y}<br>" + f"{ccy}NOK: " + "%{y:.3f}<extra></extra>",
        ))

        fig.update_layout(
            title=f"{ccy}NOK – Spot rate",
            xaxis_title="Date",
            yaxis_title=f"{ccy}NOK",
            template="simple_white",
            margin=dict(l=60, r=40, t=60, b=70),  # extra bottom space for the slider
        )

        fig.update_xaxes(
            type="date",
            showgrid=True,
            gridcolor="rgba(200,200,200,0.4)",

            # Slider at bottom
            rangeslider=dict(visible=True, thickness=0.05),

            # Optional quick buttons (kept small and out of the title area)
            rangeselector=dict(
                buttons=[
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year",  stepmode="backward"),
                    dict(count=3, label="3Y", step="year",  stepmode="backward"),
                    dict(count=5, label="5Y", step="year",  stepmode="backward"),
                    dict(count=10,label="10Y",step="year",  stepmode="backward"),
                    dict(step="all", label="All"),
                ],
                x=0, xanchor="left",
                y=1.02, yanchor="bottom",   # sits just above plot area, typically not over the title
                font=dict(size=10),
            ),
        )

        fig.update_yaxes(showgrid=True, gridcolor="rgba(200,200,200,0.4)")
        return fig


    # =========================================================
    # 4) MAIN DASHBOARD LOGIC (no __main__ check here)
    # =========================================================

    # --- Load data ---
    df_gov = load_govt_curve()
    df_nowa = load_nowa()
    df_fx_long, df_fx_wide = load_nok_fx()

    # --- Individual figs ---
    fig_curve = build_govt_curve_fig(df_gov)
    fig_nowa_rv = build_nowa_rate_volume_fig(df_nowa)

    # NOWA market health now returns a single fig
    fig_nowa_health = build_nowa_market_health_figs(df_nowa)

    fig_usdnok = build_fx_line_fig(df_fx_wide, ccy="USD")
    fig_eurnok = build_fx_line_fig(df_fx_wide, ccy="EUR")
    fig_audnok = build_fx_line_fig(df_fx_wide, ccy="AUD")

    # Return as a vertical list
    return [fig_curve, fig_nowa_rv, fig_nowa_health, fig_usdnok, fig_eurnok, fig_audnok], df_fx_wide


import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def fx_seasonality_matrix(
    df_fx_wide: pd.DataFrame,
    ccy: str,
    method: str = "pct",
    years_exclude=None,
    years_include=None,
):
    """
    Returns:
      mat: DataFrame indexed by year, columns Jan..Dec with monthly returns in %
      year_ret: Series of full-year return in %
      avg_row: Series of average monthly return in %
      grow_fall: DataFrame with 'Up'/'Down' counts per month
    """
    s = df_fx_wide[ccy].dropna().sort_index()
    s.index = pd.to_datetime(s.index)

    # Optional year filtering
    if years_include is not None:

        years_include = set(map(int, years_include))
        s = s[s.index.year.isin(years_include)]

    if years_exclude is not None:
        years_exclude = set(map(int, years_exclude))
        s = s[~s.index.year.isin(years_exclude)]


    # Month-end levels
    m = s.resample("ME").last()

    # Monthly returns
    if method == "log":
        r = np.log(m).diff()
        r_pct = r * 100.0
        # "Year" uses log-sum then convert to %
        year_ret = r.groupby(r.index.year).sum()
        year_ret = (np.exp(year_ret) - 1.0) * 100.0
    else:
        r = m.pct_change()
        r_pct = r * 100.0
        # "Year" uses compounded monthly returns
        year_ret = (1.0 + r).groupby(r.index.year).prod() - 1.0
        year_ret = year_ret * 100.0

    df = r_pct.to_frame("ret_pct")
    df["year"] = df.index.year
    df["month"] = df.index.month

    mat = df.pivot_table(index="year", columns="month", values="ret_pct", aggfunc="mean")
    mat = mat.reindex(columns=range(1, 13))  # Jan..Dec

    # Average row (across years)
    avg_row = mat.mean(axis=0)

    # Growing/falling counts per QUARTER (Q1..Q4) + Avg quarterly return
    # Build quarter-end levels from month-end levels
    q = m.resample("QE").last()

    if method == "log":
        q_ret_pct = np.log(q).diff() * 100.0
    else:
        q_ret_pct = q.pct_change() * 100.0

    dfq = q_ret_pct.to_frame("q_ret_pct")
    dfq["year"] = dfq.index.year
    dfq["quarter"] = dfq.index.quarter

    qmat = dfq.pivot_table(index="year", columns="quarter", values="q_ret_pct", aggfunc="mean")
    qmat = qmat.reindex(columns=[1, 2, 3, 4])

    up = (qmat > 0).sum(axis=0)
    down = (qmat < 0).sum(axis=0)
    avg_q = qmat.mean(axis=0)

    grow_fall = pd.DataFrame({"Up": up, "Down": down, "AvgRet": avg_q})


    # Align year returns to mat years
    year_ret = year_ret.reindex(mat.index)

    # Pretty month labels
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    mat.columns = month_labels
    avg_row.index = month_labels
    grow_fall.index = ["Q1", "Q2", "Q3", "Q4"]

    return mat, year_ret, avg_row, grow_fall


def plot_seasonality_table(
    df_fx_wide: pd.DataFrame,
    ccy: str,
    method: str = "pct",
    title=None,
    years_exclude=None,
    years_include=None,
):

    mat, year_ret, avg_row, grow_fall = fx_seasonality_matrix(
        df_fx_wide,
        ccy,
        method=method,
        years_exclude=years_exclude,
        years_include=years_include,
    )

    # Quarterly averages computed from monthly matrix
    q_avg = pd.DataFrame(index=mat.index)
    q_avg["Q1 Avg"] = mat[["Jan", "Feb", "Mar"]].mean(axis=1)
    q_avg["Q2 Avg"] = mat[["Apr", "May", "Jun"]].mean(axis=1)
    q_avg["Q3 Avg"] = mat[["Jul", "Aug", "Sep"]].mean(axis=1)
    q_avg["Q4 Avg"] = mat[["Oct", "Nov", "Dec"]].mean(axis=1)


    # Build display matrix with extra "Year" column and "Average" row
    disp = mat.copy()
    disp["Year"] = year_ret.values
    disp.loc["Average"] = list(avg_row.values) + [year_ret.mean()]

    # Numeric array for coloring
    z = disp.values.astype(float)

    # Text labels like "2.54%"
    text = np.where(np.isnan(z), "", np.vectorize(lambda v: f"{v:.2f}%")(z))

    # Symmetric color scale around 0
    finite = z[np.isfinite(z)]
    zmax = np.percentile(np.abs(finite), 95) if finite.size else 1.0
    zmax = float(max(zmax, 0.01))

    # ✅ CREATE FIGURE BEFORE USING fig.update_layout / add_annotation / add_shape
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=list(disp.columns),
            y=[str(i) for i in disp.index],
            colorscale=[
                [0.0, "#8b1e2d"],
                [0.5, "#f2f2f2"],
                [1.0, "#1f7a4a"],
            ],
            zmid=0,
            zmin=-zmax,
            zmax=zmax,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 12, "color": "#111111"},
            hovertemplate="Year %{y}<br>%{x}: %{z:.2f}%<extra></extra>",
            showscale=False,
        )
    )

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        title=dict(
            text=title or f"{ccy}NOK Seasonality (Monthly Returns) — {('Log' if method=='log' else 'Percent')}",
            font=dict(color="#111111", size=18),
            x=0.01,
            xanchor="left",
        ),
        margin=dict(l=60, r=30, t=60, b=90),
        font=dict(color="#111111"),
        xaxis=dict(side="top", tickfont=dict(color="#111111")),
        yaxis=dict(autorange="reversed", tickfont=dict(color="#111111")),
    )


    gf_text = "  |  ".join([f"{m}: ▲{grow_fall.loc[m,'Up']} ▼{grow_fall.loc[m,'Down']}" for m in grow_fall.index])
    fig.add_annotation(
        text=f"Growing/falling: {gf_text}",
        xref="paper", yref="paper",
        x=0, y=-0.18,
        showarrow=False,
        align="left",
        font=dict(size=11, color="#111111")
    )



    # =========================
    # Quarter-average strip (clean colors + consistent fonts + centered text)
    # =========================
    q_avg = grow_fall["AvgRet"].astype(float)  # Q1..Q4 average quarter return (%)

    # Use EXACT same colorscale as the heatmap
    heat_colorscale = [
        [0.0, "#8b1e2d"],   # red
        [0.5, "#f2f2f2"],   # neutral
        [1.0, "#1f7a4a"],   # green
    ]

    # IMPORTANT: scale the strip using quarter data (not monthly zmax),
    # otherwise colors look washed out.
    q_finite = q_avg[np.isfinite(q_avg)].values
    qmax = np.percentile(np.abs(q_finite), 95) if q_finite.size else 1.0
    qmax = float(max(qmax, 0.01))

    from plotly.colors import sample_colorscale

    def strip_color(v):
        # map v in [-qmax, +qmax] -> [0, 1] for colorscale sampling
        if not np.isfinite(v):
            t = 0.5
        else:
            t = (v + qmax) / (2 * qmax)
            t = max(0.0, min(1.0, t))
        return sample_colorscale(heat_colorscale, t)[0]  # returns 'rgb(r,g,b)'

    # Strip placement (paper coordinates, below the heatmap)
    y0, y1 = -0.30, -0.22
    quarter_spans = [
        ("Q1", -0.5,  2.5),  # Jan..Mar
        ("Q2",  2.5,  5.5),  # Apr..Jun
        ("Q3",  5.5,  8.5),  # Jul..Sep
        ("Q4",  8.5, 11.5),  # Oct..Dec
    ]

    y_mid = (y0 + y1) / 2

    for q, x0, x1 in quarter_spans:
        v = float(q_avg.loc[q])

        # rectangle "merged cell"
        fig.add_shape(
            type="rect",
            xref="x", yref="paper",
            x0=x0, x1=x1,
            y0=y0, y1=y1,
            line=dict(color="#111111", width=1),
            fillcolor=strip_color(v),
            layer="below",
        )

        # centered label
        fig.add_annotation(
            x=(x0 + x1) / 2,
            y=y_mid,
            xref="x", yref="paper",
            text=f"{q} Avg: {v:.2f}%",
            showarrow=False,
            xanchor="center",
            yanchor="middle",          # <-- vertical centering fix
            align="center",
            font=dict(                 # keep consistent with rest of figure
                family=fig.layout.font.family,
                size=fig.layout.font.size,   # match global font size
                color="#111111",
            ),
        )

    # give enough bottom space so strip doesn't get clipped
    fig.update_layout(margin=dict(l=60, r=30, t=60, b=150))


    return fig

#Plot everything out in streamlit

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Seasonality", "Rates & FX", "FX playground"])

with tab1:
    fig_cpi, fig_u, fig_m, fig_ext, fig_niip_stock, fig_niip_flow = dashboard_1()

    with st.container():
        r1c1, r1c2, r1c3 = st.columns(3, gap="large")
        with r1c1:
            st.plotly_chart(fig_cpi, use_container_width=True)
            st.caption("Significance:Headline CPI captures realised inflation, while CPI-ATE (core) is Norges Bank’s preferred measure for underlying inflation.")
        with r1c2:
            st.plotly_chart(fig_u, use_container_width=True)
            st.caption("Significance: Unemployment is a high-signal proxy for labour-market tightness, rising unemployment typically reduces wage/price pressure and supports a more dovish rates profile.")
        with r1c3:
            st.plotly_chart(fig_m, use_container_width=True)
            st.caption("Significance: Liquidity/credit impulse context—supports rates and risk sentiment read-through.")

        r2c1, r2c2, r2c3 = st.columns(3, gap="large")
        with r2c1:
            st.plotly_chart(fig_ext, use_container_width=True)
            st.caption("Significance: These settlements track cash moving into/out of Norway across industries, offering a timely read on balance-of-payments flow momentum.")
        with r2c2:
            st.plotly_chart(fig_niip_stock, use_container_width=True)
            st.caption("Significance: NIIP stocks show the cumulative result of offshore investing—portfolio assets dominate the external balance sheet, consistent with NBIM’s role.")
        with r2c3:
            st.plotly_chart(fig_niip_flow, use_container_width=True)
            st.caption("Significance: Financial account flows offer a practical read on NBIM/GPFG’s outward investment impulse, proxied by general government portfolio asset outflows.")
with tab2:
    st.subheader("Seasonality")

    (
        fig_cpi_heat,
        fig_u_heat,
        fig_niip_box,
        fig_cpi_bar,
        fig_u_bar,
        fig_niip_table,
        fig_cpi_table,
        fig_u_table,
    ) = dashboard_2_plotly()

    # Make NIIP table fit nicely in its cell
    fig_niip_table.update_layout(
        height=420,
        margin=dict(t=60, b=10, l=10, r=10),
    )

    # Optional: normalize margins so charts line up visually
    fig_cpi_heat.update_layout(margin=dict(t=70, b=60, l=60, r=20))
    fig_u_heat.update_layout(margin=dict(t=70, b=60, l=60, r=20))
    fig_niip_box.update_layout(margin=dict(t=70, b=80, l=60, r=20))
    fig_cpi_bar.update_layout(margin=dict(t=70, b=80, l=60, r=20))
    fig_u_bar.update_layout(margin=dict(t=70, b=80, l=60, r=20))

    # =========================
    # ROW 1 (3 columns)
    # =========================
    r1c1, r1c2, r1c3 = st.columns(3, gap="large")

    with r1c1:
        st.plotly_chart(fig_cpi_heat, use_container_width=True)
        st.caption("CPI MoM heatmap: identifies recurring monthly inflation patterns and regime shifts.")

    with r1c2:
        st.plotly_chart(fig_u_heat, use_container_width=True)
        st.caption("Unemployment heatmap: highlights recurring hiring/layoff seasonality (NSA).")

    with r1c3:
        st.plotly_chart(fig_niip_box, use_container_width=True)
        st.caption("NIIP boxplot: compares quarterly distribution; useful for spotting Q1 skew / tail risk.")

    # =========================
    # ROW 2 (3 columns)
    # =========================
    r2c1, r2c2, r2c3 = st.columns(3, gap="large")

    with r2c1:
        st.plotly_chart(fig_cpi_bar, use_container_width=True)
        st.caption("Avg CPI MoM by month (with window toggles): isolates seasonal impulse vs trend.")

    with r2c2:
        st.plotly_chart(fig_u_bar, use_container_width=True)
        st.caption("Avg unemployment by month (with window toggles): tracks labor seasonality stability.")

    with r2c3:
        st.plotly_chart(fig_niip_table, use_container_width=True)



with tab3:
    st.subheader("Rates & FX")

    figs3, df_fx_wide = dashboard_3()

    # Unpack for clarity
    (
        fig_curve,
        fig_nowa_rv,
        fig_nowa_health,
        fig_usdnok,
        fig_eurnok,
        fig_audnok,
    ) = figs3

    # Row 1: Curve + NOWA rate/volume
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.plotly_chart(fig_curve, use_container_width=True)
        st.caption("Govt curve: tracks repricing across the front-end and belly; useful for Norges Bank path and term premium shifts.")
    with c2:
        st.plotly_chart(fig_nowa_rv, use_container_width=True)
        st.caption("NOWA rate vs volume: flags funding/turnover changes that can precede money-market stress or policy transmission issues.")

    # Row 2: NOWA market health + FX
    c3, c4 = st.columns(2, gap="large")
    with c3:
        st.plotly_chart(fig_nowa_health, use_container_width=True)
        st.caption("NOWA market health: Quick check on NOK overnight funding conditions")
    with c4:
        ccy = st.selectbox("FX pair", ["USD", "EUR", "AUD"], index=0)
        fx_map = {"USD": fig_usdnok, "EUR": fig_eurnok, "AUD": fig_audnok}
        st.plotly_chart(fx_map[ccy], use_container_width=True)
        st.caption("Spot NOK: links funding/curve signals into the traded outcome; range slider + buttons support quick regime checks.")

with tab4:
    st.subheader("FX playground")

    # Ensure df_fx_wide is available
    figs3, df_fx_wide = dashboard_3()

    pair_map = {"USDNOK": "USD", "EURNOK": "EUR", "GBPNOK": "GBP", "AUDNOK": "AUD"}
    pair = st.selectbox("FX pair (seasonality)", list(pair_map.keys()), index=0)
    method = st.radio("Return method", ["pct", "log"], index=0, horizontal=True)

    fig_fx = plot_seasonality_table(
        df_fx_wide=df_fx_wide,
        ccy=pair_map[pair],
        method=method,
        title=f"{pair} Seasonality",
    )

    st.plotly_chart(fig_fx, use_container_width=True)
