import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------- Helper: SSB "YYYYMmm" -> timestamp ----------
def ssb_to_timestamp(x: str) -> pd.Timestamp:
    """
    Convert SSB time key like '2023M01' into a pandas Timestamp (YYYY-MM-01).
    """
    y, m = x.split("M")
    return pd.to_datetime(f"{y}-{m}-01")


# ---------- CPI index loader for seasonality ----------
def load_cpi_idx_for_seasonality():
    """
    Load headline & core CPI indices (2015=100) and compute MoM %.

    Returns
    -------
    df_cpi_idx : DataFrame
        Columns:
          - date
          - headline_index
          - core_index
          - headline_mom  (MoM %)
          - core_mom      (MoM %)
    """
    # --- Headline CPI index (TOTAL, 2015=100) from 03013 ---
    url_headline_idx = (
        "https://data.ssb.no/api/pxwebapi/v2/tables/03013/data?lang=en"
        "&valueCodes[Konsumgrp]=TOTAL"
        "&valueCodes[ContentsCode]=KpiIndMnd"
        "&valueCodes[Tid]=top(120)"
        "&outputformat=json-px"
    )
    resp_headline_idx = requests.get(url_headline_idx)
    resp_headline_idx.raise_for_status()
    j_headline_idx = resp_headline_idx.json()

    headline_rows = []
    for row in j_headline_idx["data"]:
        tid = row["key"][-1]          # e.g. ["TOTAL", "2024M05"] -> "2024M05"
        raw_val = row["values"][0]    # string from SSB, may be "." etc.
        try:
            value = float(raw_val)
        except ValueError:
            # skip non-numeric obs
            continue
        headline_rows.append({"Tid": tid, "headline_index": value})

    df_headline_idx = pd.DataFrame(headline_rows)
    df_headline_idx["date"] = df_headline_idx["Tid"].apply(ssb_to_timestamp)
    df_headline_idx = df_headline_idx[["date", "headline_index"]].sort_values("date")

    # --- Core CPI-ATE index (JAE_TOTAL, 2015=100) from 05327 ---
    url_core_idx = (
        "https://data.ssb.no/api/pxwebapi/v2/tables/05327/data?lang=en"
        "&valueCodes[Konsumgrp]=JAE_TOTAL"
        "&valueCodes[ContentsCode]=KPIJustIndMnd"
        "&valueCodes[Tid]=top(120)"
        "&outputformat=json-px"
    )
    resp_core_idx = requests.get(url_core_idx)
    resp_core_idx.raise_for_status()
    j_core_idx = resp_core_idx.json()

    core_idx_rows = []
    for row in j_core_idx["data"]:
        tid = row["key"][-1]
        raw_val = row["values"][0]
        try:
            value = float(raw_val)
        except ValueError:
            continue
        core_idx_rows.append({"Tid": tid, "core_index": value})

    df_core_idx = pd.DataFrame(core_idx_rows)
    df_core_idx["date"] = df_core_idx["Tid"].apply(ssb_to_timestamp)
    df_core_idx = df_core_idx[["date", "core_index"]].sort_values("date")

    # --- Merge + build MoM % changes ---
    df_cpi_idx = (
        pd.merge(df_headline_idx, df_core_idx, on="date", how="inner")
        .sort_values("date")
    )

    df_cpi_idx["headline_mom"] = df_cpi_idx["headline_index"].pct_change() * 100
    df_cpi_idx["core_mom"] = df_cpi_idx["core_index"].pct_change() * 100

    return df_cpi_idx


# ---------- Unemployment (NSA) loader for seasonality ----------
def load_unemployment_nsa():
    """
    Load unemployment (NSA, 15–74, % of labour force).

    Returns
    -------
    df_u_nsa : DataFrame
        Columns:
          - date
          - unemployment_nsa
    """
    url_u_nsa = (
        "https://data.ssb.no/api/pxwebapi/v2/tables/13760/data?lang=en"
        "&valueCodes[ContentsCode]=ArbledProsArbstyrk"
        "&valueCodes[Kjonn]=0"
        "&valueCodes[Alder]=15-74"
        "&valueCodes[Justering]=IS"      # U = not seasonally adjusted
        "&valueCodes[Tid]=top(200)"
        "&outputformat=json-px"
    )

    resp = requests.get(url_u_nsa)
    resp.raise_for_status()
    j = resp.json()

    cols = [c["code"] for c in j["columns"]]

    rows = []
    for row in j["data"]:
        key_map = dict(zip(cols, row["key"]))
        tid = key_map["Tid"]
        raw_val = row["values"][0]
        try:
            value = float(raw_val)
        except ValueError:
            # skip "." etc
            continue
        rows.append({"Tid": tid, "unemployment_nsa": value})

    df_u_nsa = pd.DataFrame(rows)
    df_u_nsa["date"] = df_u_nsa["Tid"].apply(ssb_to_timestamp)
    df_u_nsa = df_u_nsa[["date", "unemployment_nsa"]].sort_values("date")

    return df_u_nsa



# ---------- Norges Bank color palette ----------
NB_DARK_BLUE   = "#003B5C"   # primary blue (key lines)
NB_MED_BLUE    = "#4F87B2"   # secondary blue
NB_LIGHT_BLUE  = "#A7C6E5"   # light blue

NB_GREEN       = "#2A8C55"   # green (derivatives / positive accents)
NB_RED         = "#B03A2E"   # red (outgoing / negative)
NB_GREY        = "#7F7F7F"   # neutral grey
NB_LIGHT_GREY  = "#E5E5E5"   # gridlines / light borders

# Extra accent colours for NIIP stacks
ACC_ORANGE     = "#E08E0B"   # strong orange / gold
ACC_TEAL       = "#008C95"   # teal (blue-green)
ACC_PURPLE     = "#5A4E8C"   # muted purple (not used yet, spare)



def dashboard_1():
    st.subheader("Norway Macro Dashboard")
    st.caption("Live data from SSB (Statistics Norway)")
    # ========================================================
    # 1) CPI: headline + core (merged)
    # ========================================================

    # --- Core CPI (CPI-ATE, JAE_TOTAL) from 05327 ---
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
        tid = row["key"][-1]  # ["JAE_TOTAL", "2025M01"] -> "2025M01"
        raw_val = row["values"][0] #string from SSB
        #do safety handle ".", "..", or any weird non-numeric value
        try:
            value = float(raw_val)
        except ValueError:
            #if its "." skip this observation
            continue
        
        core_rows.append({"Tid": tid, "core_yoy": value})

    df_core = pd.DataFrame(core_rows)
    df_core["date"] = df_core["Tid"].apply(ssb_to_timestamp)
    df_core = df_core[["date", "core_yoy"]].sort_values("date")

    # --- Headline CPI (all-item, TOTAL) from 03013 ---
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
            continue   # or set value = 0.0 if you prefer
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
            continue   # or set value = 0.0 if you prefer
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
        "&valueCodes[Tid]=top(40)"
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

        # safely pull numeric value
        raw_val = row["values"][0]
        try:
            value = float(raw_val)
        except ValueError:
            # skip rows with ".", "..", or non-numeric
            continue

        records_ext.append({
            "industry": key_map["NACE2007"],
            "period": key_map["Tid"],
            "flow": key_map["Inngaaende2"],   # '01' incoming, '02' outgoing
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

    df_net = pd.DataFrame({
        "Quarter": in_sum.index,
        "incoming_total": in_sum.values,
        "outgoing_total": out_sum.reindex(in_sum.index).values,
    })
    df_net["net"] = df_net["incoming_total"] - df_net["outgoing_total"]
    df_net = df_net.sort_values("Quarter")


    # ========================================================
    # 5) NIIP STOCKS (Beholdning)
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
        side = key_map["FordringGjeld"]          # 'F' = assets, 'G' = liabilities
        func = key_map["FunksjObjSektor"]        # '3.1', '3.2', ...
        quarter = key_map["Tid"]                 # '2022K3'

        raw_val = row["values"][0]
        value = 0.0 if raw_val == "." else float(raw_val)
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
    quarters_stock = df_niip_stock.index.tolist()


    # ========================================================
    # 6) NIIP FLOWS (Transaksjoner)
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
        value = 0.0 if val_raw == "." else float(val_raw)

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

    # Functions (columns) we'll use repeatedly
    func_cols = [
        "Direct investment",
        "Portfolio investment",
        "Other investments",
        "Financial derivatives",
        "Reserve assets",
    ]

    # Net financial account flow across all functions
    df_niip_flow["Total net flow"] = df_niip_flow[func_cols].sum(axis=1)

    quarters_flow = df_niip_flow.index.tolist()


    # ========================================================
    # 7) Build 3x2-subplot Plotly figure
    # ========================================================

    fig = make_subplots(
        rows=2,
        cols=3,
        shared_xaxes=False,
        vertical_spacing=0.06,
        horizontal_spacing=0.06,
        subplot_titles=[
            "CPI Inflation (Headline vs Core)",
            "Unemployment (SA, 15–74)",
            "Monetary Aggregates (M1 / M2 / M3)",
            "External Settlements by Industry",
            "NIIP by Function (Stocks)",
            "NIIP – Financial Account Flows",
        ],
    )

    # --- Row 1, Col 1: CPI ---
    fig.add_trace(
        go.Scatter(
            x=df_cpi["date"],
            y=df_cpi["headline_yoy"],
            mode="lines+markers",
            name="Headline CPI YoY",
            legendgroup="cpi",
            line=dict(color=NB_DARK_BLUE, width=2),
            marker=dict(color=NB_DARK_BLUE, size=6),
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df_cpi["date"],
            y=df_cpi["core_yoy"],
            mode="lines+markers",
            name="Core CPI-ATE YoY",
            legendgroup="cpi",
            line=dict(color=NB_MED_BLUE, width=2, dash="dot"),
            marker=dict(color=NB_MED_BLUE, size=6),
        ),
        row=1, col=1,
    )

    # --- Row 1, Col 2: Unemployment ---
    fig.add_trace(
        go.Scatter(
            x=df_u["date"],
            y=df_u["unemployment_sa"],
            mode="lines+markers",
            name="Unemployment SA",
            legendgroup="labour",
            line=dict(color=NB_DARK_BLUE, width=2),
            marker=dict(color=NB_DARK_BLUE, size=6),
        ),
        row=1, col=2,
    )

    # --- Row 1, Col 3: Monetary aggregates ---
    fig.add_trace(
        go.Scatter(
            x=df_m["date"],
            y=df_m["M1"],
            mode="lines+markers",
            name="M1",
            legendgroup="money",
            line=dict(color=NB_DARK_BLUE, width=2),
            marker=dict(color=NB_DARK_BLUE, size=5),
        ),
        row=1, col=3,
    )

    fig.add_trace(
        go.Scatter(
            x=df_m["date"],
            y=df_m["M2"],
            mode="lines+markers",
            name="M2",
            legendgroup="money",
            line=dict(color=NB_MED_BLUE, width=2),
            marker=dict(color=NB_MED_BLUE, size=5),
        ),
        row=1, col=3,
    )

    fig.add_trace(
        go.Scatter(
            x=df_m["date"],
            y=df_m["M3"],
            mode="lines+markers",
            name="M3",
            legendgroup="money",
            line=dict(color=NB_LIGHT_BLUE, width=2),
            marker=dict(color=NB_LIGHT_BLUE, size=5),
        ),
        row=1, col=3,
    )

    # --- Row 2, Col 1: External settlements ---
    for industry in df_in_long["Industry"].unique():
        df_ind = df_in_long[df_in_long["Industry"] == industry]
        fig.add_trace(
            go.Bar(
                x=df_ind["Quarter"],
                y=df_ind["Value"],
                name=f"Incoming – {industry}",
                legendgroup="ext_in",
                showlegend=False,
                marker=dict(
                    color=NB_MED_BLUE,
                    line=dict(color="white", width=0.2),
                ),
                hovertemplate=(
                    "Quarter=%{x}<br>"
                    "Industry=%{customdata}<br>"
                    "Incoming=%{y:,.0f} NOKm"
                    "<extra></extra>"
                ),
                customdata=df_ind["Industry"],
            ),
            row=2, col=1,
        )

    for industry in df_out_long["Industry"].unique():
        df_ind = df_out_long[df_out_long["Industry"] == industry]
        fig.add_trace(
            go.Bar(
                x=df_ind["Quarter"],
                y=-df_ind["Value"],
                name=f"Outgoing – {industry}",
                legendgroup="ext_out",
                showlegend=False,
                marker=dict(
                    color=NB_RED,
                    line=dict(color="white", width=0.2),
                ),
                customdata=list(zip(df_ind["Industry"], df_ind["Value"])),
                hovertemplate=(
                    "Quarter=%{x}<br>"
                    "Industry=%{customdata[0]}<br>"
                    "Outgoing=%{customdata[1]:,.0f} NOKm"
                    "<extra></extra>"
                ),
            ),
            row=2, col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=df_net["Quarter"],
            y=df_net["net"],
            mode="lines+markers",
            name="Net (Incoming - Outgoing)",
            legendgroup="ext_net",
            line=dict(width=2, color=NB_DARK_BLUE),
            marker=dict(color=NB_DARK_BLUE, size=5),
            hovertemplate="Quarter=%{x}<br>Net flow=%{y:,.0f} NOKm<extra></extra>",
        ),
        row=2, col=1,
    )

    fig.add_trace(
        go.Bar(
            x=[None],
            y=[None],
            name="Incoming (stacked)",
            legendgroup="ext_in",
            showlegend=True,
            marker=dict(color=NB_MED_BLUE),
        ),
        row=2, col=1,
    )

    fig.add_trace(
        go.Bar(
            x=[None],
            y=[None],
            name="Outgoing (stacked)",
            legendgroup="ext_out",
            showlegend=True,
            marker=dict(color=NB_RED),
        ),
        row=2, col=1,
    )

    # --- Row 2, Col 2: NIIP stock (stacked bars + line) ---
    colors_stock = {
        "Direct investment":     NB_DARK_BLUE,  # blue anchor
        "Portfolio investment":  ACC_ORANGE,    # orange
        "Other investments":     ACC_TEAL,      # teal
        "Financial derivatives": NB_GREEN,      # green
        "Reserve assets":        NB_GREY,       # grey
    }

    for func in ["Direct investment", "Portfolio investment",
                "Other investments", "Financial derivatives", "Reserve assets"]:
        fig.add_trace(
            go.Bar(
                x=quarters_stock,
                y=df_niip_stock[func],
                name=func,
                marker=dict(
                    color=colors_stock[func],
                    line=dict(color="white", width=0.2),
                ),
                legendgroup="niip_stock",
            ),
            row=2, col=2,
        )

    fig.add_trace(
        go.Scatter(
            x=quarters_stock,
            y=df_niip_stock["Total NIIP"],
            mode="lines+markers",
            name="Total NIIP",
            line=dict(color=NB_DARK_BLUE, width=2),
            marker=dict(color=NB_DARK_BLUE, size=5),
            legendgroup="niip_stock",
        ),
        row=2, col=2,
    )


    # --- Row 2, Col 3: NIIP flows (stacked bars + net line) ---
    colors_flow = {
        "Direct investment":     NB_DARK_BLUE,
        "Portfolio investment":  ACC_ORANGE,
        "Other investments":     ACC_TEAL,
        "Financial derivatives": NB_GREEN,
        "Reserve assets":        NB_GREY,
    }

    # Stacked bars by function
    for func in func_cols:
        fig.add_trace(
            go.Bar(
                x=quarters_flow,
                y=df_niip_flow[func],
                name=func,
                marker=dict(
                    color=colors_flow[func],
                    line=dict(color="white", width=0.2),
                ),
                legendgroup="niip_flow",
                showlegend=False,  # avoid duplicating same legend labels
            ),
            row=2, col=3,
        )

    # Net total NIIP flow line
    fig.add_trace(
        go.Scatter(
            x=quarters_flow,
            y=df_niip_flow["Total net flow"],
            mode="lines+markers",
            name="Net NIIP flow",
            legendgroup="niip_flow",
            line=dict(color=NB_DARK_BLUE, width=2),
            marker=dict(color=NB_DARK_BLUE, size=5),
            hovertemplate="Quarter=%{x}<br>Net flow=%{y:,.0f} NOKm<extra></extra>",
        ),
        row=2, col=3,
    )

    # One dummy legend entry so stacked bars still show group
    fig.add_trace(
        go.Bar(
            x=[None],
            y=[None],
            name="NIIP flows (stacked by function)",
            legendgroup="niip_flow",
            showlegend=True,
            marker=dict(color=NB_MED_BLUE),
        ),
        row=2, col=3,
    )

    # --- Layout tweaks ---
    fig.update_yaxes(title_text="YoY %", row=1, col=1)
    fig.update_yaxes(title_text="% of labour force", row=1, col=2)
    fig.update_yaxes(title_text="Index / NOK bn (level)", row=1, col=3)
    fig.update_yaxes(title_text="NOK million", row=2, col=1)
    fig.update_yaxes(title_text="Net position (NOK million)", row=2, col=2)
    fig.update_yaxes(title_text="Net flow (NOK million)", row=2, col=3)

    fig.update_layout(
        title="Norway Macro Dashboard",
        barmode="relative",
        height=950,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.18,
            xanchor="center",
            x=0.5,
            font=dict(size=9),
            tracegroupgap=40,
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    # --- Gridlines for the top row only ---
    fig.update_yaxes(
        showgrid=True,
        gridwidth=0.3,
        gridcolor=NB_LIGHT_GREY,
        row=1, col=1
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=0.3,
        gridcolor=NB_LIGHT_GREY,
        row=1, col=2
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=0.3,
        gridcolor=NB_LIGHT_GREY,
        row=1, col=3
    )

    # Turn OFF grids on bottom row only
    fig.update_yaxes(showgrid=False, row=2, col=1)
    fig.update_yaxes(showgrid=False, row=2, col=2)
    fig.update_yaxes(showgrid=False, row=2, col=3)

    # Remove x-axis grids for all
    fig.update_xaxes(showgrid=False)

    st.plotly_chart(fig, width="stretch")

def dashboard_2():
    # ========================================================
    # 1) Load NIIP Transactions from SSB
    # ========================================================
    url = (
        "https://data.ssb.no/api/pxwebapi/v2/tables/10644/data?lang=en"
        "&valueCodes[FordringGjeld]=F,G"
        "&valueCodes[FunksjObjSektor]=3.1,3.2,3.3,3.4,3.5"
        "&valueCodes[ContentsCode]=Transaksj"
        "&valueCodes[Tid]=top(80)"
        "&outputformat=json-px"
    )

    resp = requests.get(url)
    resp.raise_for_status()
    j = resp.json()

    cols = [c["code"] for c in j["columns"]]

    records = []
    for row in j["data"]:
        key_map = dict(zip(cols, row["key"]))
        side = key_map["FordringGjeld"]      # F = assets, G = liabilities
        quarter = key_map["Tid"]             # e.g. "2023K1"

        raw_val = row["values"][0]

        # Convert "..", ".", or any non-numeric entries to 0
        try:
            value = float(raw_val)
        except Exception:
            value = 0.0

        records.append({"quarter": quarter, "side": side, "value": value})

    df = pd.DataFrame(records)

    # ========================================================
    # 2) Aggregate to Assets / Liabilities / Net
    # ========================================================
    df_clean = (
        df.groupby(["quarter", "side"])["value"]
        .sum()
        .unstack("side")
        .fillna(0)
    )

    df_clean = df_clean.rename(columns={"F": "assets", "G": "liabilities"})
    df_clean["net"] = df_clean["assets"] - df_clean["liabilities"]

    # Drop quarters with missing liabilities (safety filter)
    df_clean = df_clean[df_clean["liabilities"] != 0]

    # Filter out very small numbers (treat as 0 if < 100)
    df_clean["assets"] = df_clean["assets"].where(df_clean["assets"] > 100, 0)
    df_clean["liabilities"] = df_clean["liabilities"].where(df_clean["liabilities"] > 100, 0)
    df_clean["net"] = df_clean["assets"] - df_clean["liabilities"]

    # Bring "quarter" out of the index into a column
    df_clean = df_clean.reset_index()  # quarter becomes a column

    # Extract year and quarter-of-year
    df_clean["year"] = df_clean["quarter"].str[:4].astype(int)
    df_clean["Q"] = df_clean["quarter"].str[-1].astype(int)

    # ========================================================
    # 3) NIIP Seasonality Table (Mean / Std by Quarter)
    # ========================================================
    seasonality = (
        df_clean.groupby("Q")["net"]
                .agg(["mean", "std", "count"])
                .sort_index()
    )

    # Prepare nicely labeled table for display
    seasonality_display = (
        seasonality
        .rename(columns={
            "mean": "Mean net flow (NOKm)",
            "std": "Std dev (NOKm)",
            "count": "Obs",
        })
        .reset_index()
        .rename(columns={"Q": "Quarter"})
    )

    nb_dark_blue = "#003B5C"
    nb_light_blue = "#4F87B2"

    # ========================================================
    # A) CPI seasonality objects (build df_cpi_idx here)
    # ========================================================
    df_cpi_idx = load_cpi_idx_for_seasonality()
    df_cpi_idx["year"] = df_cpi_idx["date"].dt.year
    df_cpi_idx["month"] = df_cpi_idx["date"].dt.month

    # CPI heatmap pivot
    heat_cpi_mom = df_cpi_idx.pivot_table(
        index="year", columns="month", values="headline_mom"
    )

    # CPI monthly averages
    seasonal_cpi = (
        df_cpi_idx
        .groupby("month")[["headline_mom", "core_mom"]]
        .mean()
        .rename(columns={
            "headline_mom": "headline_mom_avg",
            "core_mom": "core_mom_avg",
        })
    )

    # CPI table values + conditional colors
    seasonal_cpi_rounded = seasonal_cpi.round(2)
    cpi_months = seasonal_cpi_rounded.index.tolist()
    headline_vals = seasonal_cpi_rounded["headline_mom_avg"].tolist()
    core_vals = seasonal_cpi_rounded["core_mom_avg"].tolist()

    cpi_all_vals = headline_vals + core_vals
    cpi_vmin, cpi_vmax = min(cpi_all_vals), max(cpi_all_vals)

    def cpi_value_to_color(val):
        if cpi_vmax == cpi_vmin:
            z = 0.5
        else:
            z = (val - cpi_vmin) / (cpi_vmax - cpi_vmin)
        r, g, b, _ = plt.get_cmap("coolwarm")(z)
        return f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"

    headline_colors = [cpi_value_to_color(v) for v in headline_vals]
    core_colors = [cpi_value_to_color(v) for v in core_vals]

    cpi_cell_colors = [
        ["white"] * len(cpi_months),  # Month column
        headline_colors,
        core_colors,
    ]

    # ========================================================
    # B) Unemployment seasonality objects (NSA)
    # ========================================================
    df_u_nsa = load_unemployment_nsa()
    df_u_nsa["year"] = df_u_nsa["date"].dt.year
    df_u_nsa["month"] = df_u_nsa["date"].dt.month

    # Unemployment heatmap pivot
    heat_u = df_u_nsa.pivot_table(
        index="year", columns="month", values="unemployment_nsa"
    )

    # Monthly average unemployment
    seasonal_u = (
        df_u_nsa
        .groupby("month")["unemployment_nsa"]
        .mean()
        .rename("unemployment_avg")
    )

    u_rounded = seasonal_u.round(2)
    u_months = u_rounded.index.tolist()
    u_vals = u_rounded.values.tolist()

    u_vmin, u_vmax = min(u_vals), max(u_vals)

    def u_value_to_color(val):
        if u_vmax == u_vmin:
            z = 0.5
        else:
            z = (val - u_vmin) / (u_vmax - u_vmin)
        r, g, b, _ = plt.get_cmap("coolwarm")(z)
        return f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"

    u_colors = [u_value_to_color(v) for v in u_vals]

    u_cell_colors = [
        ["white"] * len(u_months),  # Month column
        u_colors,
    ]

    # ========================================================
    # B) Unemployment seasonality objects (NSA)
    # ========================================================
    df_u_nsa["year"] = df_u_nsa["date"].dt.year
    df_u_nsa["month"] = df_u_nsa["date"].dt.month

    # Unemployment heatmap pivot
    heat_u = df_u_nsa.pivot_table(
        index="year", columns="month", values="unemployment_nsa"
    )

    # Monthly average unemployment
    seasonal_u = (
        df_u_nsa
        .groupby("month")["unemployment_nsa"]
        .mean()
        .rename("unemployment_avg")
    )

    u_rounded = seasonal_u.round(2)
    u_months = u_rounded.index.tolist()
    u_vals = u_rounded.values.tolist()

    u_vmin, u_vmax = min(u_vals), max(u_vals)

    def u_value_to_color(val):
        if u_vmax == u_vmin:
            z = 0.5
        else:
            z = (val - u_vmin) / (u_vmax - u_vmin)
        r, g, b, _ = plt.get_cmap("coolwarm")(z)
        return f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"


    u_colors = [u_value_to_color(v) for v in u_vals]

    u_cell_colors = [
        ["white"] * len(u_months),  # Month column
        u_colors,
    ]

    # ========================================================
    # C) Combined Plotly Subplots – 3 columns x 3 rows
    # ========================================================
    fig = make_subplots(
        rows=3, cols=3,
        specs=[
            # row 1: CPI heatmap, Unemp heatmap, NIIP box
            [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "box"}],
            # row 2: CPI bar, Unemp bar, NIIP table
            [{"type": "xy"},      {"type": "xy"},      {"type": "table"}],
            # row 3: CPI table, Unemp table, empty
            [{"type": "table"},   {"type": "table"},   None],
        ],
        subplot_titles=[
            # Row 1
            "Headline CPI MoM – Seasonality Heatmap",
            "Unemployment rate (NSA) – Seasonality Heatmap",
            "Norway NIIP Seasonality – Quarterly Boxplot",
            # Row 2
            "Average MoM Inflation by Month (Headline vs Core)",
            "Average Unemployment Rate by Month (NSA)",
            "NIIP Net Flow Seasonality Table",
            # Row 3
            "Seasonal CPI MoM Factors",
            "Seasonal Unemployment Pattern by Month (NSA)",
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    # =======================
    # ROW 1
    # =======================

    # (1,1) CPI MoM Heatmap
    fig.add_trace(
        go.Heatmap(
            x=heat_cpi_mom.columns,
            y=heat_cpi_mom.index,
            z=heat_cpi_mom.values,
            colorscale="RdBu",
            reversescale=True,
            zmid=0,
            showscale=False,     # <--- hide colorbar
        ),
        row=1, col=1
    )



    fig.update_xaxes(title_text="Month", row=1, col=1)
    fig.update_yaxes(title_text="Year", row=1, col=1)

    # (1,2) Unemployment Heatmap
    fig.add_trace(
        go.Heatmap(
            x=heat_u.columns,
            y=heat_u.index,
            z=heat_u.values,
            colorscale="RdBu",
            reversescale=True,
            zmid=heat_u.stack().mean(),
            showscale=False,     # <--- hide colorbar
        ),
        row=1, col=2
    )



    fig.update_xaxes(title_text="Month", row=1, col=2)
    fig.update_yaxes(title_text="Year", row=1, col=2)

    # (1,3) NIIP Boxplot
    fig.add_trace(
        go.Box(
            x=df_clean["Q"],      # 1–4
            y=df_clean["net"],
            name="Net NIIP Flow",
            marker_color=nb_light_blue,
            line_color=nb_dark_blue,
            boxpoints="outliers",
        ),
        row=1, col=3
    )
    fig.update_xaxes(title_text="Quarter", row=1, col=3)
    fig.update_yaxes(title_text="Net NIIP Flow (NOKm)", row=1, col=3)

    # =======================
    # ROW 2
    # =======================

    # (2,1) CPI Seasonal Bar Chart
    cpi_months_for_bar = seasonal_cpi.index.tolist()
    fig.add_trace(
        go.Bar(
            x=cpi_months_for_bar,
            y=seasonal_cpi["headline_mom_avg"],
            name="Headline MoM avg",
            marker_color=nb_dark_blue,
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Bar(
            x=cpi_months_for_bar,
            y=seasonal_cpi["core_mom_avg"],
            name="Core MoM avg",
            marker_color=nb_light_blue,
        ),
        row=2, col=1
    )
    fig.update_xaxes(title_text="Month", row=2, col=1)
    fig.update_yaxes(title_text="Average MoM %", row=2, col=1)

    # (2,2) Unemployment Seasonal Bar Chart
    u_months_for_bar = seasonal_u.index.tolist()
    fig.add_trace(
        go.Bar(
            x=u_months_for_bar,
            y=seasonal_u.values,
            name="Avg unemployment (NSA)",
            marker_color=nb_dark_blue,
        ),
        row=2, col=2
    )
    fig.update_xaxes(title_text="Month", row=2, col=2)
    fig.update_yaxes(title_text="Unemployment %", row=2, col=2)

    # (2,3) NIIP Seasonality Table
    fig.add_trace(
        go.Table(
            header=dict(
                values=list(seasonality_display.columns),
                fill_color=nb_dark_blue,
                align="center",
                font=dict(color="white", size=12),
                height=32,
            ),
            cells=dict(
                values=[seasonality_display[col] for col in seasonality_display.columns],
                align="center",
                fill_color="white",
                font=dict(color="black", size=11),
                height=28,
            ),
        ),
        row=2, col=3
    )

    # =======================
    # ROW 3
    # =======================

    # (3,1) CPI Conditional Table
    fig.add_trace(
        go.Table(
            header=dict(
                values=["Month", "Headline MoM Avg (%)", "Core MoM Avg (%)"],
                fill_color=nb_dark_blue,
                align="center",
                font=dict(color="white", size=12),
                height=32,
            ),
            cells=dict(
                values=[cpi_months, headline_vals, core_vals],
                fill_color=cpi_cell_colors,
                align="center",
                height=28,
                font=dict(size=12),
            ),
        ),
        row=3, col=1
    )

    # (3,2) Unemployment Table (conditional colors on unemployment)
    fig.add_trace(
        go.Table(
            header=dict(
                values=["Month", "Unemployment Avg (%)"],
                fill_color=nb_dark_blue,
                align="center",
                font=dict(color="white", size=12),
                height=32,
            ),
            cells=dict(
                values=[u_months, u_vals],
                fill_color=u_cell_colors,
                align="center",
                height=28,
                font=dict(size=12),
            ),
        ),
        row=3, col=2
    )

    # (3,3) is None/empty on purpose (no NIIP chart there)

    # =======================
    # Layout & legend
    # =======================
    fig.update_layout(
        title="Norway Macro Seasonality Dashboard",
        height=1100,
        width=1400,
        barmode="group",
        showlegend=True,
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.08,
            yanchor="top",
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        margin=dict(t=80, l=120, r=60, b=140),  # l=120 so bars aren’t cut off
    )

    st.plotly_chart(fig, width="stretch")

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

    def load_nok_fx():
        """
        Pull USDNOK, EURNOK, AUDNOK spot rates from
        Norges Bank EXR dataset (last 360 obs).

        Returns:
            df_long  - tidy format: date | currency | value
            df_wide  - pivoted: date | USD | EUR | AUD
        """

        url = (
            "https://data.norges-bank.no/api/data/"
            "EXR/B.USD+EUR+AUD.NOK.SP?"
            "format=sdmx-json&lastNObservations=360&locale=en"
        )

        resp = requests.get(url)
        resp.raise_for_status()
        j = resp.json()

        # ---- Time dimension ----
        obs_dims = j["data"]["structure"]["dimensions"]["observation"]
        time_vals = next(d["values"] for d in obs_dims if d["id"] == "TIME_PERIOD")
        dates = [pd.to_datetime(v["id"]) for v in time_vals]

        # ---- Base currency dimension (USD/EUR/AUD) ----
        series_dims = j["data"]["structure"]["dimensions"]["series"]
        base_pos = next(
            i for i, d in enumerate(series_dims)
            if d["id"] in ("BASE_CUR", "CURRENCY")
        )
        base_currencies = [v["id"] for v in series_dims[base_pos]["values"]]

        # ---- Observations ----
        records = []
        series_dict = j["data"]["dataSets"][0]["series"]

        for key, entry in series_dict.items():
            idxs = list(map(int, key.split(":")))
            base_cur = base_currencies[idxs[base_pos]]   # 'USD', 'EUR', 'AUD'

            for obs_idx_str, val_arr in entry["observations"].items():
                obs_idx = int(obs_idx_str)
                value = float(val_arr[0])

                records.append({
                    "date": dates[obs_idx],
                    "currency": base_cur,
                    "value": value,
                })

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
        Simple 2-axis chart: NOWA rate (line) vs overnight volume (bar).
        Uses full history in df_nowa.
        """
        fig = make_subplots(
            rows=1,
            cols=1,
            specs=[[{"secondary_y": True}]],
        )

        # Rate
        fig.add_trace(
            go.Scatter(
                x=df_nowa["date"],
                y=df_nowa["R"],
                mode="lines",
                name="NOWA rate",
                line=dict(width=2, color="#003B5C"),
            ),
            row=1, col=1, secondary_y=False,
        )

        # Volume
        fig.add_trace(
            go.Bar(
                x=df_nowa["date"],
                y=df_nowa["V"],
                name="Overnight volume",
                opacity=0.6,
                marker=dict(color="#4F87B2"),
            ),
            row=1, col=1, secondary_y=True,
        )

        fig.update_layout(
            title="NOWA Overnight – Rate vs Volume",
            xaxis_title="Date",
            yaxis_title="Rate (%)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(0,0,0,0)",
            ),
            template="simple_white",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=60, r=60, t=70, b=40),
        )

        fig.update_yaxes(
            title_text="Rate (%)",
            secondary_y=False,
            showgrid=True,
            gridcolor="rgba(200,200,200,0.4)",
        )

        fig.update_yaxes(
            title_text="Volume (NOK mn)",
            secondary_y=True,
            showgrid=False,
        )

        fig.update_xaxes(
            type="date",
            tickformat="%b\n%Y",   # 'Sep\n2025'
            dtick="M1",
            showgrid=True,
            gridcolor="rgba(200,200,200,0.4)",
        )

        return fig

    def build_nowa_market_health_fig(df_nowa):
        """
        3-row stacked chart focusing on the last 360 days:
        Row 1: Volume + 5d MA
        Row 2: Transactions, Banks lending, Banks borrowing
        Row 3: Average deal size (V/T)
        """
        df = df_nowa.copy().sort_values("date")
        cutoff = df["date"].max() - pd.Timedelta(days=360)
        df = df[df["date"] >= cutoff].reset_index(drop=True)

        df["avg_deal_size"] = df["V"] / df["T"]
        df["V_ma5"] = df["V"].rolling(5).mean()

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.45, 0.30, 0.25],
        )

        # Row 1: Volume + MA
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

        # Row 2: T, BL, BB
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

        # Row 3: Average deal size
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

        # Axes
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
            title="NOWA Market Health (last 360 days)",
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

        return fig

    def build_fx_line_fig(df_fx_wide, ccy="USD"):
        """
        Simple spot line chart for one FX cross: USDNOK, EURNOK or AUDNOK.
        ccy: 'USD', 'EUR' or 'AUD'
        """
        if ccy not in df_fx_wide.columns:
            raise ValueError(f"{ccy} not in FX dataframe columns: {df_fx_wide.columns.tolist()}")

        df = df_fx_wide[[ccy]].dropna().reset_index().rename(columns={"index": "date"})

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df[ccy],
                mode="lines",
                name=f"{ccy}NOK",
                line=dict(width=2, color="#003B5C"),
                hovertemplate="%{x|%d %b %Y}<br>" + f"{ccy}NOK: " + "%{y:.3f}<extra></extra>",
            )
        )

        fig.update_layout(
            title=f"{ccy}NOK – Spot rate",
            xaxis_title="Date",
            yaxis_title=f"{ccy}NOK",
            template="simple_white",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=60, r=40, t=60, b=40),
        )

        fig.update_xaxes(
            type="date",
            tickformat="%b\n%Y",
            dtick="M1",
            showgrid=True,
            gridcolor="rgba(200,200,200,0.4)",
        )

        fig.update_yaxes(
            showgrid=True,
            gridcolor="rgba(200,200,200,0.4)",
        )

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
    fig_nowa_health = build_nowa_market_health_fig(df_nowa)

    fig_usdnok = build_fx_line_fig(df_fx_wide, ccy="USD")
    fig_eurnok = build_fx_line_fig(df_fx_wide, ccy="EUR")
    fig_audnok = build_fx_line_fig(df_fx_wide, ccy="AUD")

    # --- 2x3 combined grid ---
    fig_grid = make_subplots(
        rows=2,
        cols=3,
        specs=[
            [{}, {"secondary_y": True}, {}],
            [{}, {}, {}],
        ],
        subplot_titles=[
            "Govt curve (TBIL + GBON)",
            "NOWA – Rate vs Volume",
            "NOWA – Market Health (last 360d)",
            "USDNOK spot",
            "EURNOK spot",
            "AUDNOK spot",
        ],
        horizontal_spacing=0.06,
        vertical_spacing=0.15,
    )

    # Row 1, Col 1: govt curve
    for trace in fig_curve.data:
        fig_grid.add_trace(trace, row=1, col=1)

    # Row 1, Col 2: NOWA rate (primary y) + volume (secondary y)
    fig_grid.add_trace(fig_nowa_rv.data[0], row=1, col=2, secondary_y=False)
    fig_grid.add_trace(fig_nowa_rv.data[1], row=1, col=2, secondary_y=True)
    fig_grid.update_yaxes(title_text="Rate (%)", row=1, col=2, secondary_y=False)
    fig_grid.update_yaxes(title_text="Volume (NOK mn)", row=1, col=2, secondary_y=True)

    # Row 1, Col 3: NOWA market health
    for trace in fig_nowa_health.data:
        fig_grid.add_trace(trace, row=1, col=3)

    # Row 2: FX crosses
    for trace in fig_usdnok.data:
        fig_grid.add_trace(trace, row=2, col=1)

    for trace in fig_eurnok.data:
        fig_grid.add_trace(trace, row=2, col=2)

    for trace in fig_audnok.data:
        fig_grid.add_trace(trace, row=2, col=3)

    fig_grid.update_layout(
        height=900,
        width=1400,
        showlegend=False,
        template="simple_white",
        title_text="NOWA & NOK Dashboard",
    )

    st.plotly_chart(fig_grid, width="stretch")




st.set_page_config(page_title="Norway Macro Dashboard Suite", layout="wide")

st.title("Norway Macro Dashboard Suite")
st.caption("Live data from Statistics Norway (SSB) and Norges Bank APIs")

tab1, tab2, tab3 = st.tabs(["Macro Overview", "Macro Seasonality", "Norway Money Markets, FX & Rates"])

with tab1:
    dashboard_1()

with tab2:
    dashboard_2()

with tab3:
    dashboard_3()
