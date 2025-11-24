import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------------------------------------------------------
# Streamlit page config
# -------------------------------------------------------------------
st.set_page_config(page_title="Norway Macro Dashboard", layout="wide")
st.title("Norway Macro Dashboard")
st.caption("Live data from SSB (Statistics Norway)")

# ---------- Helper: SSB "YYYYMmm" -> timestamp ----------
def ssb_to_timestamp(x: str) -> pd.Timestamp:
    y, m = x.split("M")
    return pd.to_datetime(f"{y}-{m}-01")


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
    value = float(row["values"][0])
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
    value = float(row["values"][0])
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
    value = float(row["values"][0])
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
    records_ext.append({
        "industry": key_map["NACE2007"],
        "period": key_map["Tid"],
        "flow": key_map["Inngaaende2"],   # '01' incoming, '02' outgoing
        "value": float(row["values"][0]),
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
        line=dict(width=2),
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
    ),
    row=2, col=1,
)

# --- Row 2, Col 2: NIIP stock (stacked bars + line) ---
colors_stock = {
    "Direct investment": "#f2b233",
    "Portfolio investment": "#33aaff",
    "Other investments": "#cc3333",
    "Financial derivatives": "#66cc66",
    "Reserve assets": "#666666",
}

for func in ["Direct investment", "Portfolio investment",
             "Other investments", "Financial derivatives", "Reserve assets"]:
    fig.add_trace(
        go.Bar(
            x=quarters_stock,
            y=df_niip_stock[func],
            name=func,
            marker=dict(color=colors_stock[func]),
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
        line=dict(color="black", width=2),
        legendgroup="niip_stock",
    ),
    row=2, col=2,
)

# --- Row 2, Col 3: NIIP flows (stacked bars + net line) ---
colors_flow = {
    "Direct investment": "#FFC000",
    "Portfolio investment": "#4F81BD",
    "Other investments": "#C0504D",
    "Financial derivatives": "#9BBB59",
    "Reserve assets": "#7F7F7F",
}

# Stacked bars by function
for func in func_cols:
    fig.add_trace(
        go.Bar(
            x=quarters_flow,
            y=df_niip_flow[func],
            name=func,
            marker_color=colors_flow[func],
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
        line=dict(color="black", width=2),
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
    gridcolor="lightgrey",
    row=1, col=1
)
fig.update_yaxes(
    showgrid=True,
    gridwidth=0.3,
    gridcolor="lightgrey",
    row=1, col=2
)
fig.update_yaxes(
    showgrid=True,
    gridwidth=0.3,
    gridcolor="lightgrey",
    row=1, col=3
)

# Turn OFF grids on bottom row only
fig.update_yaxes(showgrid=False, row=2, col=1)
fig.update_yaxes(showgrid=False, row=2, col=2)
fig.update_yaxes(showgrid=False, row=2, col=3)

# And only remove x-axis grids if you want
fig.update_xaxes(showgrid=False)

# -------------------------------------------------------------------
# Streamlit render
# -------------------------------------------------------------------
st.plotly_chart(fig, use_container_width=True)
