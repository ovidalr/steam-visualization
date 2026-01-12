import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# =========================
# Config
# =========================
INPUT_PATH = "games_cleaned.xlsx"
OUTPUT_HTML = "index.html"

MIN_YEAR = 2005
MAX_YEAR = 2024

# Visibilitat per llindars
THRESHOLDS = [10, 100, 1000]
DEFAULT_THRESHOLD = 100
MIN_GAMES_PER_CELL = 30  # amaguem cel·les amb poca mostra

# Recomanacions
REC_THRESHOLD = 100

# Perfils (4 categories), coherent i amb mostra gran
PROFILE_ORDER = [
    "Single-player pur",
    "Experiència híbrida",
    "Multiplayer",
    "Altres / No definit",
]

# Owners buckets (rang discret -> usem buckets)
OWNER_BINS = [0, 20000, 50000, 100000, 500000, 10_000_000_000]
OWNER_LABELS = ["0–20k", "20k–50k", "50k–100k", "100k–500k", ">500k"]

# Preu
PRICE_BUCKETS = ["Free-to-play", "0–10€", "10–30€", ">30€"]


# =========================
# Load
# =========================
df = pd.read_excel(INPUT_PATH, sheet_name=0, engine="openpyxl")


# =========================
# Feature engineering
# =========================
df["Release date"] = pd.to_datetime(df["Release date"], errors="coerce")
df["release_year"] = df["Release date"].dt.year

# Reviews totals (variable calculada)
df["Positive"] = pd.to_numeric(df.get("Positive", 0), errors="coerce").fillna(0)
df["Negative"] = pd.to_numeric(df.get("Negative", 0), errors="coerce").fillna(0)
df["reviews_total"] = df["Positive"] + df["Negative"]

# Percent positive
df["Percent positive reviews"] = pd.to_numeric(df.get("Percent positive reviews", np.nan), errors="coerce")

# Recomanacions
df["Recommendations"] = pd.to_numeric(df.get("Recommendations", 0), errors="coerce").fillna(0)

# Preu final
df["Final Price"] = pd.to_numeric(df.get("Final Price", 0), errors="coerce").fillna(0)

def price_bucket(v):
    if pd.isna(v):
        return np.nan
    if v <= 0:
        return "Free-to-play"
    if v <= 10:
        return "0–10€"
    if v <= 30:
        return "10–30€"
    return ">30€"

df["price_bucket"] = df["Final Price"].apply(price_bucket)
df["price_bucket"] = pd.Categorical(df["price_bucket"], categories=PRICE_BUCKETS, ordered=True)


# Tractament del booleans
def to_bool(s: pd.Series) -> pd.Series:
    return (
        s.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

df["is_single"] = to_bool(df["Is single-player?"])
df["is_multi"]  = to_bool(df["Is multi-player?"])
df["is_coop"]   = to_bool(df["Has Cooperative?"])

# Tractament dels perfils:
# - Single pur: només single (ni multi ni coop)
# - Experiència híbrida: single + (multi o coop)
# - Multiplayer: qualsevol cosa sense single però amb multi o coop
# - Altres/no definit: cap flag
def play_profile(row) -> str:
    s, m, c = row["is_single"], row["is_multi"], row["is_coop"]
    if s and (not m) and (not c):
        return "Single-player pur"
    if s and (m or c):
        return "Experiència híbrida"
    if (not s) and (m or c):
        return "Multiplayer"
    return "Altres / No definit"

df["play_profile"] = df.apply(play_profile, axis=1)
df["play_profile"] = pd.Categorical(df["play_profile"], categories=PROFILE_ORDER, ordered=True)

# Owners midpoint (format "a - b")
def owners_midpoint(s):
    if pd.isna(s):
        return np.nan
    s = str(s).replace(",", "").strip()
    if "-" in s:
        a, b = s.split("-")
        try:
            a = float(a.strip())
            b = float(b.strip())
            return (a + b) / 2
        except Exception:
            return np.nan
    return np.nan

df["owners_est"] = df["Estimated owners"].apply(owners_midpoint) if "Estimated owners" in df.columns else np.nan


# =========================
# Filtre per any (temporals)
# =========================
d = df[df["release_year"].notna()].copy()
d = d[(d["release_year"] >= MIN_YEAR) & (d["release_year"] <= MAX_YEAR)]
years = sorted(d["release_year"].unique())


# =========================
# ACTE 1 — Context: volum de llançaments (STACKED AREA)
# =========================
count_pivot = (
    d.groupby(["release_year", "play_profile"])
    .size()
    .reset_index(name="n_games")
    .pivot(index="play_profile", columns="release_year", values="n_games")
    .reindex(PROFILE_ORDER)
    .reindex(columns=years)
    .fillna(0)
)

fig_volume = go.Figure()
for prof in PROFILE_ORDER:
    fig_volume.add_trace(
        go.Scatter(
            x=years,
            y=count_pivot.loc[prof].values,
            mode="lines",
            name=prof,
            stackgroup="one",
            hovertemplate=f"Any: %{{x}}<br>Perfil: {prof}<br>N llançaments: %{{y:.0f}}<extra></extra>",
        )
    )

fig_volume.update_layout(
    title=dict(text="Volum de llançaments per perfil de joc", x=0.02),
    xaxis_title="Any",
    yaxis_title="Nº de jocs",
    margin=dict(l=90, r=30, t=70, b=60),
    height=440,
)


# =========================
# ACTE 2 — Visibilitat: heatmap per llindars
# =========================
rows = []
for t in THRESHOLDS:
    tmp = (
        d.groupby(["release_year", "play_profile"])["reviews_total"]
        .agg(
            n_games="size",
            pct_visible=lambda x: (x >= t).mean(),
        )
        .reset_index()
    )
    tmp["threshold"] = t
    rows.append(tmp)

agg_vis = pd.concat(rows, ignore_index=True)
agg_vis.loc[agg_vis["n_games"] < MIN_GAMES_PER_CELL, "pct_visible"] = np.nan

def pivot_for_threshold(t: int) -> pd.DataFrame:
    sub = agg_vis[agg_vis["threshold"] == t]
    piv = sub.pivot(index="play_profile", columns="release_year", values="pct_visible")
    return piv.reindex(PROFILE_ORDER).reindex(columns=years)

heat_by_t = {t: pivot_for_threshold(t) for t in THRESHOLDS}

t0 = DEFAULT_THRESHOLD if DEFAULT_THRESHOLD in THRESHOLDS else THRESHOLDS[0]
z0 = heat_by_t[t0].values

# customdata amb mida mostra (n jocs)
n_pivot = (
    d.groupby(["release_year", "play_profile"])
    .size()
    .reset_index(name="n")
    .pivot(index="play_profile", columns="release_year", values="n")
    .reindex(PROFILE_ORDER)
    .reindex(columns=years)
    .fillna(0)
)

fig_visibility = go.Figure()
fig_visibility.add_trace(
    go.Heatmap(
        z=z0,
        x=years,
        y=PROFILE_ORDER,
        zmin=0,
        zmax=1,
        colorbar=dict(title="Proporció visible"),
        customdata=n_pivot.values,
        hovertemplate=(
            "Any: %{x}<br>"
            "Perfil: %{y}<br>"
            "Proporció visible: %{z:.2f}<br>"
            "N jocs (any, perfil): %{customdata:.0f}"
            "<extra></extra>"
        ),
    )
)

buttons = []
for t in THRESHOLDS:
    z = heat_by_t[t].values
    buttons.append(
        dict(
            label=f"Llindar: ≥ {t} reviews",
            method="update",
            args=[
                {"z": [z]},
                {"title": f"Acte 2 — Visibilitat: probabilitat d'arribar a ≥{t} reviews"},
            ],
        )
    )

# Marge superior + posició menú perquè NO se solapi amb títol
fig_visibility.update_layout(
    title=dict(text=f"Probabilitat d'arribar a ≥{t0} reviews", x=0.02, y=0.97),
    xaxis_title="Any",
    yaxis_title="Perfil de joc",
    margin=dict(l=90, r=30, t=130, b=140),
    height=560,
    updatemenus=[
        dict(
            type="dropdown",
            x=0.02,
            y=1.14,
            xanchor="left",
            yanchor="top",
            showactive=True,
            buttons=buttons,
        )
    ],
)

# Nota a sota
fig_visibility.add_annotation(
    text=(
        f"Nota: cel·les amb menys de {MIN_GAMES_PER_CELL} jocs es mostren buides per evitar soroll. "
        f"Anàlisi: {MIN_YEAR}–{MAX_YEAR}."
    ),
    xref="paper",
    yref="paper",
    x=0,
    y=-0.24,
    showarrow=False,
    align="left",
    font=dict(size=12),
)


# =========================
# Epileg — Preu: baseline vs jocs molt recomanats (lift)
# =========================
base_price = (
    df["price_bucket"]
    .value_counts(normalize=True)
    .reindex(PRICE_BUCKETS)
    .fillna(0)
)

df_price_rec = df[df["Recommendations"] >= REC_THRESHOLD].copy()

rec_price = (
    df_price_rec["price_bucket"]
    .value_counts(normalize=True)
    .reindex(PRICE_BUCKETS)
    .fillna(0)
)

price_comp = pd.DataFrame({"baseline": base_price, "recommended": rec_price})
price_comp["lift"] = np.where(price_comp["baseline"] > 0, price_comp["recommended"] / price_comp["baseline"], np.nan)

fig_price = go.Figure()
fig_price.add_trace(
    go.Bar(
        x=price_comp.index.astype(str),
        y=price_comp["baseline"],
        name="Mercat global",
        opacity=0.75,
        hovertemplate="Preu: %{x}<br>Proporció mercat: %{y:.1%}<extra></extra>",
    )
)
fig_price.add_trace(
    go.Bar(
        x=price_comp.index.astype(str),
        y=price_comp["recommended"],
        name=f"Jocs amb ≥{REC_THRESHOLD} recomanacions",
        hovertemplate="Preu: %{x}<br>Proporció recomanats: %{y:.1%}<extra></extra>",
    )
)

for idx, row in price_comp.iterrows():
    if row["baseline"] > 0:
        fig_price.add_annotation(
            x=str(idx),
            y=max(row["baseline"], row["recommended"]) + 0.02,
            text=f"x{row['lift']:.2f}",
            showarrow=False,
            font=dict(size=12),
        )

fig_price.update_layout(
    title=dict(text="Preu i recomanacions (baseline vs molt recomanats)", x=0.02),
    yaxis=dict(title="Proporció de jocs", tickformat=".0%"),
    xaxis_title="Preu del joc",
    barmode="group",
    height=440,
    margin=dict(l=90, r=30, t=70, b=90),
    legend=dict(orientation="h", y=-0.25),
)


# =========================
# ACTE 3 — Qualitat percebuda: mediana % positives
# =========================
q = d.copy()
q = q[(q["reviews_total"] >= 100) & (q["Percent positive reviews"].notna())]

qual = (
    q.groupby(["release_year", "play_profile"])["Percent positive reviews"]
    .median()
    .reset_index(name="median_pct_pos")
)

fig_quality = go.Figure()
for prof in PROFILE_ORDER:
    tmp = qual[qual["play_profile"] == prof]
    fig_quality.add_trace(
        go.Scatter(
            x=tmp["release_year"],
            y=tmp["median_pct_pos"],
            mode="lines+markers",
            name=prof,
            hovertemplate=f"Any: %{{x}}<br>Perfil: {prof}<br>Mediana % positives: %{{y:.1f}}<extra></extra>",
        )
    )

fig_quality.update_layout(
    title=dict(text="Mediana de % positives (només jocs amb ≥100 reviews)", x=0.02),
    xaxis_title="Any",
    yaxis_title="% positives (mediana)",
    margin=dict(l=90, r=30, t=70, b=60),
    height=440,
)


# =========================
# Export narratives HTML 
# =========================
html_volume = pio.to_html(fig_volume, include_plotlyjs="cdn", full_html=False)
html_visibility = pio.to_html(fig_visibility, include_plotlyjs=False, full_html=False)
html_price = pio.to_html(fig_price, include_plotlyjs=False, full_html=False)
html_quality = pio.to_html(fig_quality, include_plotlyjs=False, full_html=False)

page = f"""
<!doctype html>
<html lang="ca">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Steam: mercat d'atenció — visibilitat i qualitat</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 0; background: #fff; color:#111; }}
    .wrap {{ max-width: 1150px; margin: 0 auto; padding: 28px 18px 70px; }}
    h1 {{ font-size: 30px; margin: 0 0 10px; }}
    p  {{ line-height: 1.55; margin: 8px 0 16px; color:#333; }}
    .card {{ background: #fff; border: 1px solid #e6e6e6; border-radius: 14px; padding: 14px; box-shadow: 0 1px 6px rgba(0,0,0,.06); margin: 18px 0; }}
    .note {{ font-size: 13px; color:#555; margin-top: -6px; }}
    h2 {{ margin: 0 0 8px; font-size: 18px; }}
    .act {{ margin-top: 26px; font-weight: 800; font-size: 16px; color:#111; }}
    .divider {{ height: 1px; background:#eee; margin: 18px 0; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Steam com a mercat d'atenció</h1>
    <p>
      Steam és avui un dels principals mercats de distribució de videojocs, però també és un mercat cada cop més saturat. 
      En el context actual, publicar un joc no garanteix ni visibilitat ni molt menys èxit.
      En aquest projecte realitzem l'analisi de la pressió competitiva (llançaments), la probabilitat de ser vist (reviews), i què passa quan un joc és vist
      (qualitat percebuda). Tanquem amb un epíleg sobre el rol del preu en l’engagement (recomanacions).
    </p>
 
    <div class="divider"></div>
    <div class="act">ACTE 1 — Context</div>
    <div class="card">
      <h2>Quants jocs es publiquen cada any?</h2>
      <p class="note">Des de 2013 la tendencia és clara, la producció creix a marxes forçades. Però la distribució pel que fa a perfils de jocs no és uniforme.
      Aquesta evolució, genera una pressió competitiva enorme: cada any hi ha més jocs lluitant entre ells per l'atenció dels usuaris.</p>
      {html_volume}
    </div>

    <div class="divider"></div>
    <div class="act">ACTE 2 — Visibilitat</div>
    <div class="card">
      <h2>Quina probabilitat té un joc de ser vist realment?</h2>
      <p class="note">Modificant el llindar de reviews(≥10/100/1000) veiem que la visivilitat cau en el temps. 
      Cada cop hi ha menys percentatge de jocs que arrivin a ternir impacte en el mercat.</p>
      {html_visibility}
    </div>

    <div class="divider"></div>
    <div class="act">ACTE 3 — Qualitat percebuda</div>
    <div class="card">
      <h2>Quan un joc és vist, és un joc de qualitat? Visibilitat = qualitat?</h2>
      <p class="note">Si mirem a partir de 2013, veiem com son justament els jocs single-player els que mantenen la qualitat percebuda.</p>
      {html_quality}
    </div>

    <div class="divider"></div>
    <div class="act">EPÍLEG — Estratègia de mercat</div>
    <div class="card">
      <h2>Hi ha variables que generin compromís dels jugadors?</h2>
      <p class="note">El preu actua com a senyal: comparem la distribució del mercat amb la dels jocs que acumulen recomanacions.
      El sector d’entre 10 i 30€ acumula el 40 % de jocs més recomanats amb només un 17% de la oferta total</p>
      {html_price}
    </div>

  </div>
</body>
</html>
"""



page += """
  </div>
</body>
</html>
"""

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(page)

print(f"[OK] Created {OUTPUT_HTML}")

