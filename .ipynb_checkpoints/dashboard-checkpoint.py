"""
Valorant VCT Analytics — Streamlit App
Unifica: Perfil do Jogador | Análises Gráficas | Predição ML

streamlit run dashboard.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

# ── Configuração da página ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Valorant VCT Analytics",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS / Tema ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* fundo geral */
  .stApp { background-color: #0f0f1a; color: #e0e0e0; }
  section[data-testid="stSidebar"] { background-color: #13132b; }
  /* cabeçalho das seções */
  h1, h2, h3 { color: #ff4655; }
  /* métricas */
  div[data-testid="metric-container"] {
      background: #1a1a2e; border-radius: 8px; padding: 8px 12px;
  }
  /* tabelas */
  .dataframe thead tr th { background-color: #1a1a2e !important; color: #ff4655 !important; }
  .dataframe tbody tr:nth-child(even) { background-color: #161625 !important; }
  /* inputs */
  .stSelectbox label, .stNumberInput label, .stTextInput label { color: #aaa; }
  div[data-baseweb="select"] { background-color: #1a1a2e; }
</style>
""", unsafe_allow_html=True)

# ── Constantes ─────────────────────────────────────────────────────────────────
AGENTES_VALORANT = {
    "duelistas":     ["Jett", "Phoenix", "Reyna", "Raze", "Yoru", "Neon", "Iso", "Waylay"],
    "controladores": ["Brimstone", "Viper", "Omen", "Astra", "Harbor", "Clove", "Miks"],
    "iniciadores":   ["Sova", "Breach", "Skye", "Kayo", "Fade", "Gekko", "Tejo"],
    "sentinelas":    ["Sage", "Cypher", "Killjoy", "Chamber", "Deadlock", "Vyse", "Veto"],
}
TODOS_AGENTES = sorted([a for lst in AGENTES_VALORANT.values() for a in lst])
AGENTE_PARA_ROLE = {a: r for r, lst in AGENTES_VALORANT.items() for a in lst}

COLUNAS_STATS = ['R', 'ACS', 'K', 'D', 'A', '+/-', 'KAST', 'ADR', 'HS%', 'FK', 'FD', '+/-_FK_FD']
PESO_ROLE = {
    'duelistas':     {'R':1.20,'ACS':1.20,'K':1.15,'D':1.00,'A':1.05,'+/-':1.00,'KAST':1.10,'ADR':1.15,'HS%':1.00,'FK':1.15,'FD':1.00,'+/-_FK_FD':1.00},
    'iniciadores':   {'R':1.15,'ACS':1.05,'K':1.10,'D':1.00,'A':1.20,'+/-':1.00,'KAST':1.25,'ADR':1.20,'HS%':1.00,'FK':1.05,'FD':1.00,'+/-_FK_FD':1.00},
    'controladores': {'R':1.15,'ACS':1.05,'K':1.10,'D':1.00,'A':1.20,'+/-':1.00,'KAST':1.25,'ADR':1.25,'HS%':1.00,'FK':1.00,'FD':1.00,'+/-_FK_FD':1.00},
    'sentinelas':    {'R':1.15,'ACS':1.05,'K':1.10,'D':1.00,'A':1.15,'+/-':1.00,'KAST':1.30,'ADR':1.20,'HS%':1.00,'FK':1.05,'FD':1.00,'+/-_FK_FD':1.00},
}

BG  = "#1a1a2e"
FG  = "white"
RED = "#ff4655"

# ══════════════════════════════════════════════════════════════════════════════
# CARREGAMENTO E PRÉ-PROCESSAMENTO
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Carregando dados…")
def carregar_dados(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # limpa colunas extras
    drop_cols = ['Unnamed: 0', 'Time1', 'Mapas_T1', 'Score_T1',
                 'Time2', 'Mapas_T2', 'Score_T2']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df = df.drop_duplicates().reset_index(drop=True)

    # Role
    df['Role'] = df['Agente'].map(AGENTE_PARA_ROLE)

    # Win rates
    df['win_rate_jogador']        = df.groupby('Jogador')['target'].transform('mean')
    df['win_rate_agente']         = df.groupby('Agente')['target'].transform('mean')
    df['win_rate_mapa']           = df.groupby('Mapa')['target'].transform('mean')
    df['win_rate_jogador_agente'] = df.groupby(['Jogador','Agente'])['target'].transform('mean')
    df['win_rate_jogador_mapa']   = df.groupby(['Jogador','Mapa'])['target'].transform('mean')

    # Impacto
    df['Impacto'] = 0.0
    df['Impacto_Time_%'] = 0.0
    for i in range(0, len(df), 5):
        grupo = df.iloc[i:i+5]
        impactos = []
        for _, row in grupo.iterrows():
            role = row.get('Role')
            if role and role in PESO_ROLE:
                imp = sum(row[c] * PESO_ROLE[role][c] for c in COLUNAS_STATS)
            else:
                imp = 0.0
            impactos.append(imp)
        total = sum(impactos) or 1
        for j, idx in enumerate(grupo.index):
            df.loc[idx, 'Impacto']       = round(impactos[j], 3)
            df.loc[idx, 'Impacto_Time_%'] = round(impactos[j] / total * 100, 3)

    return df


@st.cache_resource(show_spinner="Treinando modelo ML…")
def treinar_modelo(df: pd.DataFrame):
    df_ml = df.dropna(subset=['Role'] + COLUNAS_STATS).copy()
    df_ml = df_ml.reset_index(drop=True)

    le = LabelEncoder()
    df_ml['Jogador_enc'] = le.fit_transform(df_ml['Jogador'])

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ohe_array = ohe.fit_transform(df_ml[['Mapa', 'Agente']])
    ohe_cols  = ohe.get_feature_names_out(['Mapa', 'Agente'])
    df_ohe    = pd.DataFrame(ohe_array, columns=ohe_cols, index=df_ml.index)
    df_ml     = pd.concat([df_ml, df_ohe], axis=1)

    features_num = [
        'R','ACS','K','D','A','+/-','KAST','ADR','HS%','FK','FD','+/-_FK_FD',
        'Impacto','Impacto_Time_%',
        'win_rate_jogador','win_rate_agente','win_rate_mapa',
        'win_rate_jogador_agente','win_rate_jogador_mapa',
    ]
    features = features_num + ['Jogador_enc'] + list(ohe_cols)

    # win rate suavizada
    K_SUAV = 10
    global_mean = df_ml['target'].mean()

    def wr_suav(sub, k=K_SUAV):
        n = len(sub)
        return (sub['target'].sum() + k * global_mean) / (n + k) if n else global_mean

    for idx, row in df_ml.iterrows():
        j, a, m = row['Jogador'], row['Agente'], row['Mapa']
        fj  = df_ml[df_ml['Jogador'] == j]
        fa  = df_ml[df_ml['Agente']  == a]
        fm  = df_ml[df_ml['Mapa']    == m]
        fja = fj[fj['Agente'] == a]
        fjm = fj[fj['Mapa']   == m]
        df_ml.at[idx, 'win_rate_jogador']        = wr_suav(fj)
        df_ml.at[idx, 'win_rate_agente']         = wr_suav(fa)
        df_ml.at[idx, 'win_rate_mapa']           = wr_suav(fm)
        df_ml.at[idx, 'win_rate_jogador_agente'] = wr_suav(fja)
        df_ml.at[idx, 'win_rate_jogador_mapa']   = wr_suav(fjm)

    X = df_ml[features].fillna(0)
    y = df_ml['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    modelo = CalibratedClassifierCV(rf, method='isotonic', cv=5)
    modelo.fit(X_train, y_train)

    y_proba = modelo.predict_proba(X_test)[:, 1]
    metricas = {
        'ROC-AUC': round(roc_auc_score(y_test, y_proba), 4),
        'Log Loss': round(log_loss(y_test, y_proba), 4),
        'Brier':    round(float(np.mean((y_proba - y_test.values) ** 2)), 4),
    }

    return modelo, le, ohe, ohe_cols, features, df_ml, metricas


# ══════════════════════════════════════════════════════════════════════════════
# FUNÇÕES AUXILIARES ML
# ══════════════════════════════════════════════════════════════════════════════

def calcular_impacto(df_time: pd.DataFrame) -> pd.DataFrame:
    df = df_time.copy()
    impactos = [
        sum(row[c] * PESO_ROLE[row['Role']][c] for c in COLUNAS_STATS)
        if row['Role'] in PESO_ROLE else 0.0
        for _, row in df.iterrows()
    ]
    total = sum(impactos) or 1
    df['Impacto']        = [round(v, 3) for v in impactos]
    df['Impacto_Time_%'] = [round(v / total * 100, 3) for v in impactos]
    return df


def obter_historico(jogador, agente, mapa, df_ml, global_mean):
    for mask in [
        (df_ml['Jogador']==jogador) & (df_ml['Agente']==agente) & (df_ml['Mapa']==mapa),
        (df_ml['Jogador']==jogador) & (df_ml['Agente']==agente),
        (df_ml['Jogador']==jogador) & (df_ml['Mapa']==mapa),
        (df_ml['Jogador']==jogador),
    ]:
        sub = df_ml[mask]
        if not sub.empty:
            r = sub.iloc[0]
            return {k: r[k] for k in ['win_rate_jogador','win_rate_agente','win_rate_mapa',
                                       'win_rate_jogador_agente','win_rate_jogador_mapa','Jogador_enc']}
    return {k: global_mean for k in ['win_rate_jogador','win_rate_agente','win_rate_mapa',
                                      'win_rate_jogador_agente','win_rate_jogador_mapa']} | {'Jogador_enc': -1}


def prever_jogador(jogador, agente, mapa, stats_row, df_ml, modelo, ohe, ohe_cols, features, global_mean):
    hist = obter_historico(jogador, agente, mapa, df_ml, global_mean)
    feat = {c: stats_row[c] for c in COLUNAS_STATS}
    feat['Impacto']        = stats_row['Impacto']
    feat['Impacto_Time_%'] = stats_row['Impacto_Time_%']
    feat.update(hist)
    ohe_vals = ohe.transform([[mapa, agente]])[0]
    feat.update(dict(zip(ohe_cols, ohe_vals)))
    X_in = pd.DataFrame([feat])[features].fillna(0)
    return float(np.clip(modelo.predict_proba(X_in)[0][1], 0.05, 0.95))


# ══════════════════════════════════════════════════════════════════════════════
# FUNÇÕES DE GRÁFICO
# ══════════════════════════════════════════════════════════════════════════════

def fig_barh(series: pd.Series, titulo: str, fmt=None):
    n = len(series)
    fig, ax = plt.subplots(figsize=(9, max(3, n * 0.45)))
    cmap = plt.cm.get_cmap('RdYlGn', n + 2)
    cores = [cmap(i + 2) for i in range(n)]
    vals  = series.values[::-1]
    labs  = series.index[::-1]
    bars  = ax.barh(labs, vals, color=cores, edgecolor='none', height=0.55)
    for bar, v in zip(bars, vals):
        label = fmt(v) if fmt else str(int(v))
        ax.text(bar.get_width() * 1.005 + 0.001, bar.get_y() + bar.get_height() / 2,
                label, va='center', fontsize=10, fontweight='bold', color=FG)
    ax.set_facecolor(BG); fig.patch.set_facecolor(BG)
    ax.tick_params(colors=FG, labelsize=10)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_title(titulo, color=FG, fontsize=13, fontweight='bold', pad=10)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🎯 Valorant VCT")
    st.markdown("---")
    csv_path = st.text_input("Caminho do CSV", value="status_individual2.csv")
    st.markdown("---")
    pagina = st.radio("Navegação", [
        "🏠 Início",
        "👤 Perfil do Jogador",
        "📊 Análises Gráficas",
        "🤖 Predição ML",
    ])

# ── Carrega dados ──────────────────────────────────────────────────────────────
try:
    df = carregar_dados(csv_path)
except FileNotFoundError:
    st.error(f"❌ Arquivo `{csv_path}` não encontrado. Ajuste o caminho na barra lateral.")
    st.stop()

ligas_disp = ["Todas"] + sorted(df['Liga'].dropna().unique().tolist())
mapas_disp = sorted(df[df['Mapa'] != 'AllMaps']['Mapa'].unique().tolist())
jogadores_disp = sorted(df['Jogador'].unique().tolist())


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA: INÍCIO
# ══════════════════════════════════════════════════════════════════════════════
if pagina == "🏠 Início":
    st.title("🎯 Valorant VCT Analytics")
    st.markdown("Análise completa de dados do VCT — perfis de jogadores, estatísticas visuais e predição de vitória via Machine Learning.")
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Partidas (linhas)", f"{len(df):,}")
    c2.metric("Jogadores únicos", df['Jogador'].nunique())
    c3.metric("Mapas", df[df['Mapa'] != 'AllMaps']['Mapa'].nunique())
    c4.metric("Ligas", df['Liga'].nunique())
    st.markdown("---")
    st.subheader("Distribuição de Ligas")
    liga_counts = df.groupby('Liga')['Jogador'].count()
    st.bar_chart(liga_counts)
    st.markdown("---")
    st.subheader("Agentes mais escolhidos")
    top_ag = df['Agente'].value_counts().head(15)
    st.bar_chart(top_ag)


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA: PERFIL DO JOGADOR
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "👤 Perfil do Jogador":
    st.title("👤 Perfil do Jogador")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        nome = st.selectbox("Jogador", jogadores_disp)
    with col_b:
        liga_filtro = st.selectbox("Liga", ligas_disp)

    df_f = df if liga_filtro == "Todas" else df[df['Liga'] == liga_filtro]
    jogador = df_f[df_f['Jogador'] == nome]

    if jogador.empty:
        st.warning("Nenhuma partida encontrada com os filtros selecionados.")
        st.stop()

    # cabeçalho
    st.markdown(f"### {nome}")
    times_str = ", ".join(jogador['Time'].unique())
    st.caption(f"Time(s): **{times_str}** | Liga: **{liga_filtro}**")
    st.markdown("---")

    # métricas gerais
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Partidas", len(jogador))
    m2.metric("Win Rate", f"{jogador['target'].mean():.1%}")
    m3.metric("Rating médio", f"{jogador['R'].mean():.3f}")
    m4.metric("ACS médio", f"{jogador['ACS'].mean():.0f}")
    m5.metric("Mapas diferentes", jogador['Mapa'].nunique())

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Médias")
        med = jogador[['R','ACS','KAST','ADR','HS%']].mean().round(3)
        st.dataframe(med.rename("Média").to_frame(), use_container_width=True)

        st.subheader("Totais")
        soma = jogador[['K','D','A','+/-','FK','FD','+/-_FK_FD']].sum().astype(int)
        st.dataframe(soma.rename("Total").to_frame(), use_container_width=True)

    with col2:
        st.subheader("Por Mapa")
        por_mapa = (jogador.groupby('Mapa')[['ACS','target']].mean().round(3)
                    .rename(columns={'ACS':'ACS_med','target':'Win_Rate'})
                    .sort_values('ACS_med', ascending=False))
        por_mapa['Win_Rate'] = por_mapa['Win_Rate'].map(lambda x: f"{x:.1%}")
        st.dataframe(por_mapa, use_container_width=True)

        st.subheader("Por Agente")
        por_ag = (jogador.groupby('Agente')[['ACS','target']].mean().round(3)
                  .rename(columns={'ACS':'ACS_med','target':'Win_Rate'})
                  .sort_values('ACS_med', ascending=False))
        por_ag['Win_Rate'] = por_ag['Win_Rate'].map(lambda x: f"{x:.1%}")
        st.dataframe(por_ag, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA: ANÁLISES GRÁFICAS
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "📊 Análises Gráficas":
    st.title("📊 Análises Gráficas")

    liga_g = st.selectbox("Liga", ligas_disp, key="liga_graf")
    df_g = df if liga_g == "Todas" else df[df['Liga'] == liga_g]
    df_g = df_g[df_g['Mapa'] != 'AllMaps']

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Mapas", "Agentes", "Top Jogadores", "Stats por Agente", "Melhor por Mapa"
    ])

    # ── Mapas ──────────────────────────────────────────────────────────────
    with tab1:
        mapas_cnt = df_g['Mapa'].value_counts()
        titulo = f"Jogos por Mapa — {liga_g}"
        fig, ax = plt.subplots(figsize=(9, max(3, len(mapas_cnt) * 0.45)))
        cores = [RED if v == mapas_cnt.max() else '#4a90d9' for v in mapas_cnt.values]
        vals  = mapas_cnt.values[::-1]; labs = mapas_cnt.index[::-1]
        bars  = ax.barh(labs, vals, color=cores[::-1], edgecolor='none', height=0.55)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width()*1.005, bar.get_y()+bar.get_height()/2,
                    str(v), va='center', fontsize=11, fontweight='bold', color=FG)
        ax.set_facecolor(BG); fig.patch.set_facecolor(BG)
        ax.tick_params(colors=FG, labelsize=11)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.xaxis.set_visible(False)
        ax.set_title(titulo, color=FG, fontsize=14, fontweight='bold', pad=12)
        plt.tight_layout()
        st.pyplot(fig)

    # ── Agentes ────────────────────────────────────────────────────────────
    with tab2:
        ag_cnt = df_g['Agente'].value_counts()
        fig2   = fig_barh(ag_cnt, f"Agentes Mais Escolhidos — {liga_g}")
        st.pyplot(fig2)

    # ── Top Jogadores ──────────────────────────────────────────────────────
    with tab3:
        STATS_MEDIA = ['R', 'ACS', 'KAST', 'ADR', 'HS%']
        STATS_SOMA  = ['K', 'D', 'A', '+/-', 'FK', 'FD', '+/-_FK_FD']
    
        c1, c2 = st.columns(2)
        stat_t = c1.selectbox("Estatística", STATS_MEDIA + STATS_SOMA, key='stat_top')
        top_n  = c2.slider("Top N", 5, 30, 10, key='topn')
    
        agg_func   = 'mean' if stat_t in STATS_MEDIA else 'sum'
        label_agg  = 'Média' if stat_t in STATS_MEDIA else 'Total'
        fmt_func   = (lambda v: f"{v:.2f}") if stat_t in STATS_MEDIA else (lambda v: str(int(v)))
    
        top_j = (df_g.groupby('Jogador')[stat_t]
                     .agg(agg_func)
                     .sort_values(ascending=False)
                     .head(top_n))
    
        fig3 = fig_barh(top_j, f"Top {top_n} Jogadores — {stat_t} ({label_agg}) — {liga_g}", fmt=fmt_func)
        st.pyplot(fig3)

    # ── Stats por Agente ───────────────────────────────────────────────────
    with tab4:
        c1, c2 = st.columns(2)
        stat_a = c1.selectbox("Estatística", ['R','ACS','K','D','A','+/-','KAST','ADR','HS%','FK','FD','+/-_FK_FD'], key='stat_ag')
        min_p  = c2.slider("Mín. partidas", 1, 30, 5, key='minp')
        stats_ag = (df_g.groupby('Agente')
                       .agg(Partidas=('Agente','count'), Val=(stat_a,'mean'))
                       .query(f'Partidas >= {min_p}')
                       .sort_values('Val', ascending=False)['Val'])
        fig4 = fig_barh(stats_ag, f"{stat_a} Médio por Agente — {liga_g}",
                        fmt=lambda v: f"{v:.2f}")
        st.pyplot(fig4)

    # ── Melhor por Mapa ────────────────────────────────────────────────────
    with tab5:
        c1, c2 = st.columns(2)
        stat_m  = c1.selectbox("Estatística", ['R','ACS','ADR','KAST'], key='stat_mapa')
        min_pm  = c2.slider("Mín. partidas", 1, 10, 2, key='minpm')
        melhor  = (df_g.groupby(['Mapa','Jogador'])
                       .agg(Val=(stat_m,'mean'), Partidas=(stat_m,'count'))
                       .query(f'Partidas >= {min_pm}')
                       .reset_index()
                       .sort_values('Val', ascending=False)
                       .groupby('Mapa').first().reset_index()
                       .sort_values('Val', ascending=False))
        melhor.index = [f"{r.Mapa}  →  {r.Jogador}" for _, r in melhor.iterrows()]
        fig5 = fig_barh(melhor['Val'], f"Melhor por Mapa ({stat_m}) — {liga_g}",
                        fmt=lambda v: f"{v:.2f}")
        st.pyplot(fig5)


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA: PREDIÇÃO ML
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "🤖 Predição ML":
    st.title("🤖 Predição de Vitória — ML")

    # treina / usa cache
    with st.spinner("Preparando modelo…"):
        modelo, le, ohe, ohe_cols, features, df_ml, metricas = treinar_modelo(df)

    global_mean = df_ml['target'].mean()

    # métricas do modelo
    st.subheader("Desempenho do Modelo")
    mc = st.columns(3)
    mc[0].metric("ROC-AUC",  metricas['ROC-AUC'])
    mc[1].metric("Log Loss", metricas['Log Loss'])
    mc[2].metric("Brier",    metricas['Brier'])
    st.markdown("---")

    # seleção do mapa
    mapa_sel = st.selectbox("Mapa da partida", mapas_disp)
    st.markdown("---")

    # formulário do time (5 jogadores)
    st.subheader("Insira os 5 jogadores do time")

    jogadores_rows = []
    for i in range(1, 6):
        st.markdown(f"**Jogador {i}**")
        c = st.columns([2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        jog  = c[0].text_input("Nome",   key=f"j{i}_nome",  placeholder="ex: TenZ")
        ag   = c[1].selectbox("Agente",  TODOS_AGENTES, key=f"j{i}_ag")
        r    = c[2].number_input("R",    value=1.0,  step=0.01, key=f"j{i}_r",    format="%.2f")
        acs  = c[3].number_input("ACS",  value=200,  step=1,    key=f"j{i}_acs")
        k    = c[4].number_input("K",    value=15,   step=1,    key=f"j{i}_k")
        d    = c[5].number_input("D",    value=15,   step=1,    key=f"j{i}_d")
        a    = c[6].number_input("A",    value=5,    step=1,    key=f"j{i}_a")
        kast = c[7].number_input("KAST", value=70,   step=1,    key=f"j{i}_kast")
        adr  = c[8].number_input("ADR",  value=130,  step=1,    key=f"j{i}_adr")
        hs   = c[9].number_input("HS%",  value=25,   step=1,    key=f"j{i}_hs")
        fk   = c[10].number_input("FK",  value=1,    step=1,    key=f"j{i}_fk")
        fd   = c[11].number_input("FD",  value=1,    step=1,    key=f"j{i}_fd")

        pm = fk - fd
        role = AGENTE_PARA_ROLE.get(ag, "duelistas")
        jogadores_rows.append({
            'Jogador': jog or f"Jogador{i}",
            'Agente': ag, 'Role': role,
            'R': r, 'ACS': acs, 'K': k, 'D': d, 'A': a,
            '+/-': k - d, 'KAST': kast, 'ADR': adr, 'HS%': hs,
            'FK': fk, 'FD': fd, '+/-_FK_FD': pm,
        })

    if st.button("🎯 Calcular Probabilidade de Vitória", type="primary"):
        df_time = calcular_impacto(pd.DataFrame(jogadores_rows))
        resultados = []
        for _, row in df_time.iterrows():
            prob = prever_jogador(
                jogador=row['Jogador'], agente=row['Agente'], mapa=mapa_sel,
                stats_row=row, df_ml=df_ml, modelo=modelo,
                ohe=ohe, ohe_cols=ohe_cols, features=features,
                global_mean=global_mean,
            )
            hist = len(df_ml[(df_ml['Jogador']==row['Jogador']) & (df_ml['Agente']==row['Agente'])])
            resultados.append({
                'Jogador':         row['Jogador'],
                'Agente':          row['Agente'],
                'Role':            row['Role'],
                'Impacto no Time': f"{row['Impacto_Time_%']:.1f}%",
                'Jogos históricos': hist,
                'Prob. Vitória':   f"{prob*100:.1f}%",
                '_prob':           prob,
            })

        df_res = pd.DataFrame(resultados).sort_values('_prob', ascending=False)
        media_prob = df_res['_prob'].mean()

        st.markdown("---")
        st.subheader(f"Resultado — Mapa: {mapa_sel}")

        # gauge de prob média
        cor_gauge = RED if media_prob < 0.45 else ("#f5a623" if media_prob < 0.55 else "#7ed321")
        st.markdown(
            f"<h2 style='text-align:center;color:{cor_gauge}'>"
            f"Probabilidade média do time: {media_prob*100:.1f}%</h2>",
            unsafe_allow_html=True,
        )

        st.dataframe(
            df_res.drop(columns='_prob').reset_index(drop=True),
            use_container_width=True,
        )

        # gráfico de barras horizontais
        probs = df_res.set_index('Jogador')['_prob']
        fig, ax = plt.subplots(figsize=(8, 3.5))
        cores = [RED if v < 0.45 else ("#f5a623" if v < 0.55 else "#7ed321") for v in probs.values]
        bars  = ax.barh(probs.index[::-1], probs.values[::-1], color=cores[::-1], height=0.5)
        for bar, v in zip(bars, probs.values[::-1]):
            ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
                    f"{v*100:.1f}%", va='center', fontsize=11, fontweight='bold', color=FG)
        ax.axvline(0.5, color='white', linestyle='--', alpha=0.4)
        ax.set_xlim(0, 1.12)
        ax.set_facecolor(BG); fig.patch.set_facecolor(BG)
        ax.tick_params(colors=FG, labelsize=11)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.xaxis.set_visible(False)
        ax.set_title("Probabilidade de Vitória por Jogador", color=FG, fontsize=13, fontweight='bold', pad=10)
        plt.tight_layout()
        st.pyplot(fig)