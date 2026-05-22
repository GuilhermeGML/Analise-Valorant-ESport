"""
dashboard.py
============
Streamlit dashboard interativo para análise de stats VCT Americas.
Instalar:  pip install streamlit plotly pandas matplotlib
Rodar:     streamlit run dashboard.py
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─────────────────────────────────────────────
#  CONFIGURAÇÃO DA PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="VCT Americas Dashboard",
    page_icon="🎯",
    layout="wide",
)
st.markdown("""
<style>
    .main { background-color: #0e0e1a; }
    h1, h2, h3 { color: #ff4655; }
    .stSelectbox label, .stMultiSelect label { color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CARREGAR DADOS
# ─────────────────────────────────────────────
@st.cache_data
def carregar_dados():
    stats   = pd.read_csv("vlr_stats2.csv")
    placar  = pd.read_csv("vlr_placar2.csv")
    impacto = pd.read_csv("df_impacto_cada_mapa.csv")

    funcoes = {
        'Duelista':    ['Jett', 'Reyna', 'Raze', 'Yoru', 'Neon', 'Iso', 'Waylay'],
        'Iniciador':   ['Sova', 'Breach', 'Skye', 'Kayo', 'Fade', 'Gekko', 'Tejo'],
        'Controlador': ['Brimstone', 'Viper', 'Omen', 'Astra', 'Harbor', 'Clove'],
        'Sentinela':   ['Killjoy', 'Cypher', 'Sage', 'Chamber', 'Deadlock', 'Vyse'],
    }
    agente_funcao = {a: f for f, agents in funcoes.items() for a in agents}
    stats['Funcao'] = stats['Agente'].map(agente_funcao).fillna('Outro')

    return stats, placar, impacto

stats, placar, df_impc = carregar_dados()

df      = stats[stats['Mapa'] != 'AllMaps']
df_all  = stats[stats['Mapa'] == 'AllMaps']

df_placar = placar[
    (placar['Mapa'] == 'AllMaps') &
    placar['Mapas_T1'].notna() &
    placar['Mapas_T2'].notna()
].drop_duplicates(subset='URL', keep='last')

df_mapas_ind = placar[placar['Mapa'] != 'AllMaps']
df_placar_mapas = placar[placar['Mapa'] != 'AllMaps'].copy()

MAPAS     = sorted(df['Mapa'].dropna().unique())
TIMES     = sorted(df['Time'].dropna().unique())
JOGADORES = sorted(df['Jogador'].dropna().unique())
AGENTES   = sorted(df['Agente'].dropna().unique())
STATS     = ['R', 'ACS', 'K', 'D', 'A', '+/-', 'KAST', 'ADR', 'HS%', 'FK', 'FD']
TEMPLATE  = "plotly_dark"
COR_PRIN  = "#ff4655"

TIMES_IMPACTO = sorted(df_impc['Time'].dropna().unique())

TAG_NOME = {
    'SEN':  'Sentinels',   'NRG':  'NRG',         'C9':   'Cloud9',
    'FUR':  'FURIA',       'LEV':  'LEVIATÁN',    'EG':   'Evil Geniuses',
    '100T': '100 Thieves', 'KRU':  'KRÜ Esports', 'G2':   'G2 Esports',
    'ENVY': 'ENVY',        'MIBR': 'MIBR',         'LOUD': 'LOUD',
    '2G':   '2Game Esports',
}

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.title("🎯 VCT Americas Dashboard")
st.markdown("---")

# ─────────────────────────────────────────────
#  MÉTRICAS GERAIS NO TOPO
# ─────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total de Jogos",  df_placar['URL'].nunique())
c2.metric("Mapas Jogados",   len(df_mapas_ind))
c3.metric("Times",           df['Time'].nunique())
c4.metric("Jogadores",       df['Jogador'].nunique())
c5.metric("Agentes Usados",  df['Agente'].nunique())
st.markdown("---")

# ─────────────────────────────────────────────
#  ABAS PRINCIPAIS
# ─────────────────────────────────────────────
aba1, aba2, aba3, aba4, aba5, aba6, aba7, aba8 = st.tabs([
    "📊 Ranking de Stats",
    "🗺️ Análise por Mapa",
    "👤 Perfil do Jogador",
    "🧬 Análise por Agente",
    "🏆 Resultados",
    "⚔️ Confrontos Diretos",
    "💥 Impacto por Time",
    "📈 Visão Geral",
])

# ══════════════════════════════════════════════
#  ABA 1 — RANKING DE STATS
# ══════════════════════════════════════════════
with aba1:
    st.subheader("Ranking de Jogadores por Estatística")
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        stat_sel = st.selectbox("Estatística", STATS, index=0)
    with col2:
        agregacao = st.selectbox("Agregação", ["Média", "Total", "Máximo"])
    with col3:
        top_n = st.slider("Top N", 5, len(JOGADORES), 15)

    with st.expander("Filtros opcionais"):
        f_time = st.multiselect("Filtrar por Time", TIMES)
        f_mapa = st.multiselect("Filtrar por Mapa", MAPAS)
        f_min_partidas = st.slider("Mínimo de partidas", 1, 20, 3)

    df_rank = df.copy()
    if f_time:
        df_rank = df_rank[df_rank['Time'].isin(f_time)]
    if f_mapa:
        df_rank = df_rank[df_rank['Mapa'].isin(f_mapa)]

    agg_func = {'Média': 'mean', 'Total': 'sum', 'Máximo': 'max'}[agregacao]
    ranking = (df_rank.groupby(['Jogador', 'Time'])[stat_sel]
               .agg(['count', agg_func])
               .reset_index()
               .rename(columns={'count': 'Partidas', agg_func: stat_sel})
               .query(f'Partidas >= {f_min_partidas}')
               .sort_values(stat_sel, ascending=False)
               .head(top_n))

    if ranking.empty:
        st.warning("Nenhum dado com esses filtros.")
    else:
        fig = px.bar(
            ranking[::-1], x=stat_sel, y='Jogador', orientation='h',
            color=stat_sel, color_continuous_scale='RdYlGn',
            text=ranking[stat_sel][::-1].round(2),
            hover_data=['Time', 'Partidas'], template=TEMPLATE,
            title=f"Top {top_n} — {stat_sel} ({agregacao})",
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(height=max(400, top_n * 35), showlegend=False,
                          coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(ranking.reset_index(drop=True), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════
#  ABA 2 — ANÁLISE POR MAPA
# ══════════════════════════════════════════════
with aba2:
    st.subheader("Análise por Mapa")
    mapa_sel = st.selectbox("Selecione o Mapa", MAPAS, key="mapa_aba2")
    df_mapa  = df[df['Mapa'] == mapa_sel]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"#### Médias em {mapa_sel}")
        medias = df_mapa[STATS].mean().round(2).reset_index()
        medias.columns = ['Estatística', 'Média']
        fig = px.bar(medias, x='Média', y='Estatística', orientation='h',
                     color='Média', color_continuous_scale='Blues',
                     template=TEMPLATE, text='Média')
        fig.update_traces(textposition='outside')
        fig.update_layout(height=400, showlegend=False,
                          coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(f"#### Melhores Jogadores em {mapa_sel}")
        stat_mapa = st.selectbox("Stat", STATS, key="stat_mapa")
        top_mapa = (df_mapa.groupby(['Jogador', 'Time'])[stat_mapa]
                    .mean().round(2).reset_index()
                    .sort_values(stat_mapa, ascending=False).head(10))
        fig2 = px.bar(top_mapa[::-1], x=stat_mapa, y='Jogador', orientation='h',
                      color=stat_mapa, color_continuous_scale='RdYlGn',
                      hover_data=['Time'], template=TEMPLATE,
                      text=top_mapa[stat_mapa][::-1])
        fig2.update_traces(textposition='outside')
        fig2.update_layout(height=400, showlegend=False,
                           coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown(f"#### Agentes Mais Usados em {mapa_sel}")
    agentes_mapa = df_mapa['Agente'].value_counts().reset_index()
    agentes_mapa.columns = ['Agente', 'Vezes']
    fig3 = px.bar(agentes_mapa, x='Agente', y='Vezes',
                  color='Vezes', color_continuous_scale='Reds',
                  template=TEMPLATE, text='Vezes')
    fig3.update_traces(textposition='outside')
    fig3.update_layout(height=350, showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown(f"#### Média de Kills por Mapa — Todos os Mapas")
    kills_mapa = (df.groupby('Mapa')['K'].mean().round(2)
                  .sort_values().reset_index())
    kills_mapa.columns = ['Mapa', 'Média de Kills']
    fig_km = px.bar(kills_mapa, x='Média de Kills', y='Mapa', orientation='h',
                    color='Média de Kills', color_continuous_scale='YlOrRd',
                    template=TEMPLATE, text='Média de Kills')
    fig_km.update_traces(textposition='outside')
    fig_km.update_layout(height=400, showlegend=False,
                         coloraxis_showscale=False, yaxis_title="")
    st.plotly_chart(fig_km, use_container_width=True)

# ══════════════════════════════════════════════
#  ABA 3 — PERFIL DO JOGADOR
# ══════════════════════════════════════════════
with aba3:
    st.subheader("Perfil do Jogador")
    col1, col2 = st.columns([2, 2])
    with col1:
        jogador_sel = st.selectbox("Jogador", JOGADORES)
    with col2:
        mapa_jogador = st.selectbox("Mapa (opcional)", ["Todos"] + MAPAS, key="mapa_jogador")

    df_jog = df[df['Jogador'] == jogador_sel]
    if mapa_jogador != "Todos":
        df_jog = df_jog[df_jog['Mapa'] == mapa_jogador]

    if df_jog.empty:
        st.warning("Sem dados para esse jogador/mapa.")
    else:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Partidas",     len(df_jog))
        m2.metric("Rating Médio", f"{df_jog['R'].mean():.2f}")
        m3.metric("ACS Médio",    f"{df_jog['ACS'].mean():.0f}")
        m4.metric("K/D",          f"{df_jog['K'].sum() / max(df_jog['D'].sum(), 1):.2f}")
        m5.metric("KAST Médio",   f"{df_jog['KAST'].mean():.0f}%")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Radar de Performance")
            cats   = ['R', 'ACS', 'KAST', 'ADR', 'HS%']
            medias = df_jog[cats].mean().round(2)
            maximos = df[cats].max()
            norm    = (medias / maximos * 100).round(1)
            fig_radar = go.Figure(go.Scatterpolar(
                r=norm.values.tolist() + [norm.values[0]],
                theta=cats + [cats[0]],
                fill='toself',
                fillcolor='rgba(255,70,85,0.3)',
                line=dict(color=COR_PRIN, width=2),
                name=jogador_sel,
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100],
                                    color='white', gridcolor='#333'),
                    angularaxis=dict(color='white'),
                    bgcolor='#1a1a2e',
                ),
                paper_bgcolor='#0e0e1a', font_color='white',
                height=350, showlegend=False,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with col2:
            st.markdown("#### Rating por Mapa")
            rating_mapa = (df[df['Jogador'] == jogador_sel]
                           .groupby('Mapa')['R'].mean().round(2)
                           .sort_values(ascending=False).reset_index())
            fig_rm = px.bar(rating_mapa, x='Mapa', y='R',
                            color='R', color_continuous_scale='RdYlGn',
                            template=TEMPLATE, text='R')
            fig_rm.update_traces(textposition='outside')
            fig_rm.update_layout(height=350, showlegend=False,
                                 coloraxis_showscale=False, yaxis_title="Rating Médio")
            st.plotly_chart(fig_rm, use_container_width=True)

        st.markdown("#### Agentes Mais Usados")
        agentes_jog = (df[df['Jogador'] == jogador_sel]['Agente']
                       .value_counts().reset_index())
        agentes_jog.columns = ['Agente', 'Vezes']
        fig_ag = px.bar(agentes_jog, x='Agente', y='Vezes',
                        color='Vezes', color_continuous_scale='Blues',
                        template=TEMPLATE, text='Vezes')
        fig_ag.update_traces(textposition='outside')
        fig_ag.update_layout(height=300, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_ag, use_container_width=True)

        with st.expander("Ver tabela completa"):
            st.dataframe(
                df_jog[['Mapa','Agente','R','ACS','K','D','A','+/-','KAST','ADR','HS%','FK','FD']],
                use_container_width=True, hide_index=True,
            )

# ══════════════════════════════════════════════
#  ABA 4 — ANÁLISE POR AGENTE
# ══════════════════════════════════════════════
with aba4:
    st.subheader("Análise por Agente")

    col1, col2 = st.columns([2, 1])
    with col1:
        stat_agente = st.selectbox("Estatística", STATS, key="stat_agente")
    with col2:
        min_pick = st.slider("Mínimo de picks", 1, 20, 3, key="min_pick")

    stats_ag = (df.groupby(['Agente', 'Funcao'])[stat_agente]
                .agg(['mean', 'count']).reset_index()
                .rename(columns={'mean': stat_agente, 'count': 'Picks'})
                .query(f'Picks >= {min_pick}')
                .sort_values(stat_agente, ascending=False))

    fig_ag = px.bar(
        stats_ag[::-1], x=stat_agente, y='Agente', orientation='h',
        color='Funcao',
        color_discrete_map={
            'Duelista': '#ff4655', 'Iniciador': '#1a9fff',
            'Controlador': '#00c853', 'Sentinela': '#ffab00', 'Outro': '#888888',
        },
        hover_data=['Picks'], template=TEMPLATE,
        text=stats_ag[stat_agente][::-1].round(2),
        title=f"{stat_agente} médio por Agente",
    )
    fig_ag.update_traces(textposition='outside')
    fig_ag.update_layout(height=max(400, len(stats_ag) * 30),
                         yaxis_title="", legend_title="Função")
    st.plotly_chart(fig_ag, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Agentes Mais Escolhidos")
        agentes_count = df['Agente'].value_counts().sort_values().reset_index()
        agentes_count.columns = ['Agente', 'Picks']
        fig_pick = px.bar(agentes_count, x='Picks', y='Agente', orientation='h',
                          color='Picks', color_continuous_scale='Blues',
                          template=TEMPLATE, text='Picks')
        fig_pick.update_traces(textposition='outside')
        fig_pick.update_layout(height=max(400, len(agentes_count) * 22),
                               showlegend=False, coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig_pick, use_container_width=True)

    with col2:
        st.markdown("#### Rating Médio por Agente")
        stats_ag2 = (df.groupby('Agente')['R'].mean().round(2)
                     .sort_values().reset_index())
        fig_rat = px.bar(stats_ag2, x='R', y='Agente', orientation='h',
                         color='R', color_continuous_scale='RdYlGn',
                         template=TEMPLATE, text='R',
                         title='Rating Médio por Agente')
        fig_rat.update_traces(textposition='outside')
        fig_rat.update_layout(height=max(400, len(stats_ag2) * 22),
                              showlegend=False, coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig_rat, use_container_width=True)

    st.markdown("#### Impacto por Função")
    impacto_func = (df.groupby('Funcao')[['R', 'ACS', 'ADR', 'KAST']]
                    .mean().round(2).reset_index())
    fig_func = px.bar(
        impacto_func.melt(id_vars='Funcao'),
        x='Funcao', y='value', color='variable', barmode='group',
        template=TEMPLATE,
        color_discrete_sequence=['#ff4655', '#1a9fff', '#00c853', '#ffab00'],
        labels={'value': 'Média', 'variable': 'Stat'},
        title="Médias por Função",
    )
    fig_func.update_layout(height=400)
    st.plotly_chart(fig_func, use_container_width=True)

    st.markdown("#### Agente Mais Jogado por Jogador")
    agente_jogador = (df.groupby(['Jogador', 'Agente']).size()
                      .reset_index(name='Partidas')
                      .sort_values('Partidas', ascending=False))
    top_agente = (agente_jogador.groupby('Jogador').first()
                  .reset_index().sort_values('Partidas'))
    top_agente['Label'] = top_agente['Jogador'] + ' (' + top_agente['Agente'] + ')'
    fig_taj = px.bar(top_agente, x='Partidas', y='Label', orientation='h',
                     color='Partidas', color_continuous_scale='Purples',
                     template=TEMPLATE, text='Partidas')
    fig_taj.update_traces(textposition='outside')
    fig_taj.update_layout(height=max(400, len(top_agente) * 22),
                          showlegend=False, coloraxis_showscale=False, yaxis_title="")
    st.plotly_chart(fig_taj, use_container_width=True)

# ══════════════════════════════════════════════
#  ABA 5 — RESULTADOS
# ══════════════════════════════════════════════
with aba5:
    st.subheader("Resultados dos Jogos")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Vitórias por Time")
        vitorias_t1 = df_placar.groupby('Time1')['Mapas_T1'].sum()
        vitorias_t2 = df_placar.groupby('Time2')['Mapas_T2'].sum()
        vitorias    = (vitorias_t1.add(vitorias_t2, fill_value=0)
                       .astype(int).sort_values(ascending=False).reset_index())
        vitorias.columns = ['Time', 'Mapas Ganhos']
        fig_v = px.bar(vitorias[::-1], x='Mapas Ganhos', y='Time', orientation='h',
                       color='Mapas Ganhos', color_continuous_scale='RdYlGn',
                       template=TEMPLATE, text='Mapas Ganhos')
        fig_v.update_traces(textposition='outside')
        fig_v.update_layout(height=400, showlegend=False,
                            coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig_v, use_container_width=True)

    with col2:
        st.markdown("#### Rounds Totais por Time")
        score_t1 = df_placar.groupby('Time1')['Score_T1'].sum()
        score_t2 = df_placar.groupby('Time2')['Score_T2'].sum()
        rounds   = (score_t1.add(score_t2, fill_value=0)
                    .astype(int).sort_values(ascending=False).reset_index())
        rounds.columns = ['Time', 'Rounds']
        fig_r = px.bar(rounds[::-1], x='Rounds', y='Time', orientation='h',
                       color='Rounds', color_continuous_scale='Blues',
                       template=TEMPLATE, text='Rounds')
        fig_r.update_traces(textposition='outside')
        fig_r.update_layout(height=400, showlegend=False,
                            coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig_r, use_container_width=True)

    st.markdown("#### Jogos por Mapa")
    jogos_mapa = df_mapas_ind['Mapa'].value_counts().reset_index()
    jogos_mapa.columns = ['Mapa', 'Jogos']
    fig_jm = px.bar(jogos_mapa, x='Mapa', y='Jogos',
                    color='Jogos', color_continuous_scale='Reds',
                    template=TEMPLATE, text='Jogos',
                    title="Frequência de cada Mapa")
    fig_jm.update_traces(textposition='outside')
    fig_jm.update_layout(height=350, showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig_jm, use_container_width=True)

# ══════════════════════════════════════════════
#  ABA 6 — CONFRONTOS DIRETOS
# ══════════════════════════════════════════════
with aba6:
    st.subheader("Confrontos Diretos entre Times")

    df_conf = df_placar_mapas.copy()
    times_conf = sorted(set(df_conf['Time1'].dropna()).union(set(df_conf['Time2'].dropna())))

    wins  = pd.DataFrame(0, index=times_conf, columns=times_conf)
    jogos = pd.DataFrame(0, index=times_conf, columns=times_conf)

    for _, row in df_conf.iterrows():
        t1, t2, v = row.get('Time1'), row.get('Time2'), row.get('Vencedor')
        if pd.isna(t1) or pd.isna(t2) or pd.isna(v) or t1 == t2:
            continue
        jogos.loc[t1, t2] += 1
        jogos.loc[t2, t1] += 1
        if v == t1:
            wins.loc[t1, t2] += 1
        elif v == t2:
            wins.loc[t2, t1] += 1

    rows = []
    for t1 in times_conf:
        for t2 in times_conf:
            if t1 >= t2:
                continue
            n = jogos.loc[t1, t2]
            if n == 0:
                continue
            w1, w2 = int(wins.loc[t1, t2]), int(wins.loc[t2, t1])
            rows.append({"Time": t1, "Adversario": t2, "Vitorias": w1, "Derrotas": w2,
                         "Mapas": int(n), "WinRate": round(w1/n*100, 1)})
            rows.append({"Time": t2, "Adversario": t1, "Vitorias": w2, "Derrotas": w1,
                         "Mapas": int(n), "WinRate": round(w2/n*100, 1)})

    df_matchups = pd.DataFrame(rows).sort_values(["Time", "Adversario"]).reset_index(drop=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        time_conf_sel = st.selectbox("Filtrar por Time", ["Todos"] + times_conf)

    df_match_view = df_matchups if time_conf_sel == "Todos" else df_matchups[df_matchups['Time'] == time_conf_sel]

    fig_match = px.bar(
        df_match_view.sort_values('WinRate', ascending=False),
        x='Adversario', y='WinRate', color='WinRate',
        color_continuous_scale='RdYlGn', template=TEMPLATE,
        text='WinRate', hover_data=['Vitorias', 'Derrotas', 'Mapas'],
        title=f"Win Rate por Confronto — {time_conf_sel}",
        labels={'WinRate': 'Win Rate (%)'},
    )
    fig_match.update_traces(textposition='outside')
    fig_match.update_layout(height=400, showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig_match, use_container_width=True)

    with st.expander("Ver tabela completa de confrontos"):
        st.dataframe(df_match_view, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════
#  ABA 7 — IMPACTO POR TIME / JOGADOR
# ══════════════════════════════════════════════
with aba7:
    st.subheader("Análise de Impacto")

    subtab1, subtab2 = st.tabs(["🏟️ Impacto por Time", "👤 Impacto por Jogador no Time"])

    # ── Sub-aba 1: Impacto vs WinRate por Time ────────────────────────────────
    with subtab1:
        df_time_imp = (
            df_impc.groupby("Time")
            .agg(
                Impacto_medio=("Impacto_Geral", "mean"),
                WinRate_medio=("target", "mean"),
                Jogadores=("Jogador", "nunique"),
            )
            .reset_index()
        )
        df_time_imp["WinRate_medio"] = df_time_imp["WinRate_medio"] * 100

        med_imp = df_time_imp["Impacto_medio"].median()
        med_wr  = df_time_imp["WinRate_medio"].median()

        fig_imp = go.Figure()

        # Linhas de quadrante
        x_min = df_time_imp["Impacto_medio"].min() - 0.02
        x_max = df_time_imp["Impacto_medio"].max() + 0.02
        y_min = df_time_imp["WinRate_medio"].min() - 5
        y_max = df_time_imp["WinRate_medio"].max() + 5

        fig_imp.add_shape(type="line", x0=med_imp, x1=med_imp, y0=y_min, y1=y_max,
                          line=dict(color="rgba(255,255,255,0.15)", dash="dash"))
        fig_imp.add_shape(type="line", x0=x_min, x1=x_max, y0=med_wr, y1=med_wr,
                          line=dict(color="rgba(255,255,255,0.15)", dash="dash"))

        cores_quad = {
            (True, True):   "#2ecc71",
            (True, False):  "#e74c3c",
            (False, True):  "#f39c12",
            (False, False): "#95a5a6",
        }

        for _, row in df_time_imp.iterrows():
            alto_imp = row["Impacto_medio"] >= med_imp
            alta_wr  = row["WinRate_medio"]  >= med_wr
            cor = cores_quad[(alto_imp, alta_wr)]
            fig_imp.add_trace(go.Scatter(
                x=[row["Impacto_medio"]], y=[row["WinRate_medio"]],
                mode='markers+text',
                marker=dict(size=20 + row["Jogadores"] * 3, color=cor,
                            line=dict(color='white', width=1)),
                text=[row["Time"]], textposition="middle center",
                textfont=dict(color='white', size=10, family='Arial Black'),
                name=row["Time"],
                hovertemplate=(f"<b>{row['Time']}</b><br>"
                               f"Impacto: {row['Impacto_medio']:.3f}<br>"
                               f"WinRate: {row['WinRate_medio']:.1f}%<br>"
                               f"Jogadores: {int(row['Jogadores'])}<extra></extra>"),
            ))

        fig_imp.update_layout(
            template=TEMPLATE, height=550,
            xaxis_title="Impacto Médio", yaxis_title="Win Rate Médio (%)",
            title="Impacto vs Win Rate por Time",
            showlegend=False,
            xaxis=dict(range=[x_min, x_max]),
            yaxis=dict(range=[y_min, y_max]),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.markdown("🟢 **Alto Impacto & Vence**")
        col2.markdown("🔴 **Alto Impacto & Perde**")
        col3.markdown("🟠 **Baixo Impacto & Vence**")
        col4.markdown("⚪ **Baixo Impacto & Perde**")

    # ── Sub-aba 2: Impacto por Jogador no Time ────────────────────────────────
    with subtab2:
        time_sel_imp = st.selectbox("Selecione o Time", TIMES_IMPACTO, key="time_imp")
        TIME_NOME_SEL = TAG_NOME.get(time_sel_imp, time_sel_imp)

        df_filtrado = df_impc[df_impc["Time"] == time_sel_imp].copy()

        if df_filtrado.empty:
            st.warning(f"Time '{time_sel_imp}' não encontrado. Times disponíveis: {TIMES_IMPACTO}")
        else:
            df_jog_imp = (
                df_filtrado.groupby("Jogador")
                .agg(
                    Impacto_medio=("Impacto_Geral", "mean"),
                    WinRate_medio=("target", "mean"),
                    Mapas=("Mapa", "count"),
                )
                .reset_index()
            )
            df_jog_imp["WinRate_medio"] = df_jog_imp["WinRate_medio"] * 100

            med_imp_j = df_jog_imp["Impacto_medio"].median()
            med_wr_j  = df_jog_imp["WinRate_medio"].median()

            fig_jog_imp = go.Figure()

            x_min_j = df_jog_imp["Impacto_medio"].min() - 0.05
            x_max_j = df_jog_imp["Impacto_medio"].max() + 0.05
            y_min_j = df_jog_imp["WinRate_medio"].min() - 5
            y_max_j = df_jog_imp["WinRate_medio"].max() + 5

            fig_jog_imp.add_shape(type="line", x0=med_imp_j, x1=med_imp_j,
                                  y0=y_min_j, y1=y_max_j,
                                  line=dict(color="rgba(255,255,255,0.15)", dash="dash"))
            fig_jog_imp.add_shape(type="line", x0=x_min_j, x1=x_max_j,
                                  y0=med_wr_j, y1=med_wr_j,
                                  line=dict(color="rgba(255,255,255,0.15)", dash="dash"))

            for _, row in df_jog_imp.iterrows():
                alto_imp = row["Impacto_medio"] >= med_imp_j
                alta_wr  = row["WinRate_medio"]  >= med_wr_j
                cor = cores_quad[(alto_imp, alta_wr)]
                fig_jog_imp.add_trace(go.Scatter(
                    x=[row["Impacto_medio"]], y=[row["WinRate_medio"]],
                    mode='markers+text',
                    marker=dict(size=30 + row["Mapas"] * 4, color=cor,
                                line=dict(color='white', width=1)),
                    text=[row["Jogador"]], textposition="middle center",
                    textfont=dict(color='white', size=10, family='Arial Black'),
                    name=row["Jogador"],
                    hovertemplate=(f"<b>{row['Jogador']}</b><br>"
                                   f"Impacto: {row['Impacto_medio']:.3f}<br>"
                                   f"WinRate: {row['WinRate_medio']:.1f}%<br>"
                                   f"Mapas: {int(row['Mapas'])}<extra></extra>"),
                ))

            fig_jog_imp.update_layout(
                template=TEMPLATE, height=550,
                xaxis_title="Impacto Médio", yaxis_title="Win Rate Médio (%)",
                title=f"Impacto vs Win Rate — {TIME_NOME_SEL} ({time_sel_imp})",
                showlegend=False,
                xaxis=dict(range=[x_min_j, x_max_j]),
                yaxis=dict(range=[y_min_j, y_max_j]),
            )
            st.plotly_chart(fig_jog_imp, use_container_width=True)

            col1, col2, col3, col4 = st.columns(4)
            col1.markdown("🟢 **Alto Impacto & Vence**")
            col2.markdown("🔴 **Alto Impacto & Perde**")
            col3.markdown("🟠 **Baixo Impacto & Vence**")
            col4.markdown("⚪ **Baixo Impacto & Perde**")

            with st.expander("Ver dados dos jogadores"):
                st.dataframe(df_jog_imp.sort_values('Impacto_medio', ascending=False),
                             use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════
#  ABA 8 — VISÃO GERAL
# ══════════════════════════════════════════════
with aba8:
    st.subheader("Visão Geral do Torneio")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Distribuição de HS% por Função")
        fig_box = px.box(df, x='Funcao', y='HS%', color='Funcao',
                         color_discrete_map={
                             'Duelista': '#ff4655', 'Iniciador': '#1a9fff',
                             'Controlador': '#00c853', 'Sentinela': '#ffab00',
                         }, template=TEMPLATE)
        fig_box.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    with col2:
        st.markdown("#### Rating vs ACS por Jogador")
        scatter_data = (df.groupby(['Jogador', 'Time', 'Funcao'])
                        .agg(Rating=('R', 'mean'), ACS=('ACS', 'mean'),
                             Partidas=('R', 'count'))
                        .reset_index().query('Partidas >= 3'))
        fig_sc = px.scatter(
            scatter_data, x='ACS', y='Rating', color='Funcao',
            size='Partidas', hover_name='Jogador',
            hover_data=['Time', 'Partidas'], template=TEMPLATE,
            color_discrete_map={
                'Duelista': '#ff4655', 'Iniciador': '#1a9fff',
                'Controlador': '#00c853', 'Sentinela': '#ffab00',
            },
        )
        fig_sc.update_layout(height=400)
        st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("#### Top 5 Jogadores por K Médio")
    top_k = (df.groupby('Jogador')['K'].mean().sort_values(ascending=False)
               .head(5).round(2).reset_index())
    top_k.columns = ['Jogador', 'K Médio']
    fig_topk = px.bar(top_k[::-1], x='K Médio', y='Jogador', orientation='h',
                      color='K Médio', color_continuous_scale='RdYlGn',
                      template=TEMPLATE, text='K Médio',
                      title='Top 5 Jogadores por K Médio')
    fig_topk.update_traces(textposition='outside')
    fig_topk.update_layout(height=300, showlegend=False,
                           coloraxis_showscale=False, yaxis_title="")
    st.plotly_chart(fig_topk, use_container_width=True)