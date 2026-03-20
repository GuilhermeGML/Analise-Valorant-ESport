"""
dashboard.py
============
Streamlit dashboard interativo para análise de stats VCT 2026.

Instalar:  pip install streamlit plotly pandas
Rodar:     streamlit run dashboard.py
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────
#  CONFIGURAÇÃO DA PÁGINA
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="VCT 2026 Americas Dashboard",
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
    stats  = pd.read_csv("vlr_stats.csv")
    placar = pd.read_csv("vlr_placar.csv")

    # Classifica agentes por função
    funcoes = {
        'Duelista':    ['Jett', 'Reyna', 'Raze', 'Yoru', 'Neon', 'Iso', 'Waylay'],
        'Iniciador':   ['Sova', 'Breach', 'Skye', 'Kayo', 'Fade', 'Gekko', 'Tejo'],
        'Controlador': ['Brimstone', 'Viper', 'Omen', 'Astra', 'Harbor', 'Clove'],
        'Sentinela':   ['Killjoy', 'Cypher', 'Sage', 'Chamber', 'Deadlock', 'Vyse'],
    }
    agente_funcao = {a: f for f, agents in funcoes.items() for a in agents}
    stats['Funcao'] = stats['Agente'].map(agente_funcao).fillna('Outro')

    return stats, placar

stats, placar = carregar_dados()
df            = stats[stats['Mapa'] != 'AllMaps']
df_all        = stats[stats['Mapa'] == 'AllMaps']
df_placar     = placar[
    (placar['Mapa'] == 'AllMaps') &
    placar['Mapas_T1'].notna() &
    placar['Mapas_T2'].notna()
].drop_duplicates(subset='URL', keep='last')
df_mapas_ind  = placar[placar['Mapa'] != 'AllMaps']

MAPAS    = sorted(df['Mapa'].dropna().unique())
TIMES    = sorted(pd.concat([df['Time']]).dropna().unique())
JOGADORES= sorted(df['Jogador'].dropna().unique())
AGENTES  = sorted(df['Agente'].dropna().unique())
STATS    = ['R', 'ACS', 'K', 'D', 'A', '+/-', 'KAST', 'ADR', 'HS%', 'FK', 'FD']

TEMPLATE = "plotly_dark"
COR_PRIN = "#ff4655"

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────

st.title("🎯 VCT 2026 Americas Kickoff")
st.markdown("---")

# ─────────────────────────────────────────────
#  MÉTRICAS GERAIS NO TOPO
# ─────────────────────────────────────────────

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total de Jogos",    df_placar['URL'].nunique())
c2.metric("Mapas Jogados",     len(df_mapas_ind))
c3.metric("Times",             df['Time'].nunique())
c4.metric("Jogadores",         df['Jogador'].nunique())
c5.metric("Agentes Usados",    df['Agente'].nunique())

st.markdown("---")

# ─────────────────────────────────────────────
#  ABAS PRINCIPAIS
# ─────────────────────────────────────────────

aba1, aba2, aba3, aba4, aba5, aba6 = st.tabs([
    "📊 Ranking de Stats",
    "🗺️ Análise por Mapa",
    "👤 Perfil do Jogador",
    "🧬 Análise por Agente",
    "🏆 Resultados",
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

    # Filtros opcionais
    with st.expander("Filtros opcionais"):
        f_time  = st.multiselect("Filtrar por Time",  TIMES)
        f_mapa  = st.multiselect("Filtrar por Mapa",  MAPAS)
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
            ranking[::-1],
            x=stat_sel, y='Jogador',
            orientation='h',
            color=stat_sel,
            color_continuous_scale='RdYlGn',
            text=ranking[stat_sel][::-1].round(2),
            hover_data=['Time', 'Partidas'],
            template=TEMPLATE,
            title=f"Top {top_n} — {stat_sel} ({agregacao})",
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            height=max(400, top_n * 35),
            showlegend=False,
            coloraxis_showscale=False,
            yaxis_title="",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            ranking.reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )


# ══════════════════════════════════════════════
#  ABA 2 — ANÁLISE POR MAPA
# ══════════════════════════════════════════════
with aba2:
    st.subheader("Análise por Mapa")

    mapa_sel = st.selectbox("Selecione o Mapa", MAPAS, key="mapa_aba2")
    df_mapa  = df[df['Mapa'] == mapa_sel]

    col1, col2 = st.columns(2)

    # Média de stats no mapa
    with col1:
        st.markdown(f"#### Médias em {mapa_sel}")
        medias = df_mapa[STATS].mean().round(2).reset_index()
        medias.columns = ['Estatística', 'Média']

        fig = px.bar(
            medias, x='Média', y='Estatística',
            orientation='h',
            color='Média',
            color_continuous_scale='Blues',
            template=TEMPLATE,
            text='Média',
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            height=400, showlegend=False,
            coloraxis_showscale=False, yaxis_title=""
        )
        st.plotly_chart(fig, use_container_width=True)

    # Melhores jogadores no mapa
    with col2:
        st.markdown(f"#### Melhores Jogadores em {mapa_sel}")
        stat_mapa = st.selectbox("Stat", STATS, key="stat_mapa")

        top_mapa = (df_mapa.groupby(['Jogador', 'Time'])[stat_mapa]
                    .mean()
                    .round(2)
                    .reset_index()
                    .sort_values(stat_mapa, ascending=False)
                    .head(10))

        fig2 = px.bar(
            top_mapa[::-1],
            x=stat_mapa, y='Jogador',
            orientation='h',
            color=stat_mapa,
            color_continuous_scale='RdYlGn',
            hover_data=['Time'],
            template=TEMPLATE,
            text=top_mapa[stat_mapa][::-1],
        )
        fig2.update_traces(textposition='outside')
        fig2.update_layout(
            height=400, showlegend=False,
            coloraxis_showscale=False, yaxis_title=""
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Frequência de agentes no mapa
    st.markdown(f"#### Agentes Mais Usados em {mapa_sel}")
    agentes_mapa = df_mapa['Agente'].value_counts().reset_index()
    agentes_mapa.columns = ['Agente', 'Vezes']

    fig3 = px.bar(
        agentes_mapa, x='Agente', y='Vezes',
        color='Vezes', color_continuous_scale='Reds',
        template=TEMPLATE, text='Vezes',
    )
    fig3.update_traces(textposition='outside')
    fig3.update_layout(height=350, showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════
#  ABA 3 — PERFIL DO JOGADOR
# ══════════════════════════════════════════════
with aba3:
    st.subheader("Perfil do Jogador")

    col1, col2 = st.columns([2, 2])
    with col1:
        jogador_sel = st.selectbox("Jogador", JOGADORES)
    with col2:
        mapa_jogador = st.selectbox(
            "Mapa (opcional)",
            ["Todos"] + MAPAS,
            key="mapa_jogador"
        )

    df_jog = df[df['Jogador'] == jogador_sel]
    if mapa_jogador != "Todos":
        df_jog = df_jog[df_jog['Mapa'] == mapa_jogador]

    if df_jog.empty:
        st.warning("Sem dados para esse jogador/mapa.")
    else:
        # Métricas rápidas
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Partidas",    len(df_jog))
        m2.metric("Rating Médio", f"{df_jog['R'].mean():.2f}")
        m3.metric("ACS Médio",   f"{df_jog['ACS'].mean():.0f}")
        m4.metric("K/D",         f"{df_jog['K'].sum() / max(df_jog['D'].sum(), 1):.2f}")
        m5.metric("KAST Médio",  f"{df_jog['KAST'].mean():.0f}%")

        col1, col2 = st.columns(2)

        # Radar de stats
        with col1:
            st.markdown("#### Radar de Performance")
            cats   = ['R', 'ACS', 'KAST', 'ADR', 'HS%']
            medias = df_jog[cats].mean().round(2)

            # Normaliza pra 0-100 para o radar
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
                paper_bgcolor='#0e0e1a',
                font_color='white',
                height=350,
                showlegend=False,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # Stats por mapa
        with col2:
            st.markdown("#### Rating por Mapa")
            rating_mapa = (df[df['Jogador'] == jogador_sel]
                           .groupby('Mapa')['R']
                           .mean()
                           .round(2)
                           .sort_values(ascending=False)
                           .reset_index())

            fig_rm = px.bar(
                rating_mapa, x='Mapa', y='R',
                color='R', color_continuous_scale='RdYlGn',
                template=TEMPLATE, text='R',
            )
            fig_rm.update_traces(textposition='outside')
            fig_rm.update_layout(
                height=350, showlegend=False,
                coloraxis_showscale=False,
                yaxis_title="Rating Médio"
            )
            st.plotly_chart(fig_rm, use_container_width=True)

        # Agentes usados
        st.markdown("#### Agentes Mais Usados")
        agentes_jog = (df[df['Jogador'] == jogador_sel]['Agente']
                       .value_counts()
                       .reset_index())
        agentes_jog.columns = ['Agente', 'Vezes']

        fig_ag = px.bar(
            agentes_jog, x='Agente', y='Vezes',
            color='Vezes', color_continuous_scale='Blues',
            template=TEMPLATE, text='Vezes',
        )
        fig_ag.update_traces(textposition='outside')
        fig_ag.update_layout(
            height=300, showlegend=False, coloraxis_showscale=False
        )
        st.plotly_chart(fig_ag, use_container_width=True)

        # Tabela detalhada
        with st.expander("Ver tabela completa"):
            st.dataframe(
                df_jog[['Mapa', 'Agente', 'R', 'ACS', 'K', 'D', 'A',
                         '+/-', 'KAST', 'ADR', 'HS%', 'FK', 'FD']],
                use_container_width=True,
                hide_index=True,
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
                .agg(['mean', 'count'])
                .reset_index()
                .rename(columns={'mean': stat_agente, 'count': 'Picks'})
                .query(f'Picks >= {min_pick}')
                .sort_values(stat_agente, ascending=False))

    fig_ag = px.bar(
        stats_ag[::-1],
        x=stat_agente, y='Agente',
        orientation='h',
        color='Funcao',
        color_discrete_map={
            'Duelista':    '#ff4655',
            'Iniciador':   '#1a9fff',
            'Controlador': '#00c853',
            'Sentinela':   '#ffab00',
            'Outro':       '#888888',
        },
        hover_data=['Picks'],
        template=TEMPLATE,
        text=stats_ag[stat_agente][::-1].round(2),
        title=f"{stat_agente} médio por Agente",
    )
    fig_ag.update_traces(textposition='outside')
    fig_ag.update_layout(
        height=max(400, len(stats_ag) * 30),
        yaxis_title="", legend_title="Função",
    )
    st.plotly_chart(fig_ag, use_container_width=True)

    # Impacto por função
    st.markdown("#### Impacto por Função")
    impacto = (df.groupby('Funcao')[['R', 'ACS', 'ADR', 'KAST']]
               .mean()
               .round(2)
               .reset_index())

    fig_func = px.bar(
        impacto.melt(id_vars='Funcao'),
        x='Funcao', y='value', color='variable',
        barmode='group',
        template=TEMPLATE,
        color_discrete_sequence=['#ff4655', '#1a9fff', '#00c853', '#ffab00'],
        labels={'value': 'Média', 'variable': 'Stat'},
        title="Médias por Função",
    )
    fig_func.update_layout(height=400)
    st.plotly_chart(fig_func, use_container_width=True)


# ══════════════════════════════════════════════
#  ABA 5 — RESULTADOS
# ══════════════════════════════════════════════
with aba5:
    st.subheader("Resultados dos Jogos")

    col1, col2 = st.columns(2)

    # Vitórias por time
    with col1:
        st.markdown("#### Vitórias por Time")
        vitorias_t1 = df_placar.groupby('Time1')['Mapas_T1'].sum()
        vitorias_t2 = df_placar.groupby('Time2')['Mapas_T2'].sum()
        vitorias    = (vitorias_t1.add(vitorias_t2, fill_value=0)
                       .astype(int)
                       .sort_values(ascending=False)
                       .reset_index())
        vitorias.columns = ['Time', 'Mapas Ganhos']

        fig_v = px.bar(
            vitorias[::-1],
            x='Mapas Ganhos', y='Time',
            orientation='h',
            color='Mapas Ganhos',
            color_continuous_scale='RdYlGn',
            template=TEMPLATE,
            text='Mapas Ganhos',
        )
        fig_v.update_traces(textposition='outside')
        fig_v.update_layout(
            height=400, showlegend=False,
            coloraxis_showscale=False, yaxis_title=""
        )
        st.plotly_chart(fig_v, use_container_width=True)

    # Rounds totais por time
    with col2:
        st.markdown("#### Rounds Totais por Time")
        score_t1 = df_placar.groupby('Time1')['Score_T1'].sum()
        score_t2 = df_placar.groupby('Time2')['Score_T2'].sum()
        rounds   = (score_t1.add(score_t2, fill_value=0)
                    .astype(int)
                    .sort_values(ascending=False)
                    .reset_index())
        rounds.columns = ['Time', 'Rounds']

        fig_r = px.bar(
            rounds[::-1],
            x='Rounds', y='Time',
            orientation='h',
            color='Rounds',
            color_continuous_scale='Blues',
            template=TEMPLATE,
            text='Rounds',
        )
        fig_r.update_traces(textposition='outside')
        fig_r.update_layout(
            height=400, showlegend=False,
            coloraxis_showscale=False, yaxis_title=""
        )
        st.plotly_chart(fig_r, use_container_width=True)

    # Jogos por mapa
    st.markdown("#### Jogos por Mapa")
    jogos_mapa = (df_mapas_ind['Mapa']
                  .value_counts()
                  .reset_index())
    jogos_mapa.columns = ['Mapa', 'Jogos']

    fig_jm = px.bar(
        jogos_mapa, x='Mapa', y='Jogos',
        color='Jogos', color_continuous_scale='Reds',
        template=TEMPLATE, text='Jogos',
        title="Frequência de cada Mapa",
    )
    fig_jm.update_traces(textposition='outside')
    fig_jm.update_layout(
        height=350, showlegend=False, coloraxis_showscale=False
    )
    st.plotly_chart(fig_jm, use_container_width=True)


# ══════════════════════════════════════════════
#  ABA 6 — VISÃO GERAL
# ══════════════════════════════════════════════
with aba6:
    st.subheader("Visão Geral do Torneio")

    col1, col2 = st.columns(2)

    # Distribuição de HS% por função
    with col1:
        st.markdown("#### Distribuição de HS% por Função")
        fig_box = px.box(
            df, x='Funcao', y='HS%',
            color='Funcao',
            color_discrete_map={
                'Duelista':    '#ff4655',
                'Iniciador':   '#1a9fff',
                'Controlador': '#00c853',
                'Sentinela':   '#ffab00',
            },
            template=TEMPLATE,
        )
        fig_box.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    # Scatter Rating x ACS
    with col2:
        st.markdown("#### Rating vs ACS por Jogador")
        scatter_data = (df.groupby(['Jogador', 'Time', 'Funcao'])
                        .agg(Rating=('R', 'mean'), ACS=('ACS', 'mean'),
                             Partidas=('R', 'count'))
                        .reset_index()
                        .query('Partidas >= 3'))

        fig_sc = px.scatter(
            scatter_data,
            x='ACS', y='Rating',
            color='Funcao',
            size='Partidas',
            hover_name='Jogador',
            hover_data=['Time', 'Partidas'],
            template=TEMPLATE,
            color_discrete_map={
                'Duelista':    '#ff4655',
                'Iniciador':   '#1a9fff',
                'Controlador': '#00c853',
                'Sentinela':   '#ffab00',
            },
        )
        fig_sc.update_layout(height=400)
        st.plotly_chart(fig_sc, use_container_width=True)

    # Média de kills por mapa
    st.markdown("#### Média de Kills por Mapa")
    kills_mapa = (df.groupby('Mapa')['K']
                  .mean()
                  .round(2)
                  .sort_values(ascending=False)
                  .reset_index())
    kills_mapa.columns = ['Mapa', 'Média de Kills']

    fig_km = px.bar(
        kills_mapa, x='Mapa', y='Média de Kills',
        color='Média de Kills',
        color_continuous_scale='YlOrRd',
        template=TEMPLATE,
        text='Média de Kills',
        title="Média de Kills por Mapa",
    )
    fig_km.update_traces(textposition='outside')
    fig_km.update_layout(
        height=350, showlegend=False, coloraxis_showscale=False
    )
    st.plotly_chart(fig_km, use_container_width=True)