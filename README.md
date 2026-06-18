# Valorant VCT Analise

Sistema de análise e predição de partidas do **VCT (Valorant Champions Tour)** com coleta automatizada de dados, análises estatísticas, perfis de jogadores e modelo de Machine Learning — tudo integrado em um dashboard Streamlit interativo.

---

## Estrutura do Projeto

```
valorant-vct-analytics/
│
├── 0 - Codigo_de_Coleta.ipynb           # Etapa 1: Coleta de dados via web scraping
├── 1 - Análises_Valorant_VCT_s.ipynb    # Etapa 2: Análises gráficas exploratórias
├── 2 - Perfil_do_Jogador.ipynb          # Etapa 3: Perfil histórico de jogadores
├── 3 - Aplicação_de_ML.ipynb            # Etapa 4: Feature engineering e modelo ML
├── dashboard.py                         # Etapa 5: App Streamlit unificado
│
├── status_individual2.csv               # Dataset consolidado (todas as ligas)
├── status_individual_americas.csv       # Dataset VCT Americas
├── status_individual_emea.csv           # Dataset VCT EMEA
├── status_individual_pacifico.csv       # Dataset VCT Pacífico
└── status_individual_china2.csv         # Dataset VCT China
```

---

## Etapas do Projeto

### Etapa 0 — Coleta de Dados (`0 - Codigo_de_Coleta.ipynb`)

Realiza o **web scraping** do site [vlr.gg](https://www.vlr.gg), que é a principal fonte de estatísticas do cenário competitivo de Valorant.

**O que faz:**
- Acessa URLs de partidas da VCT a partir de uma planilha Google Sheets
- Integração com **Google Sheets** via API (`gspread`) para gerenciar URLs pendentes
- Marca URLs como `"ok"` ou `"erro"` após processar, permitindo retomada do processo
- Cobre as quatro ligas regionais: **Americas, EMEA, Pacífico e China**
- Extrai estatísticas individuais de cada jogador por mapa: `R`, `ACS`, `K`, `D`, `A`, `+/-`, `KAST`, `ADR`, `HS%`, `FK`, `FD`
- Limpa e formata os dados usando `BeautifulSoup` e `pandas`
- Salva o resultado em CSV para uso nas etapas seguintes

**Bibliotecas principais:** `requests`, `BeautifulSoup`, `pandas`, `gspread`, `google-auth`

---

### Etapa 1 — Análises Gráficas (`1 - Análises_Valorant_VCT_s.ipynb`)

Análise exploratória dos dados com **visualizações estilizadas**.

**Gráficos disponíveis:**

| Função | Descrição |
|---|---|
| `plot_jogos_por_mapa()` | Frequência de cada mapa jogado por liga |
| `plot_agentes()` | Agentes mais escolhidos globalmente ou por liga |
| `plot_top_jogadores()` | Ranking de jogadores por qualquer estatística (`R`, `ACS`, `K`…) |
| `plot_rounds_por_time()` | Rounds totais conquistados por time |

Todos os gráficos podem ser filtrados por liga (`VCT Americas`, `VCT China`, `VCT Emea`, `VCT Pacifico`) ou exibidos de forma global.

---

### Etapa 2 — Perfil do Jogador (`2 - Perfil_do_Jogador.ipynb`)

Gera um **relatório textual completo** sobre o histórico de um jogador específico.

**O que exibe:**
- Total de partidas, Win Rate geral e times em que atuou
- Médias de `R`, `ACS`, `KAST`, `ADR`, `HS%`
- Totais de `K`, `D`, `A`, `+/-`, `FK`, `FD`
- Desempenho **por mapa** (ordenado por ACS médio)
- Desempenho **por agente** (ordenado por ACS médio)

**Como usar:**
```python
perfil_historico('cortezia', df)                       # todos os dados
perfil_historico('cortezia', df, liga='VCT Americas')  # filtrado por liga
```

---

### Etapa 3 — Feature Engineering e Modelo ML (`3 - Aplicação_de_ML.ipynb`)

Construção do **modelo preditivo de resultados** de partidas por mapa.

#### 3.1 — Enriquecimento de Features

A partir do dataset bruto, são criadas as seguintes features:

| Feature | Descrição |
|---|---|
| `Role` | Função do agente (Duelista, Iniciador, Controlador, Sentinela) |
| `win_rate_jogador` | Win rate histórica do jogador |
| `win_rate_agente` | Win rate histórica do agente |
| `win_rate_mapa` | Win rate histórica do mapa |
| `win_rate_jogador_agente` | Win rate do jogador com aquele agente específico |
| `win_rate_jogador_mapa` | Win rate do jogador naquele mapa específico |
| `Impacto` | Score ponderado por role, somando todas as stats individuais |
| `Impacto_Time_%` | Participação percentual do jogador no impacto total do time |

#### 3.2 — Construção do Dataset de Confrontos

Os dados são pivotados de nível individual para nível de confronto entre dois times:
- **Time A** e **Time B** com seus 5 jogadores e agentes cada
- Features derivadas: `diff_wr_jogadores`, `diff_wr_agentes`, `media_wr_A/B`, `media_agente_A/B`
- Mapa codificado via One-Hot Encoding (`pd.get_dummies`)

#### 3.3 — Data Augmentation

Para lidar com desbalanceamento e aumentar robustez:
- Cada confronto é **espelhado** (A ↔ B com target invertido)
- Isso garante simetria e balanceamento 50/50 no conjunto de treino

#### 3.4 — Modelo Final

```
Pipeline:
  StandardScaler → LogisticRegression (max_iter=1000)
```

- Avaliado com `ROC-AUC`, `classification_report`, `brier_score_loss` e `log_loss`
- Treino/teste com split estratificado (80/20)

#### 3.5 — Função de Predição

```python
resultado = prever_confronto(
    mapa='Haven',
    jogadores_A=['Less', 'aspas', 'saadhak', 'tuyz', 'cauanzin'],
    agentes_A  =['Viper', 'Jett', 'Kayo', 'Brimstone', 'Skye'],
    jogadores_B=['crashies', 'Victor', 'ardiis', 'FiNESSE', 's0m'],
    agentes_B  =['Sova', 'Kayo', 'Jett', 'Killjoy', 'Omen'],
)
# Retorna: prob_A, prob_B, vencedor
```

---

### Etapa 4 — Dashboard Streamlit (`dashboard.py`)

App interativo que **unifica todas as etapas anteriores** em uma interface web com tema Valorant.

**Páginas disponíveis (sidebar):**

| Página | Funcionalidade |
|---|---|
| Perfil do Jogador | Busca histórico completo de qualquer jogador, com médias, totais e desempenho por mapa/agente |
| Análises Gráficas | Gráficos interativos filtráveis por liga: mapas mais jogados, agentes, ranking de jogadores, rounds por time |
| Predição ML | Seleciona dois times (jogadores + agentes + mapa) e exibe a probabilidade de vitória de cada lado |

**Destaques técnicos:**
- Dataset carregado com `@st.cache_data` para performance
- Modelo treinado via `@st.cache_resource` (treinado uma vez, reutilizado em todas as predições)
- Auto-preenchimento de stats históricas dos jogadores ao selecionar na página de predição
- Tema dark customizado via CSS inline com as cores oficiais do Valorant (`#ff4655`, `#1a1a2e`)

**Como rodar:**
```bash
streamlit run dashboard.py
```

---

## Instalação

```bash
pip install pandas numpy matplotlib scikit-learn streamlit requests beautifulsoup4 gspread google-auth
```

---

## Dados

Os dados são coletados do [vlr.gg](https://www.vlr.gg) e cobrem partidas das quatro ligas regionais do VCT:

- **VCT Americas**
- **VCT EMEA**
- **VCT Pacífico**
- **VCT China**

Cada linha do dataset representa a performance de **um jogador em um mapa** de uma partida oficial.

---

## Observações

- O scraper foi projetado para URLs do vlr.gg no formato `/ID/time-a-vs-time-b-evento`
- A coleta automatizada requer um arquivo de credenciais do Google (`*.json`) para acesso à API do Google Sheets
- O modelo prediz a **probabilidade de vitória por mapa**, não o resultado geral da série (Bo3/Bo5)
- Win rates suavizadas com fator `K=10` são usadas nas features para evitar overfitting em jogadores com poucas partidas
