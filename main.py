import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing, load_diabetes, load_wine, load_breast_cancer, fetch_openml, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from mpl_toolkits.mplot3d import Axes3D


# Configuração da página
st.set_page_config(
    page_title="Educação Machine Learning",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.streamlit.io/community',
        'Report a bug': "https://github.com/LoowdY/educacao_machine_learning",
        'About': "# Aplicativo de Análise de Regressão\n"
                 "Este aplicativo foi desenvolvido para explorar modelos de regressão "
                 "usando conjuntos de dados éticos e técnicas avançadas de análise."
    }
)

# # Função para aplicar estilo CSS personalizado
# def aplicar_estilo():
#     st.markdown("""
#     <style>
#     .main {
#         background-color: #f0f2f6;
#     }
#     .stButton>button {
#         background-color: #4CAF50;
#         color: white;
#     }
#     .stSelectbox {
#         background-color: #e1e5eb;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# Aplicar estilo personalizado
# aplicar_estilo()

# Função para calcular RMSE
def raiz_erro_quadratico_medio(y_verdadeiro, y_previsto):
    return np.sqrt(mean_squared_error(y_verdadeiro, y_previsto))

# Função para carregar os dados com cache
@st.cache_data
def carregar_dados(nome_conjunto_dados):
    if nome_conjunto_dados == "Habitação na Califórnia":
        dados = fetch_california_housing()
        X = pd.DataFrame(dados.data, columns=dados.feature_names)
        y = pd.Series(dados.target, name='alvo')
        descricao = """
        Este conjunto de dados contém informações sobre habitação na Califórnia. 
        Ele inclui métricas como renda média, idade da casa, número médio de cômodos, 
        entre outros. O objetivo é prever o valor médio das casas em um bloco (medido em centenas de milhares de dólares).

        Características:
        - MedInc: renda mediana da população do bloco
        - HouseAge: idade mediana das casas no bloco
        - AveRooms: número médio de cômodos por residência
        - AveBedrms: número médio de quartos por residência
        - Population: população do bloco
        - AveOccup: número médio de ocupantes por residência
        - Latitude: latitude do bloco
        - Longitude: longitude do bloco

        Alvo:
        - Valor médio das casas para o bloco (em centenas de milhares de dólares)
        """
    elif nome_conjunto_dados == "Diabetes":
        dados = load_diabetes()
        X = pd.DataFrame(dados.data, columns=dados.feature_names)
        y = pd.Series(dados.target, name='alvo')
        descricao = """
        Este conjunto de dados contém informações sobre pacientes com diabetes. 
        Ele inclui várias medidas fisiológicas e um indicador quantitativo de 
        progressão da doença um ano após a linha de base.

        Características:
        - age: idade
        - sex: sexo
        - bmi: índice de massa corporal
        - bp: pressão arterial média
        - s1-s6: seis medidas sanguíneas

        Alvo:
        - Medida quantitativa da progressão da doença um ano após a linha de base
        """
    elif nome_conjunto_dados == "Vinho":
        dados = load_wine()
        X = pd.DataFrame(dados.data, columns=dados.feature_names)
        y = pd.Series(dados.target, name='alvo')
        descricao = """
        Este conjunto de dados contém informações sobre diferentes tipos de vinho. 
        Ele inclui várias medidas químicas que podem ser usadas para determinar 
        a origem do vinho.

        Características:
        - alcohol: teor alcoólico
        - malic_acid: ácido málico
        - ash: cinzas
        - alcalinity_of_ash: alcalinidade das cinzas
        - magnesium: magnésio
        - total_phenols: fenóis totais
        - flavanoids: flavonoides
        - nonflavanoid_phenols: fenóis não flavonoides
        - proanthocyanins: proantocianinas
        - color_intensity: intensidade da cor
        - hue: tonalidade
        - od280/od315_of_diluted_wines: OD280/OD315 de vinhos diluídos
        - proline: prolina

        Alvo:
        - Classe do vinho (0, 1, 2)
        """
    elif nome_conjunto_dados == "Câncer de Mama":
        dados = load_breast_cancer()
        X = pd.DataFrame(dados.data, columns=dados.feature_names)
        y = pd.Series(dados.target, name='alvo')
        descricao = """
        Este conjunto de dados contém informações sobre características de células 
        de câncer de mama. Ele pode ser usado para prever se um tumor é maligno ou benigno.

        Características:
        - mean radius: raio médio
        - mean texture: textura média
        - mean perimeter: perímetro médio
        - mean area: área média
        - mean smoothness: suavidade média
        ... (e outras características similares)

        Alvo:
        - Diagnóstico (0 = maligno, 1 = benigno)
        """
    else:  # Habitação em Ames
        dados = fetch_openml(name="house_prices", as_frame=True)
        X = dados.data
        y = dados.target
        descricao = """
        Este conjunto de dados contém informações detalhadas sobre casas em Ames, Iowa. 
        Ele inclui uma ampla variedade de características que podem influenciar o preço de venda.

        Características:
        - LotArea: Tamanho do lote em pés quadrados
        - OverallQual: Avaliação geral do material e acabamento da casa
        - YearBuilt: Ano de construção original
        - TotalBsmtSF: Área total do porão em pés quadrados
        ... (e muitas outras características)

        Alvo:
        - SalePrice: Preço de venda da propriedade em dólares
        """
    
    return X, y, descricao

# Funções de visualização
def plotar_dispersao(X, y, caracteristica):
    fig = px.scatter(x=X[caracteristica], y=y, labels={'x': caracteristica, 'y': 'Alvo'})
    fig.update_layout(title=f"Relação entre {caracteristica} e Alvo")
    return fig

def plotar_importancia_caracteristicas(modelo, nomes_caracteristicas, X_teste, y_teste):
    if hasattr(modelo, 'feature_importances_'):
        importancias = pd.DataFrame({'caracteristica': nomes_caracteristicas, 'importancia': modelo.feature_importances_})
    else:
        resultado = permutation_importance(modelo, X_teste, y_teste, n_repeats=10, random_state=42)
        importancias = pd.DataFrame({'caracteristica': nomes_caracteristicas, 'importancia': resultado.importances_mean})
    
    importancias = importancias.sort_values('importancia', ascending=False)
    fig = px.bar(importancias, x='importancia', y='caracteristica', orientation='h',
                 title='Importância das Características')
    return fig

def plotar_residuos(y_teste, y_previsto):
    residuos = y_teste - y_previsto
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_teste, y=residuos, mode='markers'))
    fig.update_layout(title='Gráfico de Resíduos',
                      xaxis_title='Valores Reais',
                      yaxis_title='Resíduos')
    return fig

# Funções para gerar gráficos educativos
def plot_linear_regression():
    X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
    model = LinearRegression()
    model.fit(X, y)
    
    fig, ax = plt.subplots()
    ax.scatter(X, y, alpha=0.5)
    ax.plot(X, model.predict(X), color='r')
    ax.set_title('Regressão Linear')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    return fig

def plot_decision_tree():
    X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
    model = DecisionTreeRegressor(max_depth=3)
    model.fit(X, y)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_tree(model, filled=True, feature_names=['X'], ax=ax)
    ax.set_title('Árvore de Decisão')
    return fig

def plot_random_forest():
    X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
    model = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)
    
    fig, ax = plt.subplots()
    ax.scatter(X, y, alpha=0.5)
    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred = model.predict(x_range)
    ax.plot(x_range, y_pred, color='r', label='Random Forest')
    ax.set_title('Random Forest')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.legend()
    return fig

def plot_svr():
    X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
    model = SVR(kernel='rbf')
    model.fit(X, y)
    
    fig, ax = plt.subplots()
    ax.scatter(X, y, alpha=0.5)
    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred = model.predict(x_range)
    ax.plot(x_range, y_pred, color='r', label='SVR')
    ax.set_title('Support Vector Regression')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.legend()
    return fig

def plot_knn():
    X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X, y)
    
    fig, ax = plt.subplots()
    ax.scatter(X, y, alpha=0.5)
    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred = model.predict(x_range)
    ax.plot(x_range, y_pred, color='r', label='KNN')
    ax.set_title('K-Nearest Neighbors')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.legend()
    return fig

# Título da aplicação
st.title("📊 Aplicação Avançada de Modelos de Regressão Éticos")
st.latex(r'''
\text{Esta aplicação permite explorar diferentes conjuntos de dados éticos e modelos de regressão de forma interativa.} \\
\text{Você pode selecionar o conjunto de dados, o modelo, seus parâmetros e a variável resposta.}
''')

# Barra lateral: Seleção de conjunto de dados
st.sidebar.title("🛠️ Configurações")
nome_conjunto_dados = st.sidebar.selectbox(
    "Escolha o Conjunto de Dados", 
    ("Habitação na Califórnia", "Diabetes", "Vinho", "Câncer de Mama", "Habitação em Ames")
)

# Carregar os dados
X, y, descricao_conjunto_dados = carregar_dados(nome_conjunto_dados)

# Exibir descrição do conjunto de dados
with st.expander("📌 Descrição do Conjunto de Dados"):
    st.text(descricao_conjunto_dados)

# Seleção da variável resposta
if nome_conjunto_dados != "Habitação em Ames":
    variavel_alvo = st.sidebar.selectbox("Escolha a variável resposta", X.columns.tolist() + ['alvo'])
    if variavel_alvo != 'alvo':
        y = X[variavel_alvo]
        X = X.drop(columns=[variavel_alvo])
else:
    st.sidebar.info("Para o conjunto de dados Habitação em Ames, a variável resposta é fixa como 'price'.")

# Exibir uma amostra dos dados
st.subheader("📋 Amostra dos Dados")
st.dataframe(pd.concat([X, y], axis=1).head())

# Visualização exploratória
st.subheader("🔍 Visualização Exploratória")
col1, col2 = st.columns(2)
with col1:
    caracteristica_para_plotar = st.selectbox("Selecione uma característica para visualizar:", X.columns)
    st.plotly_chart(plotar_dispersao(X, y, caracteristica_para_plotar), use_container_width=True)
with col2:
    st.write("### Matriz de Correlação")
    correlacao = pd.concat([X, y], axis=1).corr()
    fig = px.imshow(correlacao, text_auto=True, aspect="auto")
    st.plotly_chart(fig, use_container_width=True)

# Dividir os dados em treino e teste
tamanho_teste = st.sidebar.slider("Tamanho do Conjunto de Teste (%)", 10, 50, 20) / 100
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=tamanho_teste, random_state=42)

# Normalização dos dados
escalonador = StandardScaler()
X_treino_escalonado = escalonador.fit_transform(X_treino)
X_teste_escalonado = escalonador.transform(X_teste)

# Barra lateral: Seleção de modelo
nome_modelo = st.sidebar.selectbox("Escolha o Modelo", 
                                  ("Floresta Aleatória", "Aumento de Gradiente", "XGBoost", "LightGBM",
                                   "Regressão Linear", "Ridge", "Lasso", "Rede Elástica",
                                   "Árvore de Decisão", "SVR", "K-Vizinhos"))

# Definir os hiperparâmetros do modelo
if nome_modelo == "Floresta Aleatória":
    n_estimadores = st.sidebar.slider("Número de Árvores", 10, 200, 100)
    profundidade_maxima = st.sidebar.slider("Profundidade Máxima", 1, 20, 10)
    amostras_minimas_divisao = st.sidebar.slider("Mínimo de Amostras para Divisão", 2, 10, 2)
    modelo = RandomForestRegressor(n_estimators=n_estimadores, max_depth=profundidade_maxima, 
                                  min_samples_split=amostras_minimas_divisao, random_state=42)
elif nome_modelo == "Aumento de Gradiente":
    n_estimadores = st.sidebar.slider("Número de Estimadores", 50, 500, 100)
    taxa_aprendizagem = st.sidebar.slider("Taxa de Aprendizagem", 0.01, 0.3, 0.1)
    profundidade_maxima = st.sidebar.slider("Profundidade Máxima", 1, 10, 3)
    modelo = GradientBoostingRegressor(n_estimators=n_estimadores, learning_rate=taxa_aprendizagem, 
                                      max_depth=profundidade_maxima, random_state=42)
elif nome_modelo == "XGBoost":
    n_estimadores = st.sidebar.slider("Número de Estimadores", 50, 500, 100)
    taxa_aprendizagem = st.sidebar.slider("Taxa de Aprendizagem", 0.01, 0.3, 0.1)
    profundidade_maxima = st.sidebar.slider("Profundidade Máxima", 1, 10, 3)
    modelo = XGBRegressor(n_estimators=n_estimadores, learning_rate=taxa_aprendizagem, 
                         max_depth=profundidade_maxima, random_state=42)
elif nome_modelo == "LightGBM":
    n_estimadores = st.sidebar.slider("Número de Estimadores", 50, 500, 100)
    taxa_aprendizagem = st.sidebar.slider("Taxa de Aprendizagem", 0.01, 0.3, 0.1)
    profundidade_maxima = st.sidebar.slider("Profundidade Máxima", 1, 10, 3)
    modelo = LGBMRegressor(n_estimators=n_estimadores, learning_rate=taxa_aprendizagem, 
                          max_depth=profundidade_maxima, random_state=42)
elif nome_modelo == "Regressão Linear":
    modelo = LinearRegression()
elif nome_modelo == "Ridge":
    alfa = st.sidebar.slider("Alfa", 0.1, 10.0, 1.0)
    modelo = Ridge(alpha=alfa, random_state=42)
elif nome_modelo == "Lasso":
    alfa = st.sidebar.slider("Alfa", 0.1, 10.0, 1.0)
    modelo = Lasso(alpha=alfa, random_state=42)
elif nome_modelo == "Rede Elástica":
    alfa = st.sidebar.slider("Alfa", 0.1, 10.0, 1.0)
    razao_l1 = st.sidebar.slider("Razão L1", 0.0, 1.0, 0.5)
    modelo = ElasticNet(alpha=alfa, l1_ratio=razao_l1, random_state=42)
elif nome_modelo == "Árvore de Decisão":
    profundidade_maxima = st.sidebar.slider("Profundidade Máxima", 1, 20, 5)
    amostras_minimas_divisao = st.sidebar.slider("Mínimo de Amostras para Divisão", 2, 10, 2)
    modelo = DecisionTreeRegressor(max_depth=profundidade_maxima, min_samples_split=amostras_minimas_divisao, random_state=42)
elif nome_modelo == "SVR":
    C = st.sidebar.slider("C", 0.1, 10.0, 1.0)
    epsilon = st.sidebar.slider("Epsilon", 0.01, 1.0, 0.1)
    modelo = SVR(C=C, epsilon=epsilon)
else:  # K-Vizinhos
    n_vizinhos = st.sidebar.slider("Número de Vizinhos", 1, 20, 5)
    modelo = KNeighborsRegressor(n_neighbors=n_vizinhos)

# Treinar o modelo
modelo.fit(X_treino_escalonado, y_treino)
y_previsto = modelo.predict(X_teste_escalonado)

# Painel de Métricas
st.subheader("📊 Painel de Métricas")
col1, col2, col3, col4 = st.columns(4)
col1.metric("EAM", f"{mean_absolute_error(y_teste, y_previsto):.4f}")
col2.metric("EQM", f"{mean_squared_error(y_teste, y_previsto):.4f}")
col3.metric("REQM", f"{raiz_erro_quadratico_medio(y_teste, y_previsto):.4f}")
col4.metric("R²", f"{r2_score(y_teste, y_previsto):.4f}")

# Gráfico de comparação entre valores reais e previstos
st.subheader("📈 Comparação: Valores Reais vs. Previstos")
fig = px.scatter(x=y_teste, y=y_previsto, labels={'x': 'Valores Reais', 'y': 'Valores Previstos'})
fig.add_trace(go.Scatter(x=[y_teste.min(), y_teste.max()], y=[y_teste.min(), y_teste.max()],
                         mode='lines', name='Linha de Referência'))
st.plotly_chart(fig, use_container_width=True)

# Gráfico de importância das características
st.subheader("🏆 Importância das Características")
st.plotly_chart(plotar_importancia_caracteristicas(modelo, X.columns, X_teste_escalonado, y_teste), use_container_width=True)

# Gráfico de resíduos
st.subheader("🎯 Análise de Resíduos")
st.plotly_chart(plotar_residuos(y_teste, y_previsto), use_container_width=True)

# Seção Educativa Expandida
st.subheader("📚 Seção Educativa")

# Explicação das Métricas
with st.expander("Entenda as Métricas de Avaliação"):
    st.latex(r'''
    \textbf{Erro Absoluto Médio (EAM):} \\
    EAM = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| \\
    \text{Interpretação: Quanto menor, melhor. Indica a magnitude média do erro.} \\
    
    \textbf{Erro Quadrático Médio (EQM):} \\
    EQM = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \\
    \text{Interpretação: Penaliza erros maiores. Quanto menor, melhor.} \\
    
    \textbf{Raiz do Erro Quadrático Médio (REQM):} \\
    REQM = \sqrt{EQM} \\
    \text{Interpretação: Na mesma unidade da variável alvo. Menor é melhor.} \\
    
    \textbf{Coeficiente de Determinação (R²):} \\
    R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2} \\
    \text{Interpretação: Varia de 0 a 1. Quanto mais próximo de 1, melhor o modelo explica os dados.}
    ''')

# Explicação dos Algoritmos com Gráficos
with st.expander("Entenda os Algoritmos de Regressão"):
    st.markdown("""
    ### Algoritmos de Regressão

    1. **Regressão Linear**
       - **O que é**: Modelo que assume uma relação linear entre as variáveis independentes e a variável dependente.
       - **Como funciona**: Encontra a melhor linha reta que minimiza a soma dos quadrados dos resíduos.
    """)
    st.pyplot(plot_linear_regression())

    st.markdown("""
    2. **Floresta Aleatória (Random Forest)**
       - **O que é**: Ensemble de árvores de decisão.
       - **Como funciona**: Cria múltiplas árvores de decisão e combina suas previsões.
    """)
    st.pyplot(plot_random_forest())

    st.markdown("""
    3. **Árvore de Decisão**
       - **O que é**: Modelo que divide os dados em subgrupos baseados em regras de decisão.
       - **Como funciona**: Cria uma estrutura de árvore com nós de decisão e folhas de previsão.
    """)
    st.pyplot(plot_decision_tree())

    st.markdown("""
    4. **SVR (Support Vector Regression)**
       - **O que é**: Extensão do SVM para problemas de regressão.
       - **Como funciona**: Busca encontrar um hiperplano que melhor se ajusta aos dados dentro de uma margem de tolerância.
    """)
    st.pyplot(plot_svr())

    st.markdown("""
    5. **K-Vizinhos (K-Nearest Neighbors)**
       - **O que é**: Método baseado em instância que usa os vizinhos mais próximos para previsão.
       - **Como funciona**: Prevê baseado na média dos valores dos K vizinhos mais próximos.
    """)
    st.pyplot(plot_knn())

# Dicas para Interpretação dos Gráficos
with st.expander("Dicas para Interpretação dos Gráficos"):
    st.markdown("""
    1. **Gráfico de Dispersão (Característica vs. Alvo):**
       - Ajuda a identificar relações lineares ou não lineares entre características e o alvo.
       - Padrões claros indicam forte relação; dispersão aleatória sugere fraca relação.

    2. **Matriz de Correlação:**
       - Cores mais intensas (vermelhas ou azuis) indicam correlações mais fortes.
       - Correlações próximas a 1 ou -1 sugerem forte relação linear.

    3. **Gráfico de Valores Reais vs. Previstos:**
       - Pontos próximos à linha diagonal indicam previsões precisas.
       - Dispersão uniforme em torno da linha é ideal.

    4. **Importância das Características:**
       - Barras mais longas indicam características mais importantes para o modelo.
       - Útil para seleção de características e entendimento do modelo.

    5. **Gráfico de Resíduos:**
       - Idealmente, os resíduos devem ser distribuídos aleatoriamente em torno de zero.
       - Padrões nos resíduos podem indicar que o modelo não capturou alguma informação importante.
    """)

# Guia de Seleção de Modelos
with st.expander("Guia de Seleção de Modelos"):
    st.markdown("""
    ### Como Escolher o Modelo Adequado

    1. **Para dados lineares e interpretabilidade:**
       - Regressão Linear, Ridge, Lasso

    2. **Para dados complexos e não-lineares:**
       - Floresta Aleatória, Gradient Boosting, XGBoost, LightGBM

    3. **Para datasets pequenos:**
       - Regressão Linear, SVR, K-Vizinhos

    4. **Para grandes volumes de dados:**
       - Floresta Aleatória, XGBoost, LightGBM

    5. **Quando a interpretabilidade é crucial:**
       - Árvore de Decisão, Regressão Linear

    6. **Para lidar com multicolinearidade:**
       - Ridge, Lasso, Rede Elástica

    7. **Para seleção automática de features:**
       - Lasso, Floresta Aleatória (importância das features)

    8. **Quando o tempo de treinamento é limitado:**
       - K-Vizinhos, Árvore de Decisão

    Lembre-se: A escolha do modelo ideal geralmente envolve experimentação e comparação de desempenho em seus dados específicos.
    """)

# Seção de Previsão
st.subheader("🔮 Faça uma Previsão")
st.write("Use esta seção para fazer previsões com o modelo treinado.")

# Criar campos de entrada para cada característica
valores_entrada = {}
for coluna in X.columns:
    valores_entrada[coluna] = st.number_input(f"Insira o valor para {coluna}", value=X[coluna].mean())

# Botão para fazer a previsão
if st.button("Fazer Previsão"):
    # Preparar os dados de entrada
    entrada_previsao = pd.DataFrame([valores_entrada])
    entrada_previsao_escalonada = escalonador.transform(entrada_previsao)
    
    # Fazer a previsão
    previsao = modelo.predict(entrada_previsao_escalonada)
    
    # Exibir o resultado
    st.success(f"A previsão do modelo é: {previsao[0]:.2f}")

# Seção de Comparação de Modelos
st.subheader("🏆 Comparação de Modelos")
st.write("Compare o desempenho de diferentes modelos.")

# Lista de modelos para comparação
modelos_comparacao = {
    "Floresta Aleatória": RandomForestRegressor(n_estimators=100, random_state=42),
    "Aumento de Gradiente": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Regressão Linear": LinearRegression(),
    "SVR": SVR(),
    "K-Vizinhos": KNeighborsRegressor(n_neighbors=5)
}

# Treinar e avaliar cada modelo
resultados_comparacao = []
for nome_modelo, modelo in modelos_comparacao.items():
    modelo.fit(X_treino_escalonado, y_treino)
    y_previsto = modelo.predict(X_teste_escalonado)
    mse = mean_squared_error(y_teste, y_previsto)
    r2 = r2_score(y_teste, y_previsto)
    resultados_comparacao.append({"Modelo": nome_modelo, "EQM": mse, "R²": r2})

# Exibir resultados da comparação
df_comparacao = pd.DataFrame(resultados_comparacao)
st.dataframe(df_comparacao)

# Gráfico de barras para comparação visual
fig_comparacao = px.bar(df_comparacao, x="Modelo", y=["EQM", "R²"], barmode="group",
                        title="Comparação de Desempenho dos Modelos")
st.plotly_chart(fig_comparacao, use_container_width=True)

# Seção de Dicas e Melhores Práticas
st.subheader("💡 Dicas e Melhores Práticas")
st.write("""
1. **Preparação dos Dados:** Sempre verifique a qualidade dos seus dados. Trate valores ausentes e outliers.
2. **Seleção de Características:** Use técnicas como correlação e importância de características para selecionar as melhores variáveis.
3. **Validação Cruzada:** Para uma avaliação mais robusta, considere usar validação cruzada em vez de uma única divisão treino/teste.
4. **Ajuste de Hiperparâmetros:** Experimente diferentes configurações de hiperparâmetros para otimizar o desempenho do modelo.
5. **Interpretabilidade:** Considere o equilíbrio entre desempenho e interpretabilidade ao escolher um modelo.
6. **Atualização do Modelo:** Reavalie e atualize seu modelo regularmente com novos dados para manter sua precisão.
""")

# Rodapé
st.markdown("---")
st.markdown("Desenvolvido com ❤️ usando Streamlit | Criado para fins educativos")

# Botão para resetar a aplicação
if st.sidebar.button("🔄 Resetar Aplicação"):
    st.rerun()

# Informações Adicionais
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Informações Adicionais")
st.sidebar.info("""
Este aplicativo foi desenvolvido para fins educacionais e de demonstração.
Ele utiliza conjuntos de dados éticos e técnicas de aprendizado de máquina
para explorar diferentes modelos de regressão.

**Versão:** 1.0
**Última atualização:** 2023-06-01
""")

# Feedback do usuário
st.sidebar.markdown("---")
st.sidebar.subheader("📝 Feedback")
feedback = st.sidebar.text_area("Deixe seu feedback ou sugestões:")
if st.sidebar.button("Enviar Feedback"):
    # Aqui você pode implementar a lógica para salvar o feedback
    st.sidebar.success("Obrigado pelo seu feedback!")

# Seção de Recursos Adicionais
st.markdown("---")
st.subheader("📚 Recursos Adicionais")
st.markdown("""
Para aprender mais sobre regressão e ciência de dados, confira estes recursos:

1. [Curso de Machine Learning do Coursera](https://www.coursera.org/learn/machine-learning)
2. [Documentação do Scikit-learn](https://scikit-learn.org/stable/documentation.html)
3. [Livro: "Hands-On Machine Learning with Scikit-Learn and TensorFlow"](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
4. [Kaggle: Competições e datasets de Machine Learning](https://www.kaggle.com/)
5. [Towards Data Science - Medium](https://towardsdatascience.com/)
""")

# Possíveis Melhorias Futuras
st.markdown("---")
st.subheader("🚀 Possíveis Melhorias Futuras")
st.markdown("""
1. Implementar validação cruzada para uma avaliação mais robusta dos modelos.
2. Adicionar mais técnicas de pré-processamento de dados, como tratamento de outliers e codificação de variáveis categóricas.
3. Incorporar técnicas de seleção de características automáticas.
4. Implementar otimização de hiperparâmetros usando técnicas como Grid Search ou Random Search.
5. Adicionar mais visualizações interativas para exploração de dados.
6. Permitir o upload de datasets personalizados pelos usuários.
7. Implementar funcionalidades de exportação de modelos treinados.
8. Adicionar mais algoritmos de regressão avançados, como redes neurais.
""")

# Aviso Legal
st.markdown("---")
st.subheader("⚖️ Aviso Legal")
st.markdown("""
Este aplicativo é fornecido apenas para fins educacionais e de demonstração. 
Não deve ser usado para tomada de decisões em ambientes de produção sem uma 
validação adequada. Os autores não se responsabilizam por quaisquer consequências 
decorrentes do uso deste aplicativo.
""")

# Créditos e Agradecimentos
st.markdown("---")
st.subheader("🙏 Créditos e Agradecimentos")
st.markdown("""
- Desenvolvido por: João Renan Lopes E Carlos Egger
- Bibliotecas utilizadas: Streamlit, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, Plotly
- Agradecimentos especiais à comunidade de código aberto e aos contribuidores das bibliotecas utilizadas bem como ao professor Pedro Girotto.
""")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center;">
    <p>Desenvolvido com ❤️ usando Streamlit | Desenvolvedores: João Renan Lopes E Carlos Egger | Professor: pedro Girotto ;)
</div>
""", unsafe_allow_html=True)
