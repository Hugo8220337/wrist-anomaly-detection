# Estrutura do Projeto
Analise_Exploratoria_TrajData/
│
├── data/                       # Todos os dados (NUNCA alterar os ficheiros aqui dentro)
│   ├── raw/                    # Os CSVs originais com os dados dos sensores (imutável)
│   └── processed/              # Os CSVs gerados por si (ex: features_df.csv, matrizes DTW guardadas)
│
├── notebooks/                  # Onde o trabalho sujo é feito (Jupyter Notebooks)
│   ├── 1_data_prep/            # Notebooks de limpeza de sinal e extração de features
│   ├── 2_dtw_features_calc/    # Notebooks que apenas calculam e guardam as matrizes DTW e as features
│   ├── 3_models_features/      # Onde treina os modelos Feature-based (Isolation Forest, SVM...)
│   ├── 4_models_dtw/           # Onde treina os modelos baseados em DTW (LOF, DBSCAN...)
│   └── 5_evaluation/           # Notebooks que geram os gráficos Finais e Benchmarks
│
├── cache/                      # Ficheiros temporários e modelos guardados (joblib, pkl)
│   ├── models/                 # Os modelos finais treinados guardados
│   └── metadata/               # As variáveis temporárias que não quer recalcular
│
├── src/                        # (Opcional, mas de nível Sénior) Código Python reutilizável
│   ├── utils.py                # Funções repetitivas (ex: a função da Matriz de Confusão!)
│   └── plotting.py             # Funções para desenhar os gráficos sistematicamente
│
├── outliers.csv                # O "Ground Truth"

> "Don't Repeat Yourself"