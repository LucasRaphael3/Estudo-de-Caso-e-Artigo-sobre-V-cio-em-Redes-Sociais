"""
Tratamento de Dados - social_media_usage.csv
============================================
Dataset: Social Media & Mental Health (Kaggle, sintético 2010–2060)
Recorte: Teen + Young Adult (20.001 registros)
Alvo   : addiction_risk_level (Low / Medium / High)

Etapas:
  1. Carregamento e inspeção
  2. Recorte amostral por faixa etária
  3. Verificação de qualidade (nulos, duplicatas)
  4. One-Hot Encoding nas variáveis categóricas
  5. Padronização (StandardScaler) nas variáveis numéricas contínuas
  6. Exportação do dataset tratado
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURAÇÕES
# ─────────────────────────────────────────────
INPUT_FILE  = "social_media_usage.csv"
OUTPUT_FILE = "social_media_usage_tratado.csv"

AGE_GROUPS_ALVO = ["Teen", "Young Adult"]

VARIAVEIS_CATEGORICAS = ["country", "age_group", "gender", "platform"]

VARIAVEIS_NUMERICAS = [
    "daily_screen_time_hours",
    "doomscrolling_frequency",
    "notification_checks_per_day",
    "ai_recommendation_exposure",
    "productivity_loss_pct",
    "digital_detox_attempts",
]

VARIAVEL_ALVO = "addiction_risk_level"


# ─────────────────────────────────────────────
# 1. CARREGAMENTO E INSPEÇÃO
# ─────────────────────────────────────────────
def carregar_dados(caminho: str) -> pd.DataFrame:
    """Carrega o CSV e exibe informações básicas."""
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

    df = pd.read_csv(caminho)
    print("=" * 55)
    print("1. CARREGAMENTO")
    print("=" * 55)
    print(f"   Registros: {len(df):,}")
    print(f"   Colunas  : {df.shape[1]}")
    print(f"   Colunas  : {df.columns.tolist()}")
    print(f"   Tipos    :\n{df.dtypes.to_string()}")
    print()
    return df


# ─────────────────────────────────────────────
# 2. RECORTE AMOSTRAL
# ─────────────────────────────────────────────
def recortar_jovens(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra apenas Teen e Young Adult."""
    df_jovens = df[df["age_group"].isin(AGE_GROUPS_ALVO)].copy()
    df_jovens.reset_index(drop=True, inplace=True)

    print("=" * 55)
    print("2. RECORTE AMOSTRAL (Teen + Young Adult)")
    print("=" * 55)
    print(f"   Registros após recorte: {len(df_jovens):,}")
    print(f"   age_group:\n{df_jovens['age_group'].value_counts().to_string()}")
    print()
    return df_jovens


# ─────────────────────────────────────────────
# 3. VERIFICAÇÃO DE QUALIDADE
# ─────────────────────────────────────────────
def verificar_qualidade(df: pd.DataFrame) -> pd.DataFrame:
    """Checa nulos, duplicatas e distribuição do alvo."""
    print("=" * 55)
    print("3. VERIFICAÇÃO DE QUALIDADE")
    print("=" * 55)

    # Nulos
    nulos = df.isnull().sum()
    print(f"   Nulos por coluna:\n{nulos.to_string()}")
    print()

    # Duplicatas
    n_dup = df.duplicated().sum()
    print(f"   Linhas duplicadas: {n_dup}")
    if n_dup > 0:
        df = df.drop_duplicates()
        print(f"   → Duplicatas removidas. Novo shape: {df.shape}")
    print()

    # Distribuição da variável-alvo
    dist = df[VARIAVEL_ALVO].value_counts()
    pct  = df[VARIAVEL_ALVO].value_counts(normalize=True) * 100
    print(f"   Distribuição de '{VARIAVEL_ALVO}':")
    for cls in dist.index:
        print(f"     {cls}: {dist[cls]:,} ({pct[cls]:.2f}%)")
    print()

    return df


# ─────────────────────────────────────────────
# 4. ONE-HOT ENCODING
# ─────────────────────────────────────────────
def aplicar_ohe(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica One-Hot Encoding nas variáveis categóricas."""
    print("=" * 55)
    print("4. ONE-HOT ENCODING")
    print("=" * 55)

    df_ohe = pd.get_dummies(
        df,
        columns=VARIAVEIS_CATEGORICAS,
        drop_first=False,
        dtype=int,
    )

    colunas_ohe = [
        c for c in df_ohe.columns
        if any(c.startswith(p + "_") for p in VARIAVEIS_CATEGORICAS)
    ]
    print(f"   Colunas OHE geradas: {len(colunas_ohe)}")
    for cat in VARIAVEIS_CATEGORICAS:
        grp = [c for c in colunas_ohe if c.startswith(cat + "_")]
        preview = grp[:4]
        sufixo  = "..." if len(grp) > 4 else ""
        print(f"     {cat}: {len(grp)} col → {preview}{sufixo}")

    print(f"   Shape após OHE: {df_ohe.shape}")
    print()
    return df_ohe


# ─────────────────────────────────────────────
# 5. PADRONIZAÇÃO (StandardScaler)
# ─────────────────────────────────────────────
def padronizar_numericas(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Padroniza variáveis numéricas contínuas (média=0, std=1)."""
    print("=" * 55)
    print("5. PADRONIZAÇÃO (StandardScaler)")
    print("=" * 55)

    scaler = StandardScaler()
    df[VARIAVEIS_NUMERICAS] = scaler.fit_transform(df[VARIAVEIS_NUMERICAS])

    stats = df[VARIAVEIS_NUMERICAS].agg(["mean", "std"]).round(3)
    print(f"   Variáveis padronizadas: {VARIAVEIS_NUMERICAS}")
    print(f"   Média e Desvio-Padrão pós-padronização:")
    print(stats.to_string())
    print()
    return df, scaler


# ─────────────────────────────────────────────
# 6. EXPORTAÇÃO
# ─────────────────────────────────────────────
def exportar(df: pd.DataFrame, caminho: str) -> None:
    """Salva o dataset tratado em CSV."""
    df.to_csv(caminho, index=False)
    print("=" * 55)
    print("6. EXPORTAÇÃO")
    print("=" * 55)
    print(f"   Arquivo salvo: {caminho}")
    print(f"   Shape final  : {df.shape}")
    print(f"   Registros    : {len(df):,}")
    print(f"   Colunas      : {df.shape[1]}")
    print()


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────
def main():
    print("\n" + "=" * 55)
    print("  PIPELINE DE TRATAMENTO DE DADOS")
    print("  social_media_usage.csv → jovens (Teen + Young Adult)")
    print("=" * 55 + "\n")

    df = carregar_dados(INPUT_FILE)
    df = recortar_jovens(df)
    df = verificar_qualidade(df)
    df = aplicar_ohe(df)
    df, scaler = padronizar_numericas(df)
    exportar(df, OUTPUT_FILE)

    print("✅ Pipeline concluído com sucesso!")
    print(f"   Dataset tratado disponível em: {OUTPUT_FILE}\n")


if __name__ == "__main__":
    main()
