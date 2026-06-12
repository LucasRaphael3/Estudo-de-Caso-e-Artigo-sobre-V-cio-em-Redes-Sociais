# 📱 Social Media & Mental Health — Tratamento de Dados

Pipeline de pré-processamento do dataset *Social Media & Mental Health* (Kaggle),
com foco no público jovem (Teen e Young Adult) e na variável-alvo `addiction_risk_level`.

---

## 📁 Estrutura

```
.
├── tratamento_dados.py          # Script principal do pipeline
├── social_media_usage.csv       # Dataset original (não incluído no repo)
├── social_media_usage_tratado.csv  # Dataset tratado (gerado pelo script)
└── README.md
```

---

## ⚙️ Requisitos

```bash
pip install pandas numpy scikit-learn
```

Python 3.10+

---

## ▶️ Como executar

Coloque o arquivo `social_media_usage.csv` na mesma pasta do script e rode:

```bash
python tratamento_dados.py
```

O arquivo `social_media_usage_tratado.csv` será gerado automaticamente.

---

## 🔄 Etapas do Pipeline

| Etapa | Descrição |
|-------|-----------|
| 1. Carregamento | Leitura do CSV com inspeção de dimensões e tipos |
| 2. Recorte amostral | Filtro por `age_group ∈ {Teen, Young Adult}` → 20.001 registros |
| 3. Qualidade | Verificação de nulos e duplicatas |
| 4. One-Hot Encoding | Codificação de `country`, `age_group`, `gender`, `platform` |
| 5. Padronização | StandardScaler nas 6 variáveis numéricas contínuas |
| 6. Exportação | Geração do CSV tratado (20.001 × 44 colunas) |

---

## 📊 Dataset

- **Fonte:** Kaggle (dados sintéticos, 2010–2060)
- **Total original:** 50.000 registros
- **Após recorte:** 20.001 registros
- **Variável-alvo:** `addiction_risk_level` (Low / Medium / High)

| Classe | N | % |
|--------|---|---|
| Low    | 6.087 | 30,43% |
| Medium | 9.987 | 49,93% |
| High   | 3.927 | 19,63% |

---

## 📝 Observação

Os dados são **sintéticos/projetados** e não representam evidência epidemiológica.
Os resultados devem ser interpretados como exploração computacional.
