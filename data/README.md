# Dados

Os dados **não são versionados** (ver `.gitignore`): nem os arquivos do Kaggle nem a Master Table,
que é recriada pelo notebook `02_feature_engineering.ipynb`.

## Como obter

1. Entre no Kaggle e aceite as regras da competição
   [Home Credit - Credit Risk Model Stability](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/data).
2. Baixe os dados e descompacte em `data/raw/`. O projeto usa apenas:

   ```
   data/raw/
   ├── feature_definitions.csv        dicionário de dados (465 colunas descritas)
   └── parquet_files/
       └── train/                     32 arquivos train_*.parquet (1,33 GB)
   ```

   Os arquivos de `test` da competição e as versões em CSV não são usados.
3. Confira a integridade dos dois arquivos de referência (SHA-256):

   | Arquivo | Bytes | SHA-256 |
   |---|---|---|
   | `parquet_files/train/train_base.parquet` | 7.048.047 | `29d94662545eece908f20e44e6592336c9c40916dcfdad7735de92b8e9edf908` |
   | `feature_definitions.csv` | 34.047 | `628f1f75ae2842de0dbd6559e31855e0edff87da16d382d55725d6a2afba6dd4` |

   ```bash
   python -c "import hashlib; print(hashlib.sha256(open('data/raw/parquet_files/train/train_base.parquet','rb').read()).hexdigest())"
   ```

## O que os arquivos contêm (verificado nos dados)

- `train_base.parquet`: 1.526.659 casos, um por `case_id`, decididos entre 2019-01-01 e 2020-10-05.
  Colunas: `case_id`, `date_decision`, `MONTH`, `WEEK_NUM`, `target`.
- 16 famílias de tabelas secundárias em três granularidades (`depth` 0, 1 e 2). O inventário, com
  linhas, casos e cobertura de cada uma, está em [`reports/source_inventory.csv`](../reports/source_inventory.csv).
- As datas vêm como texto `AAAA-MM-DD`. As das tabelas secundárias estão **14 dias adiantadas** em
  relação ao calendário de `date_decision`, e os pares (ano, mês), 1 mês adiantados. A evidência
  está na seção 1 do notebook 02; a correção, em `src/config.py` e `src/features.py`.
- Nas colunas categóricas mascaradas (sufixo `M`), o código `a55475b1` significa valor ausente.

## Saída

`data/processed/master_table.parquet`: 1 linha por `case_id`, com as chaves da `train_base`, a coluna
`split` (TRAIN / TEST / OOT) e as features selecionadas. O catálogo das features está em
[`reports/feature_catalog.csv`](../reports/feature_catalog.csv).
