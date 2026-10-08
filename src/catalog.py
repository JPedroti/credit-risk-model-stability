"""Catalogue of sources and candidate features: what each one is, why it could matter for credit
risk, how its availability at `date_decision` was checked, and the decision taken about it.

The texts are the documentation delivered with the master table (brief, sections 7.3 and 9), so
they are written in Portuguese. `tests/test_catalog.py` keeps this file in step with the columns
actually produced by `src/features.py`.
"""
from dataclasses import dataclass

import pandas as pd

ORIGINAL, AUTHORED = "original", "autoral"
KEEP, DISCARD = "manter", "descartar"
LOW_RISK = "baixo"
INVESTIGATED = "baixo, após investigação individual (IV entre os maiores; notebook 02, seção 4.2)"


@dataclass(frozen=True)
class SourceNote:
    represents: str
    hypothesis: str
    strategy: str
    decision: str


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    source: str
    kind: str  # ORIGINAL (raw column) or AUTHORED (built in this project)
    rule: str
    hypothesis: str
    temporal_check: str
    decision: str = KEEP
    reason: str = "hipótese de negócio clara e nenhum impedimento encontrado na auditoria"
    leakage_risk: str = LOW_RISK
    # Every candidate is built only from information that exists at `date_decision` (brief, 7.3).
    available_at_decision: bool = True


# --------------------------------------------------------------------------------------------------
# Sources (brief, section 6, step 2)
# --------------------------------------------------------------------------------------------------
SOURCE_NOTES = {
    "static_0": SourceNote(
        "Pedido em análise e resumo, feito pelo credor, do comportamento do cliente em contratos anteriores",
        "Quem atrasou parcelas com o próprio credor tende a atrasar de novo; pedidos e recusas recentes indicam busca por crédito",
        "join direto + seleção de colunas",
        "usar",
    ),
    "static_cb_0": SourceNote(
        "Resumo externo: consultas ao bureau e deduções fiscais",
        "Muitas consultas recentes ao bureau indicam busca intensa por crédito",
        "join direto + seleção de colunas",
        "usar (consultas ao bureau)",
    ),
    "applprev_1": SourceNote(
        "Pedidos anteriores do cliente ao credor (os 20 mais recentes)",
        "Recusas, atrasos e pedidos recentes antecipam risco; relacionamento longo e sem atraso reduz",
        "agregação por case_id com corte na data de criação",
        "usar",
    ),
    "person_1": SourceNote(
        "Pessoas do pedido: a linha 0 é o solicitante, as demais são contatos",
        "Idade, tipo de renda e renda descrevem a estabilidade financeira do solicitante",
        "filtro num_group1 = 0 + join",
        "usar (só o solicitante)",
    ),
    "credit_bureau_a_1": SourceNote(
        "Contratos do cliente em outras instituições (bureau A), ativos e encerrados",
        "Atraso em outros credores, crédito recém-aberto e histórico curto elevam o risco; complementa o histórico interno quando ele é pequeno",
        "agregação por case_id, ativos e encerrados em separado",
        "usar",
    ),
    "credit_bureau_a_2": SourceNote(
        "Situação mensal de pagamento de cada contrato do bureau A",
        "Atraso recente pesa mais do que atraso antigo",
        "agregação em janelas de 3 e 12 meses antes da decisão",
        "usar",
    ),
    "tax_registry_a_1": SourceNote(
        "Deduções fiscais por empregador (fornecedor A)",
        "Deduções regulares indicam emprego formal e renda estável",
        "agregação unificada dos três fornecedores",
        "usar (unificada)",
    ),
    "tax_registry_b_1": SourceNote(
        "Deduções fiscais por empregador (fornecedor B)",
        "Deduções regulares indicam emprego formal e renda estável",
        "agregação unificada dos três fornecedores",
        "usar (unificada)",
    ),
    "tax_registry_c_1": SourceNote(
        "Deduções fiscais por empregador (fornecedor C)",
        "Deduções regulares indicam emprego formal e renda estável",
        "agregação unificada dos três fornecedores",
        "usar (unificada)",
    ),
    "credit_bureau_b_1": SourceNote(
        "Contratos do cliente no bureau B",
        "Mesma do bureau A",
        "-",
        "não usar: cobre cerca de 2% dos casos",
    ),
    "credit_bureau_b_2": SourceNote(
        "Pagamentos dos contratos do bureau B",
        "Mesma do bureau A",
        "-",
        "não usar: cobre cerca de 2% dos casos",
    ),
    "deposit_1": SourceNote(
        "Depósitos do cliente no credor",
        "Ter reserva financeira no credor reduz o risco",
        "-",
        "não usar: cobre cerca de 7% dos casos e o indicador de presença quase não separa",
    ),
    "debitcard_1": SourceNote(
        "Cartões de débito do cliente no credor",
        "Movimentação de conta indica renda e relacionamento",
        "-",
        "não usar: cobre cerca de 7% dos casos e o indicador de presença quase não separa",
    ),
    "other_1": SourceNote(
        "Entradas e saídas de conta e depósito",
        "Fluxo de caixa indica capacidade de pagamento",
        "-",
        "não usar: só existe a partir de 2020-01 e cobre cerca de 3% dos casos",
    ),
    "applprev_2": SourceNote(
        "Detalhes dos pedidos anteriores (bloqueio de cartão, tipo de contato)",
        "Sem hipótese clara",
        "-",
        "não usar: não consigo formular uma hipótese de risco",
    ),
    "person_2": SourceNote(
        "Endereços e vínculos de pessoas relacionadas",
        "Sem hipótese clara",
        "-",
        "não usar: não consigo formular uma hipótese de risco",
    ),
}

# --------------------------------------------------------------------------------------------------
# Temporal checks shared by groups of features
# --------------------------------------------------------------------------------------------------
TERMS = "condição do próprio pedido, fixada na decisão"
LENDER_SUMMARY = (
    "agregado calculado pelo credor; nenhuma data de static_0 passa da decisão no calendário corrigido"
)
APPLICATION_FORM = "dado cadastral informado no pedido"
BUREAU_SUMMARY = "resultado da consulta ao bureau feita no processamento do pedido"
CUT_APPLPREV = "só pedidos com data de criação <= decisão (filtro no código)"
CUT_BUREAU = "só contratos iniciados até a decisão (filtro no código); situação reportada pelo bureau até a decisão"
CUT_PAYMENTS = "só meses com pagamento reportado e <= mês da decisão (filtro no código)"
# Appended to the rule of the features that are left unknown when the bureau sent no closed contract.
COMPLETE_ONLY = "; nula se o bureau informa contratos encerrados mas não envia o detalhe deles"
CUT_TAX = "só deduções com data <= decisão (filtro no código)"

REDUNDANT = "redundante com {other}: mede o mesmo e fica a versão {kept}"
NO_SIGNAL = "sem sinal univariado em nenhum semestre do TRAIN e sem ganho sobre as partes que a compõem"


def _static(name: str, rule: str, hypothesis: str, check: str = LENDER_SUMMARY, **decision) -> FeatureSpec:
    return FeatureSpec(name, "static_0", ORIGINAL, rule, hypothesis, check, **decision)


FEATURES = [
    # ---- static_0: the application being decided --------------------------------------------------
    _static("credamount_770A", "coluna original", "valores maiores expõem mais o cliente e o credor", TERMS),
    _static("annuity_780A", "coluna original", "parcela mensal: quanto maior, maior o esforço de pagamento", TERMS),
    _static("pmtnum_254L", "coluna original (número de parcelas)", "prazos longos dão mais tempo para a situação do cliente piorar", TERMS),
    _static(
        "eir_270L", "coluna original (taxa de juros efetiva)",
        "a taxa reflete o risco que o credor já enxergava e encarece a dívida", TERMS,
        reason="mantida com ressalva: reflete a política de preço vigente, que pode mudar no tempo",
    ),
    _static(
        "credtype_322L", "coluna original (COL = bens, CAL = dinheiro, REL = rotativo)",
        "produtos diferentes atraem perfis de risco diferentes", TERMS,
        reason="mantida como segmento de produto, apesar do IV baixo",
    ),
    # ---- static_0: debts and repayment behaviour with the lender ----------------------------------
    _static("totaldebt_9A", "coluna original", "dívida já existente com o credor reduz a folga para a nova parcela"),
    _static(
        "annuitynextmonth_57A", "coluna original", "parcelas já comprometidas no próximo mês",
        decision=DISCARD, reason=REDUNDANT.format(other="totaldebt_9A", kept="de dívida total"),
    ),
    _static("avgdpdtolclosure24_3658938P", "coluna original", "atraso médio nos últimos 24 meses: hábito de pagar atrasado", leakage_risk=INVESTIGATED),
    _static("pctinstlsallpaidlate1d_3546856L", "coluna original", "proporção de parcelas pagas com atraso: frequência do atraso"),
    _static("pctinstlsallpaidlat10d_839L", "coluna original", "proporção de parcelas com 10+ dias de atraso: atraso que não é esquecimento"),
    _static("pctinstlsallpaidearl3d_427L", "coluna original", "pagar adiantado indica folga e disciplina"),
    _static("maxdpdlast3m_392P", "coluna original", "atraso recente (3 meses) é o sinal mais atual de dificuldade"),
    _static(
        "maxdpdlast12m_727P", "coluna original", "maior atraso em 12 meses",
        decision=DISCARD,
        reason="redundante: fica entre as janelas de 3 e 24 meses e repete maxdbddpdtollast12m_3658940P",
    ),
    _static("maxdpdlast24m_143P", "coluna original", "maior atraso em 24 meses: gravidade no médio prazo"),
    _static("maxdpdtolerance_374P", "coluna original", "maior atraso de todo o histórico"),
    _static("maxdbddpdtollast12m_3658940P", "coluna original (negativo = dias de antecedência)", "combina atraso e antecedência em 12 meses"),
    _static("avgdbddpdlast24m_3658932P", "coluna original (negativo = dias de antecedência)", "antecedência média com que o cliente paga"),
    _static("numinstlswithdpd10_728L", "coluna original", "quantidade de parcelas com 10+ dias de atraso"),
    _static("numinstlsallpaid_934L", "coluna original", "muitas parcelas pagas = cliente conhecido e testado"),
    # ---- static_0: recent application behaviour and shared contacts -------------------------------
    _static(
        "numrejects9m_859L", "coluna original", "recusas recentes",
        decision=DISCARD, reason=REDUNDANT.format(other="prev_n_rejected_12m", kept="com regra e corte auditáveis"),
    ),
    _static("applications30d_658L", "coluna original", "vários pedidos em 30 dias indicam urgência por crédito"),
    _static(
        "numcontrs3months_479L", "coluna original", "contratos nos últimos 3 meses",
        decision=DISCARD, reason=REDUNDANT.format(other="prev_n_applications_3m", kept="com regra e corte auditáveis"),
    ),
    _static("lastst_736L", "coluna original (D = recusado, K = quitado, A = ativo, T = cancelado)", "o desfecho do último pedido resume a relação recente com o credor"),
    _static(
        "lastrejectreason_759M", "coluna original, com o código de ausente convertido em nulo",
        "o motivo da última recusa separa recusas por risco de recusas operacionais",
        decision=DISCARD,
        reason="a mistura de códigos muda continuamente ao longo do TRAIN e os códigos são anonimizados; "
               "o histórico de recusas fica com days_since_last_rejection, prev_share_rejected e prev_n_rejected_12m",
    ),
    _static("mobilephncnt_593L", "coluna original", "telefone compartilhado por várias pessoas é sinal clássico de fraude ou de cadastro frágil"),
    _static("sellerplacecnt_915L", "coluna original", "o mesmo documento usado em vários pontos de venda indica busca por crédito em vários lugares"),
    FeatureSpec(
        "days_since_last_delinquency", "static_0", AUTHORED,
        "date_decision - lastdelinqdate_224D, no calendário corrigido (nulo = nunca atrasou)",
        "atraso recente é mais relevante do que atraso antigo", "valores negativos viram nulo (nenhum caso no TRAIN)",
    ),
    FeatureSpec(
        "days_since_last_rejection", "static_0", AUTHORED,
        "date_decision - lastrejectdate_50D, no calendário corrigido (nulo = nunca recusado)",
        "quem foi recusado há pouco tempo e volta a pedir tem risco maior", "valores negativos viram nulo (nenhum caso no TRAIN)",
    ),
    FeatureSpec(
        "downpayment_share", "static_0", AUTHORED, "downpmt_116A / price_1097A",
        "dar entrada maior mostra comprometimento com a compra", TERMS,
        decision=DISCARD, reason="quase sempre zero e sem sinal univariado",
    ),
    # ---- static_cb_0 ------------------------------------------------------------------------------
    FeatureSpec("days30_165L", "static_cb_0", ORIGINAL, "coluna original", "consultas ao bureau em 30 dias: busca imediata por crédito", BUREAU_SUMMARY),
    FeatureSpec("days120_123L", "static_cb_0", ORIGINAL, "coluna original", "consultas ao bureau em 120 dias", BUREAU_SUMMARY),
    FeatureSpec("days360_512L", "static_cb_0", ORIGINAL, "coluna original", "consultas ao bureau em 360 dias: apetite por crédito no ano", BUREAU_SUMMARY),
    FeatureSpec(
        "cb_queries_30d_share", "static_cb_0", AUTHORED, "days30_165L / days360_512L",
        "consultas concentradas no último mês indicam aceleração da busca por crédito", BUREAU_SUMMARY,
        decision=DISCARD, reason=REDUNDANT.format(other="days30_165L", kept="original"),
    ),
    # ---- person_1 (applicant) ---------------------------------------------------------------------
    FeatureSpec(
        "mainoccupationinc_384A", "person_1", ORIGINAL, "coluna original, linha do solicitante",
        "renda declarada: base da capacidade de pagamento", APPLICATION_FORM,
        reason="mantida como denominador das razões de comprometimento, apesar do IV baixo",
    ),
    FeatureSpec("incometype_1044T", "person_1", ORIGINAL, "coluna original, linha do solicitante", "aposentados e servidores têm renda mais estável do que autônomos", APPLICATION_FORM),
    FeatureSpec(
        "education_927M", "person_1", ORIGINAL, "coluna original, com o código de ausente convertido em nulo",
        "escolaridade se associa a estabilidade de renda", APPLICATION_FORM,
        decision=DISCARD,
        reason="metade dos casos sem valor, códigos anonimizados e composição que muda no último trimestre do TRAIN",
    ),
    FeatureSpec("age_years", "person_1", AUTHORED, "(date_decision - birth_259D) / 365,25", "clientes mais velhos têm renda e histórico mais estáveis", APPLICATION_FORM),
    FeatureSpec(
        "employment_years", "person_1", AUTHORED, "(date_decision - empl_employedfrom_271D) / 365,25, no calendário corrigido",
        "mais tempo no emprego = renda mais estável", APPLICATION_FORM,
        decision=DISCARD, reason="coleta instável: o preenchimento mensal oscila muito dentro do TRAIN",
    ),
    # ---- applprev_1 -------------------------------------------------------------------------------
    FeatureSpec("prev_n_applications", "applprev_1", AUTHORED, "quantidade de pedidos anteriores (0 se não há)", "muitos pedidos indicam dependência de crédito", CUT_APPLPREV),
    FeatureSpec("prev_n_contracts", "applprev_1", AUTHORED, "pedidos anteriores que viraram contrato (dateactivated_425D preenchida)", "contratos já firmados = relacionamento conhecido pelo credor", CUT_APPLPREV),
    FeatureSpec("prev_share_rejected", "applprev_1", AUTHORED, "pedidos recusados (status D) / pedidos anteriores", "quem foi recusado muitas vezes já foi visto como arriscado", CUT_APPLPREV),
    FeatureSpec("prev_days_since_last_application", "applprev_1", AUTHORED, "menor (date_decision - creationdate_885D), calendário corrigido", "pedido muito recente indica urgência ou tentativas repetidas", CUT_APPLPREV),
    FeatureSpec("prev_years_since_first_application", "applprev_1", AUTHORED, "maior (date_decision - creationdate_885D) / 365,25", "tempo de relacionamento com o credor", CUT_APPLPREV),
    FeatureSpec("prev_n_applications_3m", "applprev_1", AUTHORED, "pedidos criados nos 90 dias antes da decisão", "aceleração recente da procura por crédito", CUT_APPLPREV),
    FeatureSpec("prev_n_applications_12m", "applprev_1", AUTHORED, "pedidos criados nos 365 dias antes da decisão", "procura por crédito no último ano", CUT_APPLPREV),
    FeatureSpec("prev_n_rejected_12m", "applprev_1", AUTHORED, "pedidos recusados (status D) criados nos 365 dias antes da decisão", "recusas recentes mostram que o risco já foi percebido há pouco", CUT_APPLPREV),
    FeatureSpec("prev_max_dpd", "applprev_1", AUTHORED, "maior maxdpdtolerance_577P entre os contratos anteriores", "maior atraso já observado com o credor", CUT_APPLPREV),
    FeatureSpec("prev_share_contracts_dpd", "applprev_1", AUTHORED, "contratos com maxdpdtolerance_577P > 0 / contratos com atraso conhecido", "atrasar em vários contratos é hábito, não acidente", CUT_APPLPREV),
    FeatureSpec(
        "prev_outstanding_debt", "applprev_1", AUTHORED, "soma de outstandingdebt_522A dos pedidos anteriores",
        "dívida em aberto com o credor", CUT_APPLPREV,
        decision=DISCARD, reason=REDUNDANT.format(other="totaldebt_9A", kept="original"),
    ),
    FeatureSpec(
        "prev_max_credamount", "applprev_1", AUTHORED, "maior credamount_590A entre os contratos anteriores",
        "maior crédito que o cliente já assumiu", CUT_APPLPREV,
        decision=DISCARD, reason="ingrediente de credit_to_prev_max_credit, que concentra a hipótese",
    ),
    # ---- credit_bureau_a_1 ------------------------------------------------------------------------
    FeatureSpec(
        "cb_n_active_contracts", "credit_bureau_a_1", AUTHORED, "quantidade de contratos ativos no bureau",
        "muitos contratos simultâneos pressionam o orçamento", CUT_BUREAU,
        decision=DISCARD, reason="sem sinal univariado em nenhum semestre do TRAIN",
    ),
    FeatureSpec("cb_n_closed_contracts", "credit_bureau_a_1", AUTHORED, "contratos encerrados informados pelo bureau (numberofcontrsvalue_358L; 0 se o bureau respondeu sem nenhum)", "contratos encerrados mostram experiência de crédito concluída", BUREAU_SUMMARY),
    FeatureSpec(
        "cb_active_debt", "credit_bureau_a_1", AUTHORED, "totaloutstanddebtvalue_39A (total do bureau para os contratos ativos)",
        "dívida em aberto em outras instituições", CUT_BUREAU,
        decision=DISCARD, reason=REDUNDANT.format(other="cb_debt_to_income", kept="relativa à renda"),
    ),
    FeatureSpec(
        "cb_active_monthly_instalment", "credit_bureau_a_1", AUTHORED, "soma de monthlyinstlamount_332A dos contratos ativos",
        "parcelas mensais já comprometidas em outras instituições", CUT_BUREAU,
        decision=DISCARD, reason="sem sinal univariado em nenhum semestre do TRAIN",
    ),
    FeatureSpec("cb_active_overdue_amount", "credit_bureau_a_1", AUTHORED, "soma de overdueamount_659A dos contratos ativos", "estar inadimplente agora em outro credor é o sinal mais direto de dificuldade", CUT_BUREAU),
    FeatureSpec("cb_active_max_dpd", "credit_bureau_a_1", AUTHORED, "maior dpdmax_139P entre os contratos ativos", "gravidade do atraso nos contratos em andamento", CUT_BUREAU),
    FeatureSpec(
        "cb_closed_max_dpd", "credit_bureau_a_1", AUTHORED, "maior dpdmax_757P entre os contratos encerrados",
        "gravidade do atraso em contratos já encerrados", CUT_BUREAU,
        reason="mantida com ressalva: o bureau só passa a informar atraso de contratos encerrados em 2019-06",
    ),
    FeatureSpec("cb_share_contracts_dpd", "credit_bureau_a_1", AUTHORED, "contratos (ativos + encerrados) com atraso máximo > 0 / contratos com atraso conhecido" + COMPLETE_ONLY, "atrasar em muitos contratos é padrão de comportamento", CUT_BUREAU, leakage_risk=INVESTIGATED),
    FeatureSpec("cb_years_since_first_credit", "credit_bureau_a_1", AUTHORED, "maior (date_decision - início do contrato) / 365,25" + COMPLETE_ONLY, "histórico de crédito longo dá mais evidência e costuma indicar menor risco", CUT_BUREAU),
    FeatureSpec("cb_days_since_last_credit", "credit_bureau_a_1", AUTHORED, "menor (date_decision - início do contrato), calendário corrigido" + COMPLETE_ONLY, "crédito aberto dias antes em outro credor indica acúmulo de dívidas", CUT_BUREAU),
    FeatureSpec("cb_n_credits_12m", "credit_bureau_a_1", AUTHORED, "contratos iniciados nos 365 dias antes da decisão" + COMPLETE_ONLY, "muitos créditos novos em um ano = endividamento acelerado", CUT_BUREAU),
    FeatureSpec(
        "cb_months_since_max_dpd", "credit_bureau_a_1", AUTHORED, "meses desde o maior atraso de um contrato (ano/mês corrigidos)" + COMPLETE_ONLY,
        "atraso grave recente pesa mais", CUT_BUREAU,
        decision=DISCARD, reason=REDUNDANT.format(other="cb_pmt_months_since_last_late", kept="do histórico mensal"),
        leakage_risk=INVESTIGATED,
    ),
    # ---- credit_bureau_a_2 ------------------------------------------------------------------------
    FeatureSpec("cb_pmt_max_dpd_3m", "credit_bureau_a_2", AUTHORED, "maior atraso (dias) nos pagamentos dos 3 meses até a decisão" + COMPLETE_ONLY, "atraso nos últimos 3 meses: dificuldade em curso", CUT_PAYMENTS),
    FeatureSpec("cb_pmt_max_dpd_12m", "credit_bureau_a_2", AUTHORED, "maior atraso (dias) nos pagamentos dos 12 meses até a decisão" + COMPLETE_ONLY, "gravidade do atraso no último ano", CUT_PAYMENTS, leakage_risk=INVESTIGATED),
    FeatureSpec("cb_pmt_share_late_12m", "credit_bureau_a_2", AUTHORED, "pagamentos com atraso > 0 / pagamentos reportados nos 12 meses até a decisão" + COMPLETE_ONLY, "frequência do atraso no último ano", CUT_PAYMENTS, leakage_risk=INVESTIGATED),
    FeatureSpec("cb_pmt_months_since_last_late", "credit_bureau_a_2", AUTHORED, "meses desde o último pagamento com atraso (nulo = nunca atrasou)" + COMPLETE_ONLY, "quanto mais recente o último atraso, maior o risco", CUT_PAYMENTS, leakage_risk=INVESTIGATED),
    # ---- tax registries ---------------------------------------------------------------------------
    FeatureSpec("tax_n_deductions", "tax_registry_a/b/c_1", AUTHORED, "quantidade de deduções fiscais do fornecedor com mais registros para o caso", "deduções regulares indicam emprego formal contínuo", CUT_TAX),
    FeatureSpec(
        "tax_n_employers", "tax_registry_a/b/c_1", AUTHORED, "empregadores distintos nas deduções",
        "trocar muito de empregador indica renda instável", CUT_TAX,
        decision=DISCARD, reason="o sinal só aparece nos meses do fornecedor B: depende de quem fornece o dado",
    ),
    FeatureSpec(
        "tax_amount_cv", "tax_registry_a/b/c_1", AUTHORED, "desvio padrão / média do valor das deduções",
        "deduções irregulares indicam renda instável", CUT_TAX,
        decision=DISCARD, reason="o sinal só aparece nos meses do fornecedor B: depende de quem fornece o dado",
    ),
    # ---- ratios across sources --------------------------------------------------------------------
    FeatureSpec(
        "annuity_to_income", "static_0 + person_1", AUTHORED, "annuity_780A / mainoccupationinc_384A",
        "comprometimento da renda com a nova parcela", TERMS,
        reason="mantida pela hipótese clássica de capacidade de pagamento, apesar do IV baixo",
    ),
    FeatureSpec(
        "credit_to_income", "static_0 + person_1", AUTHORED, "credamount_770A / mainoccupationinc_384A",
        "tamanho do crédito em relação à renda", TERMS,
        reason="mantida pela hipótese clássica de capacidade de pagamento, apesar do IV baixo",
    ),
    FeatureSpec(
        "debt_service_to_income", "static_0 + credit_bureau_a_1 + person_1", AUTHORED,
        "(annuity_780A + cb_active_monthly_instalment) / mainoccupationinc_384A",
        "parcelas totais (nova + bureau) em relação à renda", CUT_BUREAU,
        decision=DISCARD, reason=NO_SIGNAL,
    ),
    FeatureSpec("cb_debt_to_income", "credit_bureau_a_1 + person_1", AUTHORED, "cb_active_debt / mainoccupationinc_384A", "endividamento em outras instituições em relação à renda", CUT_BUREAU),
    FeatureSpec("credit_to_prev_max_credit", "static_0 + applprev_1", AUTHORED, "credamount_770A / maior crédito já contratado com o credor", "pedir muito mais do que já se pagou antes é um salto de exposição", CUT_APPLPREV),
]

SPECS = {spec.name: spec for spec in FEATURES}
CANDIDATE_FEATURES = [spec.name for spec in FEATURES]
SELECTED_FEATURES = [spec.name for spec in FEATURES if spec.decision == KEEP]


def catalog_frame() -> pd.DataFrame:
    """The catalogue as a table, one row per candidate feature."""
    return pd.DataFrame([
        {
            "feature": spec.name,
            "fonte": spec.source,
            "tipo": spec.kind,
            "regra_de_calculo": spec.rule,
            "hipotese_de_negocio": spec.hypothesis,
            "disponivel_na_decisao": "sim" if spec.available_at_decision else "não",
            "verificacao_temporal": spec.temporal_check,
            "risco_de_leakage": spec.leakage_risk,
            "decisao": spec.decision,
            "motivo": spec.reason,
        }
        for spec in FEATURES
    ])


def features_of(source: str) -> pd.DataFrame:
    """Rule and hypothesis of the candidate features built from exactly this source."""
    catalog = catalog_frame()
    columns = ["feature", "tipo", "regra_de_calculo", "hipotese_de_negocio"]
    return catalog.loc[catalog["fonte"] == source, columns].reset_index(drop=True)


def combined_features() -> pd.DataFrame:
    """Rule and hypothesis of the candidate features that combine more than one source."""
    catalog = catalog_frame()
    columns = ["feature", "fonte", "regra_de_calculo", "hipotese_de_negocio"]
    return catalog.loc[catalog["fonte"].str.contains(" + ", regex=False), columns].reset_index(drop=True)


def source_notes_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "fonte": name,
            "o_que_representa": note.represents,
            "hipotese_de_negocio": note.hypothesis,
            "estrategia": note.strategy,
            "decisao": note.decision,
        }
        for name, note in SOURCE_NOTES.items()
    ])
