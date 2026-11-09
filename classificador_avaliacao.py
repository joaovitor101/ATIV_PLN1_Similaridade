import string
import nltk
import pandas as pd
from nltk.corpus import stopwords

# Carregar stopwords em português (já baixadas)
stop_words = set(stopwords.words("portuguese"))

# -----------------------------------------------------------------------------
# 1. TEXTOS DE REFERÊNCIA
# -----------------------------------------------------------------------------
ref_positiva = (
    "O produto possui excelente qualidade, chegou rapidamente e atendeu "
    "completamente às minhas expectativas. Gostei muito da compra e recomendo "
    "este produto para outras pessoas."
)

ref_negativa = (
    "O produto possui péssima qualidade, apresentou problemas logo após a "
    "compra e não atendeu às minhas expectativas. Fiquei muito decepcionado e não "
    "recomendo este produto."
)


# -----------------------------------------------------------------------------
# 2. PRÉ-PROCESSAMENTO DA REFERÊNCIA POSITIVA
# -----------------------------------------------------------------------------
# Remoção de espaços desnecessários
ref_pos_sem_espacos = " ".join(ref_positiva.strip().split())

# Minúsculas e Tokenização
tokens_ref_pos = nltk.word_tokenize(ref_pos_sem_espacos.lower())

# Remoção de pontuação
tokens_ref_pos_sem_pontuacao = [
    palavra for palavra in tokens_ref_pos
    if palavra not in string.punctuation
]

# Remoção de stopwords (artigos, preposições, etc.)
tokens_ref_pos_final = [
    palavra for palavra in tokens_ref_pos_sem_pontuacao
    if palavra not in stop_words
]

set_ref_pos = set(tokens_ref_pos_final)
str_ref_pos = " ".join(tokens_ref_pos_final)


# -----------------------------------------------------------------------------
# 3. PRÉ-PROCESSAMENTO DA REFERÊNCIA NEGATIVA
# -----------------------------------------------------------------------------
# Remoção de espaços desnecessários
ref_neg_sem_espacos = " ".join(ref_negativa.strip().split())

# Minúsculas e Tokenização
tokens_ref_neg = nltk.word_tokenize(ref_neg_sem_espacos.lower())

# Remoção de pontuação
tokens_ref_neg_sem_pontuacao = [
    palavra for palavra in tokens_ref_neg
    if palavra not in string.punctuation
]

# Remoção de stopwords
tokens_ref_neg_final = [
    palavra for palavra in tokens_ref_neg_sem_pontuacao
    if palavra not in stop_words
]

set_ref_neg = set(tokens_ref_neg_final)
str_ref_neg = " ".join(tokens_ref_neg_final)


# -----------------------------------------------------------------------------
# 4. LEITURA DO ARQUIVO EXCEL
# -----------------------------------------------------------------------------
df = pd.read_excel("avaliacoes (1).xlsx")
resultados = []


# -----------------------------------------------------------------------------
# 5. PROCESSAMENTO DAS AVALIAÇÕES (LOOP PRINCIPAL)
# -----------------------------------------------------------------------------
for index, linha in df.iterrows():
    eval_id = linha["ID"]
    texto_avaliacao = str(linha["avaliacao"])

    # 5.2 Pré-processamento dos textos:
    # 1) Remoção de espaços desnecessários
    texto_sem_espacos = " ".join(texto_avaliacao.strip().split())

    # 2) Conversão para minúsculas e Tokenização
    tokens = nltk.word_tokenize(texto_sem_espacos.lower())

    # 3) Remoção de pontuação
    tokens_sem_pontuacao = [
        palavra for palavra in tokens
        if palavra not in string.punctuation
    ]

    # 4) Remoção de stopwords (artigos, conectivos, etc.)
    tokens_finais = [
        palavra for palavra in tokens_sem_pontuacao
        if palavra not in stop_words
    ]

    # Preparar conjunto (para Jaccard) e string (para Levenshtein)
    set_eval = set(tokens_finais)
    str_eval = " ".join(tokens_finais)

    # -------------------------------------------------------------------------
    # CÁLCULO DE JACCARD (POSITIVO E NEGATIVO)
    # -------------------------------------------------------------------------
    inter_pos = len(set_eval.intersection(set_ref_pos))
    union_pos = len(set_eval.union(set_ref_pos))
    jaccard_pos = inter_pos / union_pos if union_pos > 0 else 0.0

    inter_neg = len(set_eval.intersection(set_ref_neg))
    union_neg = len(set_eval.union(set_ref_neg))
    jaccard_neg = inter_neg / union_neg if union_neg > 0 else 0.0

    if jaccard_pos > jaccard_neg:
        class_jaccard = "POSITIVA"
    else:
        class_jaccard = "NEGATIVA"

    # -------------------------------------------------------------------------
    # CÁLCULO DE LEVENSHTEIN POSITIVO
    # -------------------------------------------------------------------------
    s1, s2 = str_eval, str_ref_pos
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    linha_anterior = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        linha_atual = [i + 1]
        for j, c2 in enumerate(s2):
            insercoes = linha_anterior[j + 1] + 1
            delecoes = linha_atual[j] + 1
            substituicoes = linha_anterior[j] + (c1 != c2)
            linha_atual.append(min(insercoes, delecoes, substituicoes))
        linha_anterior = linha_atual
    lev_pos = linha_anterior[-1]

    # -------------------------------------------------------------------------
    # CÁLCULO DE LEVENSHTEIN NEGATIVO
    # -------------------------------------------------------------------------
    s1, s2 = str_eval, str_ref_neg
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    linha_anterior = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        linha_atual = [i + 1]
        for j, c2 in enumerate(s2):
            insercoes = linha_anterior[j + 1] + 1
            delecoes = linha_atual[j] + 1
            substituicoes = linha_anterior[j] + (c1 != c2)
            linha_atual.append(min(insercoes, delecoes, substituicoes))
        linha_anterior = linha_atual
    lev_neg = linha_anterior[-1]

    if lev_pos < lev_neg:
        class_levenshtein = "POSITIVA"
    else:
        class_levenshtein = "NEGATIVA"

    # Armazenar resultados da linha
    resultados.append(
        {
            "ID": eval_id,
            "avaliacao": texto_avaliacao,
            "Jaccard Pos.": round(jaccard_pos, 4),
            "Jaccard Neg.": round(jaccard_neg, 4),
            "Class. Jaccard": class_jaccard,
            "Lev. Pos.": lev_pos,
            "Lev. Neg.": lev_neg,
            "Class. Lev.": class_levenshtein,
        }
    )

# -----------------------------------------------------------------------------
# 6. SALVAR RESULTADOS NO EXCEL
# -----------------------------------------------------------------------------
df_resultado = pd.DataFrame(resultados)
df_resultado.to_excel("resultado_classificacao.xlsx", index=False)

print("Processamento concluído com sucesso!")
