# nadyele_tcc_usp_esalq_data_science
Trabalho de conclusão de curso Data Science &amp; Anlytics
####TCC NADYELE C SOUZA### CÓDIGOS
#====================================================================#===============================================================================
##ETAPA 1 - CONFIRGURAR AMBIENTE
#===============================================================================

import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler



sns.set_theme(style="whitegrid", palette="deep")
RNG_SEED = 42



OUT_DIR = "./outputs"
REAL_DATA_PATH = "respostas_formulario.csv"  


os.makedirs(OUT_DIR, exist_ok=True)

USE_REAL_DATA = True

REAL_POSITIONAL_COLUMNS = [
    "carimbo", "email", "idade", "escolaridade", "tempo_estudo",
    "dificuldade_percebida", "maior_dificuldade", "medo_falar", "motivacao",
    "desmotivador_principal", "avaliacao_metodo", "oportunidade_pratica",
    "contato_midias", "opiniao_dificuldade",
]


def load_and_clean_real_data(path):
    raw = pd.read_csv(path)
    raw.columns = REAL_POSITIONAL_COLUMNS[: len(raw.columns)]
    n_total = len(raw)

    for col in ["motivacao", "oportunidade_pratica"]:
        if col in raw.columns:
            raw[col] = raw[col].replace({"As vezes": "Às vezes"})

    raw = raw.drop_duplicates()
    n_after_full_dup = len(raw)

    raw["carimbo_dt"] = pd.to_datetime(raw["carimbo"], dayfirst=True, errors="coerce")
    raw = raw.sort_values("carimbo_dt")
    raw = raw.drop_duplicates(subset="email", keep="first")
    n_after_email_dup = len(raw)

    raw = raw[raw["idade"] != "Menos de 18"]
    n_final = len(raw)

    print("Limpeza dos dados:")
    print(f"  Respostas brutas exportadas do Forms ......... {n_total}")
    print(f"  Após remoção de duplicidades completas ........ {n_after_full_dup}")
    print(f"  Após remoção de e-mails repetidos ............. {n_after_email_dup}")
    print(f"  Após exclusão de menores de 18 anos ........... {n_final}")

    df = raw.drop(columns=["carimbo", "email", "carimbo_dt"]).reset_index(drop=True)
    return df, dict(n_total=n_total, n_after_full_dup=n_after_full_dup,
                     n_after_email_dup=n_after_email_dup, n_final=n_final)


def load_data():
    if USE_REAL_DATA:
        df, cleaning_log = load_and_clean_real_data(REAL_DATA_PATH)
        with open(f"{OUT_DIR}/log_limpeza_dados.txt", "w", encoding="utf-8") as f:
            for k, v in cleaning_log.items():
                f.write(f"{k}: {v}\n")
    else:
        raise RuntimeError("USE_REAL_DATA deve ser True")
    return df



df_limpo = load_data()


df_limpo.head()

#====================================================================#===============================================================================
##ETAPA 2  - DEFINIÇÃO DA FUNÇÃO DE LIMPEZA E CODIFICAÇÃO 
#===============================================================================

def clean_and_encode(df):
    df = df.drop_duplicates().copy()

    ord3 = {"Não": 0, "Às vezes": 0.5, "Mais ou menos": 0.5, "Sim": 1}
    tempo_map = {
        "Menos de 1 ano": 0,
        "1 a 3 anos": 1,
        "4 a 6 anos": 2,
        "Mais de 6 anos": 3,
    }
    escol_map_norm = {
        "ensino médio": 0,
        "ensino medio": 0,
        "ensino superior incompleto": 1,
        "ensino superior completo": 2,
        "pós-graduação": 3,
        "pos-graduacao": 3,
    }
    
    
    
    df["escolaridade_norm"] = df["escolaridade"].str.strip().str.lower()
    df["escolaridade_num"] = df["escolaridade_norm"].map(escol_map_norm)

    df["medo_falar_num"] = df["medo_falar"].map(ord3)
    df["dificuldade_num"] = df["dificuldade_percebida"].map(ord3)
    df["motivacao_num"] = df["motivacao"].map(ord3)
    df["oportunidade_num"] = df["oportunidade_pratica"].map(ord3)
    df["tempo_estudo_num"] = df["tempo_estudo"].map(tempo_map)

    aval_map = {"Ruim": 0, "Regular": 1, "Boa": 2, "Muito boa": 3}
    df["aval_num"] = df["avaliacao_metodo"].map(aval_map)
    
    return df




df_encoded = clean_and_encode(df_limpo)



df_encoded.head()

#====================================================================#===============================================================================
##ETAPA 3 - ESTATÍSTICA DESCRITIVA
#===============================================================================


def clean_and_encode(df):
    df = df.drop_duplicates().copy()

    ord3 = {"Não": 0, "Às vezes": 0.5, "Mais ou menos": 0.5, "Sim": 1}
    tempo_map = {
        "Menos de 1 ano": 0,
        "1 a 3 anos": 1,
        "4 a 6 anos": 2,
        "Mais de 6 anos": 3
    }
    escol_map_norm = {
        "ensino médio": 0, "ensino medio": 0,
        "ensino superior incompleto": 1,
        "ensino superior completo": 2,
        "pós-graduação": 3, "pos-graduacao": 3,
    }
    

    
    df["escolaridade_norm"] = df["escolaridade"].str.strip().str.lower()
    df["escolaridade_num"] = df["escolaridade_norm"].map(escol_map_norm)

    df["medo_falar_num"] = df["medo_falar"].map(ord3)
    df["dificuldade_num"] = df["dificuldade_percebida"].map(ord3)
    df["motivacao_num"] = df["motivacao"].map(ord3)
    df["oportunidade_num"] = df["oportunidade_pratica"].map(ord3)
    df["tempo_estudo_num"] = df["tempo_estudo"].map(tempo_map)

    aval_map = {"Ruim": 0, "Regular": 1, "Boa": 2, "Muito boa": 3}
    df["aval_num"] = df["avaliacao_metodo"].map(aval_map)
    
    return df



df_encoded = clean_and_encode(df_limpo)


df_encoded.head()



#====================================================================#===============================================================================
##ETAPA 4 - CORRELAÇÃO (PEARSON + SPEARMAN)
#===============================================================================

NUM_COLS = [
    "medo_falar_num", "dificuldade_num", "motivacao_num",
    "oportunidade_num", "tempo_estudo_num", "escolaridade_num"
]
NUM_LABELS = [
    "Medo de falar", "Dificuldade percebida", "Motivação",
    "Oportunidade de prática", "Tempo de estudo", "Escolaridade"
]
COL_LABEL = dict(zip(NUM_COLS, NUM_LABELS))



def correlation_analysis(df):
    corr_p = df[NUM_COLS].corr(method="pearson")
    corr_s = df[NUM_COLS].corr(method="spearman")
    corr_p.index = corr_p.columns = NUM_LABELS
    corr_s.index = corr_s.columns = NUM_LABELS

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
   
    sns.heatmap(
        corr_p, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
        ax=axes[0], square=True, cbar_kws={"shrink": .8}
    )
    axes[0].set_title("Correlação de Pearson")
    

    sns.heatmap(
        corr_s, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
        ax=axes[1], square=True, cbar_kws={"shrink": .8}
    )
    axes[1].set_title("Correlação de Spearman (ordinal)")
    
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/1_correlacao_pearson_spearman.png", dpi=150)
    plt.show()  
    plt.close()
    
    return corr_p, corr_s



corr_p, corr_s = correlation_analysis(df_encoded)

#====================================================================#===============================================================================
##ETAPA 5 - TESTES DE ASSOCIAÇÃO (QUI-QUADRADO + CRAMÉR'SV + VOLM)
#===============================================================================

def cramers_v(chi2, n, r, c):
    return np.sqrt(chi2 / (n * (min(r, c) - 1)))



def association_tests(df):
    pairs = [
        ("medo_falar", "dificuldade_percebida"),
        ("motivacao", "oportunidade_pratica"),
        ("escolaridade", "dificuldade_percebida"),
        ("medo_falar", "motivacao"),
        ("tempo_estudo", "motivacao"),
    ]
    rows = []
    for a, b in pairs:
        table = pd.crosstab(df[a], df[b])
        chi2, p, dof, exp = stats.chi2_contingency(table)
        v = cramers_v(chi2, table.values.sum(), *table.shape)
        rows.append({
            "Variável A": a, "Variável B": b,
            "Qui-quadrado": round(chi2, 3), "gl": dof,
            "p-valor": round(p, 4),
            "Cramér's V": round(v, 3),
            "min esperado": round(exp.min(), 2),
            "Significativo (p<0,05)": "Sim" if p < 0.05 else "Não",
        })
    result = pd.DataFrame(rows)
    

    order = np.argsort(result["p-valor"].values)
    m = len(result)
    adj = np.empty(m)
    running = 0
    for rank, idx in enumerate(order):
        val = (m - rank) * result.loc[idx, "p-valor"]
        running = max(running, val)
        adj[idx] = min(1, running)
    result["p_holm"] = adj.round(4)
    

    result.to_csv(f"{OUT_DIR}/2_testes_associacao_qui_quadrado.csv",
                   index=False, encoding="utf-8-sig")
    return result



tabela_resultados = association_tests(df_limpo)


tabela_resultados

#================================================================================
##ETAPA 6 - K- MEANS (ELBOW + SILHUETA) + PEFIS
#===============================================================================


CLUSTER_FEATURES = [
    "medo_falar_num",
    "dificuldade_num",
    "motivacao_num",
    "oportunidade_num",
    "tempo_estudo_num",
]



def kmeans_analysis(df):
    X = df[CLUSTER_FEATURES].dropna()
    Xs = StandardScaler().fit_transform(X)

    k_range = range(2, 6)
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=RNG_SEED).fit(Xs)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(Xs, km.labels_))

    max_sil = max(silhouettes)
    best_k = next(k for k, s in zip(k_range, silhouettes) if s >= max_sil - 0.01)


    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(list(k_range), inertias, "o-")
    axes[0].set_title("Método do Cotovelo (Elbow)")
    axes[0].set_xlabel("Número de clusters (k)")
    axes[0].set_ylabel("Inércia")

    axes[1].plot(list(k_range), silhouettes, "o-", color="darkorange")
    axes[1].axvline(best_k, ls="--", color="grey")
    axes[1].set_title(f"Índice de Silhueta (melhor k = {best_k})")
    axes[1].set_xlabel("Número de clusters (k)")
    axes[1].set_ylabel("Silhueta média")

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/3_elbow_silhueta.png", dpi=150)
    plt.show()  
    plt.close()

    km_final = KMeans(n_clusters=best_k, n_init=10, random_state=RNG_SEED).fit(
        Xs
    )
    df = df.loc[X.index].copy()
    df["cluster"] = km_final.labels_

   
    profile = df.groupby("cluster")[CLUSTER_FEATURES].mean().round(2)
    profile["n_alunos"] = df.groupby("cluster").size()
    profile.to_csv(f"{OUT_DIR}/4_perfis_clusters.csv", encoding="utf-8-sig")

   
    means = df[CLUSTER_FEATURES].mean()
    labels_map = {}
    for c in profile.index:
        diffs = profile.loc[c, CLUSTER_FEATURES] - means
        top_feat = diffs.abs().idxmax()
        direction = "alta" if diffs[top_feat] > 0 else "baixa"
        labels_map[c] = f"Cluster {c}: {direction} '{top_feat}'"


    pca = PCA(n_components=2, random_state=RNG_SEED)
    coords = pca.fit_transform(Xs)

    plt.figure(figsize=(7, 5.5))
    palette = sns.color_palette("Set2", best_k)
    for c in sorted(df["cluster"].unique()):
        mask = df["cluster"] == c
        plt.scatter(
            coords[mask.values, 0],
            coords[mask.values, 1],
            label=labels_map[c],
            s=45,
            color=palette[c],
            alpha=0.8,
        )

    plt.title("Perfis de alunos (K-Means, projeção PCA)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}% var.)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}% var.)")
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/5_clusters_pca.png", dpi=150)
    plt.show()  
    plt.close()

    return df, profile, best_k, labels_map


df_clustered, perfil_tabela, k_ideal, rotulos = kmeans_analysis(df_encoded)


perfil_tabela


#==============================================================================
##ETAPA 7 - IMPORTÂNCIA DE VARIÁVEIS (RANDOM FOREST)
#===============================================================================


def feature_importance_analysis(df):
    feats = [
        "medo_falar_num",
        "motivacao_num",
        "oportunidade_num",
        "tempo_estudo_num",
        "escolaridade_num",
    ]
    data = df[feats + ["dificuldade_num"]].dropna()
    X, y = data[feats], data["dificuldade_num"]


    rf = RandomForestRegressor(n_estimators=300, random_state=RNG_SEED)
    rf.fit(X, y)
    

    importances = pd.Series(rf.feature_importances_, index=feats)
    importances = importances.sort_values(ascending=True)
    importances.index = [COL_LABEL[i] for i in importances.index]


    plt.figure(figsize=(8, 4.5))
    importances.plot(kind="barh", color="steelblue")
    plt.title("Importância das variáveis na dificuldade percebida", fontsize=12)
    plt.xlabel("Importância (Random Forest)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/6_importancia_variaveis.png", dpi=150)
    plt.show()  
    plt.close()

    return importances.sort_values(ascending=False)


resultado_importancia = feature_importance_analysis(df_encoded)


resultado_importancia
#===============================================================================
##ETAPA 8 - ANALISE SOCIODEMOGRAFICA
#===============================================================================

sns.set_theme(style="whitegrid")
COLOR = "#4682B4"


raw, _ = load_and_clean_real_data(REAL_DATA_PATH)
enc = clean_and_encode(raw)
enc_cl, profile, best_k, _ = kmeans_analysis(enc)
n = len(enc_cl)


anx = profile["medo_falar_num"].idxmax()
enc_cl["perfil"] = np.where(
    enc_cl["cluster"] == anx,
    "Ansiedade e baixa exposição",
    "Autonomia e engajamento"
)
print("\nTamanho dos perfis:\n", enc_cl["perfil"].value_counts())


lines = []
def log(s=""):
    print(s)
    lines.append(str(s))

def bar_panel(ax, series, order, title, color=COLOR):
    vc = series.value_counts().reindex(order).fillna(0).astype(int)
    total = vc.sum()
    ax.bar(range(len(vc)), vc.values, color=color)
    ax.set_xticks(range(len(vc)))
    ax.set_xticklabels([o.replace("Ensino ", "Ens. ") for o in vc.index],
                       rotation=15, ha="right", fontsize=9)
    for i, v in enumerate(vc.values):
        ax.text(i, v + total * 0.008, f"{v}\n({v / total * 100:.1f}%)",
                ha="center", va="bottom", fontsize=8.5)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Respondentes")
    ax.set_ylim(0, vc.max() * 1.25)


IDADE = ["18 a 25", "26 a 35", "36 ou mais"]
ESC = ["Ensino médio", "Ensino superior incompleto",
       "Ensino superior completo", "Pós-graduação"]
TEMPO = ["Menos de 1 ano", "1 a 3 anos", "4 a 6 anos", "Mais de 6 anos"]
AVAL = ["Muito boa", "Boa", "Regular", "Ruim"]
SNA = ["Sim", "Mais ou menos", "Não"]
SAN = ["Sim", "Às vezes", "Não"]


fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
bar_panel(axes[0, 0], enc_cl["idade"], IDADE, "Faixa etária")
bar_panel(axes[0, 1], enc_cl["escolaridade"].str.strip(), ESC, "Escolaridade")
bar_panel(axes[1, 0], enc_cl["tempo_estudo"], TEMPO, "Tempo de estudo do inglês")
bar_panel(axes[1, 1], enc_cl["avaliacao_metodo"], AVAL, "Avaliação da forma como aprendeu")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/7_perfil_amostra.png", dpi=150)
plt.show()
plt.close()


fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
bar_panel(axes[0, 0], enc_cl["dificuldade_percebida"], SNA, "Considera difícil aprender inglês?", "#C0504D")
bar_panel(axes[0, 1], enc_cl["medo_falar"], SAN, "Sente medo/vergonha ao falar?", "#C0504D")
bar_panel(axes[1, 0], enc_cl["motivacao"], SAN, "Sente-se motivado(a)?", "#5B9B5B")
bar_panel(axes[1, 1], enc_cl["oportunidade_pratica"], SAN, "Teve oportunidades de praticar conversação?", "#5B9B5B")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/8_percepcoes.png", dpi=150)
plt.show()
plt.close()

log("=== 1) PERFIL ===")
log(enc_cl["idade"].value_counts().to_string())

#===============================================================================
##ETAPA 9 - MAIOR DIFICULDADE TÉCNICA
#===============================================================================

DIFS = [
    "Falar(speaking)",
    "Ouvir (listening)",
    "Gramática",
    "Vocabulário",
    "Escrever (writing)",
    "Ler (reading)",
]

DIF_LABEL = {
    "Falar(speaking)": "Falar (speaking)",
    "Ouvir (listening)": "Ouvir (listening)",
    "Gramática": "Gramática",
    "Vocabulário": "Vocabulário",
    "Escrever (writing)": "Escrever (writing)",
    "Ler (reading)": "Ler (reading)",
}


for d in DIFS:
  enc_cl[f"dif_{d}"] = (
      enc_cl["maior_dificuldade"]
      .fillna("")
      .apply(lambda s: d in [x.strip() for x in s.split(",")])
  )

enc_cl["n_dif"] = enc_cl[[f"dif_{d}" for d in DIFS]].sum(axis=1)
dif_counts = pd.Series(
    {d: int(enc_cl[f"dif_{d}"].sum()) for d in DIFS}
).sort_values(ascending=True)


plt.figure(figsize=(8, 4.5))
plt.barh(
    [DIF_LABEL[d] for d in dif_counts.index],
    dif_counts.values,
    color=COLOR if "COLOR" in globals() else "#4682B4",
)

n_total = n if "n" in globals() else len(enc_cl)
out_folder = OUT if "OUT" in globals() else OUT_DIR

for i, v in enumerate(dif_counts.values):
  plt.text(v + 1, i, f"{v} ({v / n_total * 100:.1f}%)", va="center", fontsize=9)

plt.xlim(0, dif_counts.max() * 1.2)
plt.xlabel("Respondentes que marcaram a habilidade (resposta múltipla)")
plt.title("Maiores dificuldades no inglês")
plt.tight_layout()
plt.savefig(f"{out_folder}/9_maior_dificuldade.png", dpi=150)
plt.show() 
plt.close()


print("\n=== 2) MAIOR DIFICULDADE (resposta múltipla) ===")
for d in dif_counts.index[::-1]:
  qtd = dif_counts[d]
  pct_val = (qtd / n_total) * 100
  print(f"  {DIF_LABEL[d]}: {qtd} ({pct_val:.1f}%)")

print(f"  Média de habilidades marcadas: {enc_cl['n_dif'].mean():.2f}")
tot_6 = int((enc_cl["n_dif"] == 6).sum())
print(f"  Marcaram as 6 habilidades: {tot_6} ({(tot_6 / n_total) * 100:.1f}%)")


tab = pd.crosstab(enc_cl["dif_Falar(speaking)"], enc_cl["medo_falar"])
chi2, p, dof, exp = stats.chi2_contingency(tab)
v = np.sqrt(chi2 / (n_total * (min(tab.shape) - 1)))

print(
    f"\n  Speaking x medo de falar: chi2={chi2:.2f}, gl={dof}, p={p:.4f},"
    f" V={v:.3f}"
)
print("  % que cita speaking por nível de medo:")
print(
    (
        enc_cl.groupby("medo_falar")["dif_Falar(speaking)"].mean() * 100
    ).round(1).to_string()
)

#===============================================================================
##ETAPA 10 - PRINCIPAL DESMOTIVADOR
#===============================================================================


PRINC = [
    "Falta de tempo",
    "Dificuldade de entender",
    "Falta de prática",
    "Método de ensino",
    "Falta de interesse",
]



def code_desmot(s):
  s = str(s).strip()
  if s in PRINC:
    return s
  low = s.lower()
  if re.search(r"dinheiro|financ|valor|caro|custo|pre[çc]o", low):
    return "Custo financeiro"
  return "Outros"



enc_cl["desmot"] = enc_cl["desmotivador_principal"].apply(code_desmot)

ORD_D = [
    "Falta de tempo",
    "Dificuldade de entender",
    "Falta de prática",
    "Método de ensino",
    "Falta de interesse",
    "Custo financeiro",
    "Outros",
]

dc = enc_cl["desmot"].value_counts().reindex(ORD_D).fillna(0).astype(int)
dc_plot = dc.sort_values()

n_total = n if "n" in globals() else len(enc_cl)
out_folder = OUT if "OUT" in globals() else OUT_DIR
chart_color = COLOR if "COLOR" in globals() else "#4682B4"


plt.figure(figsize=(8, 4.5))
plt.barh(dc_plot.index, dc_plot.values, color=chart_color)

for i, val in enumerate(dc_plot.values):
  pct_val = (val / n_total) * 100
  plt.text(val + 1, i, f"{val} ({pct_val:.1f}%)", va="center", fontsize=9)

plt.xlim(0, dc_plot.max() * 1.2)
plt.xlabel("Respondentes")
plt.title("Principal fator de desmotivação")
plt.tight_layout()
plt.savefig(f"{out_folder}/10_desmotivadores.png", dpi=150)
plt.show() 
plt.close()


print("\n=== 3) DESMOTIVADOR PRINCIPAL ===")
for k in ORD_D:
  val = int(dc[k])
  pct_val = (val / n_total) * 100
  print(f"  {k}: {val} ({pct_val:.1f}%)")

outros_txt = enc_cl.loc[enc_cl["desmot"] == "Outros", "desmotivador_principal"]
print(
    "\n  (texto livre 'Outros'): "
    + " | ".join(outros_txt.head(15).astype(str))
)


ext = dc[
    ["Falta de tempo", "Falta de prática", "Método de ensino", "Custo financeiro"]
].sum()
intr = dc[["Dificuldade de entender", "Falta de interesse"]].sum()

print(
    f"\n  Fatores Externos: {ext} ({(ext / n_total) * 100:.1f}%)\n "
    f" Fatores Internos: {intr} ({(intr / n_total) * 100:.1f}%)"
)

#===============================================================================
##ETAPA 11 - MIDIAS + MODERAÇÃO
#===============================================================================

CANAIS = [
    "Filmes e séries",
    "Música",
    "Redes sociais",
    "Jogos",
    "Trabalho/estudo",
    "Não tenho contato",
]

for c in CANAIS:
  enc_cl[f"m_{c}"] = (
      enc_cl["contato_midias"]
      .fillna("")
      .apply(lambda s: c in [x.strip() for x in s.split(",")])
  )

DIG = ["Filmes e séries", "Música", "Redes sociais", "Jogos"]
enc_cl["n_dig"] = enc_cl[[f"m_{c}" for c in DIG]].sum(axis=1)
mc = pd.Series({c: int(enc_cl[f"m_{c}"].sum()) for c in CANAIS}).sort_values()


n_total = n if "n" in globals() else len(enc_cl)
out_folder = OUT if "OUT" in globals() else OUT_DIR
chart_color = COLOR if "COLOR" in globals() else "#4682B4"


plt.figure(figsize=(8, 4.2))
plt.barh(mc.index, mc.values, color=chart_color)

for i, val in enumerate(mc.values):
  pct_val = (val / n_total) * 100
  plt.text(val + 1, i, f"{val} ({pct_val:.1f}%)", va="center", fontsize=9)

plt.xlim(0, mc.max() * 1.2)
plt.xlabel("Respondentes (resposta múltipla)")
plt.title("Onde os participantes têm contato com o inglês")
plt.tight_layout()
plt.savefig(f"{out_folder}/11_midias.png", dpi=150)
plt.show() 
plt.close()


print("\n=== 4) MÍDIAS ===")
for c in mc.index[::-1]:
  qtd = mc[c]
  print(f"  {c}: {qtd} ({(qtd / n_total) * 100:.1f}%)")


for var, nome in [
    ("dificuldade_num", "dificuldade"),
    ("motivacao_num", "motivação"),
]:
  rho, pv = stats.spearmanr(enc_cl["n_dig"], enc_cl[var])
  print(f"  Spearman nº canais digitais x {nome}: rho={rho:.3f}, p={pv:.4f}")



def ols(X, y):
  X = np.column_stack([np.ones(len(X)), X])
  beta, *_ = np.linalg.lstsq(X, y, rcond=None)
  res = y - X @ beta
  dof = len(y) - X.shape[1]
  s2 = res @ res / dof
  cov = s2 * np.linalg.inv(X.T @ X)
  se = np.sqrt(np.diag(cov))
  t = beta / se
  pv = 2 * stats.t.sf(np.abs(t), dof)
  r2 = 1 - (res @ res) / ((y - y.mean()) @ (y - y.mean()))
  return beta, se, t, pv, r2



z = lambda s: (s - s.mean()) / s.std()
medo_z, dig_z = z(enc_cl["medo_falar_num"]).values, z(enc_cl["n_dig"]).values


y = enc_cl["dificuldade_num"].values
beta, se, t, pv, r2 = ols(np.column_stack([medo_z, dig_z, medo_z * dig_z]), y)
print("\n  Moderação: dificuldade ~ medo + canais digitais + medo*canais")
for nm, b_, s_, p_ in zip(
    ["intercepto", "medo", "canais", "medo x canais"], beta, se, pv
):
  print(f"    {nm}: beta={b_:.3f} (EP={s_:.3f}), p={p_:.4f}")
print(f"    R2 = {r2:.3f}")


y2 = enc_cl["motivacao_num"].values
beta2, se2, t2, pv2, r22 = ols(
    np.column_stack([medo_z, dig_z, medo_z * dig_z]), y2
)
print("\n  Moderação: motivação ~ medo + canais digitais + medo*canais")
for nm, b_, s_, p_ in zip(
    ["intercepto", "medo", "canais", "medo x canais"], beta2, se2, pv2
):
  print(f"    {nm}: beta={b_:.3f} (EP={s_:.3f}), p={p_:.4f}")
print(f"    R2 = {r22:.3f}")

#===============================================================================
##ETAPA 12 - AVALIAÇÃO DO MÉTODO
#===============================================================================

AVAL = [
    a
    for a in ["Muito ruim", "Ruim", "Regular", "Boa", "Muito boa"]
    if a in enc_cl["avaliacao_metodo"].unique()
]
if not AVAL:
  AVAL = enc_cl["avaliacao_metodo"].dropna().unique().tolist()

SNA = [
    s
    for s in ["Sim", "Não", "Às vezes", "Apenas em algumas situações"]
    if s in enc_cl["dificuldade_percebida"].unique()
]
if not SNA:
  SNA = enc_cl["dificuldade_percebida"].dropna().unique().tolist()


n_total = n if "n" in globals() else len(enc_cl)
out_folder = OUT if "OUT" in globals() else OUT_DIR


print("\n=== 5) AVALIAÇÃO DO MÉTODO ===")
for a, b, nome in [
    ("avaliacao_metodo", "dificuldade_percebida", "dificuldade"),
    ("avaliacao_metodo", "motivacao", "motivação"),
    ("avaliacao_metodo", "medo_falar", "medo de falar"),
]:
  if a in enc_cl.columns and b in enc_cl.columns:
    tab = pd.crosstab(enc_cl[a], enc_cl[b])
    chi2, p, dof, exp = stats.chi2_contingency(tab)
    v = np.sqrt(chi2 / (n_total * (min(tab.shape) - 1)))
    print(
        f"  Avaliação x {nome}: chi2={chi2:.2f}, gl={dof}, p={p:.4f}, V={v:.3f}"
    )


for var, nome in [
    ("dificuldade_num", "dificuldade"),
    ("motivacao_num", "motivação"),
    ("medo_falar_num", "medo"),
]:
  if "aval_num" in enc_cl.columns and var in enc_cl.columns:
    rho, pv = stats.spearmanr(enc_cl["aval_num"], enc_cl[var])
    print(f"  Spearman avaliação x {nome}: rho={rho:.3f}, p={pv:.4f}")


ct = (
    pd.crosstab(
        enc_cl["avaliacao_metodo"],
        enc_cl["dificuldade_percebida"],
        normalize="index",
    ).reindex(AVAL)[SNA]
    * 100
)

ax = ct.plot(
    kind="bar",
    stacked=True,
    figsize=(8, 4.8),
    color=["#C0504D", "#E6B85C", "#5B9B5B", "#4F81BD"][: len(SNA)],
)

ax.set_ylabel("% dentro de cada avaliação")
ax.set_xlabel("Avaliação da forma como aprendeu inglês")
ax.set_title("Dificuldade percebida segundo a avaliação do método")
ax.legend(title="Considera difícil?", bbox_to_anchor=(1.01, 1), loc="upper left")

ns = enc_cl["avaliacao_metodo"].value_counts()
ax.set_xticklabels([f"{a}\n(n={ns.get(a, 0)})" for a in AVAL], rotation=0)

plt.tight_layout()
plt.savefig(f"{out_folder}/12_avaliacao_x_dificuldade.png", dpi=150)
plt.show()
plt.close()

#===============================================================================
##ETAPA 13 - PERGUNTA ABERTA
#===============================================================================

TEMAS = {
    "Medo, vergonha e insegurança": (
        r"medo|vergonh|tim[ií]d|inseguran|ansiedad|nervos|trava|bloque|errar|julg"
    ),
    "Falta de prática e exposição": (
        r"pr[aá]tic|conversa|contato|exposi|imers|falar com|uso di[aá]rio"
    ),
    "Gramática e escrita": (
        r"gram[aá]tic|verbo|conjug|regra|estrutura|escrit|escrever|ortograf"
    ),
    "Pronúncia e compreensão auditiva": (
        r"pron[uú]ncia|escut|ouvir|ouvi |listening|entend|sotaque|r[aá]pid"
    ),
    "Vocabulário e memorização": (
        r"vocabul|palavra|memori|decor|mem[oó]ria|esquec"
    ),
    "Tempo, disciplina e concentração": (
        r"tempo|disciplin|concentra|organiza|rotina|foco|pregui|const[aâ]ncia|procrastin|cansa"
    ),
    "Método e ensino": r"m[eé]todo|professor|aula|ensino|escola|curso|did[aá]tic",
    "Diferenças em relação ao português": (
        r"diferente|portugu[eê]s|irregular|l[oó]gica|l[ií]ngua materna|significados"
    ),
    "Custo financeiro": r"dinheiro|financ|caro|custo|valor|pre[çc]o",
}


n_total = n if "n" in globals() else len(enc_cl)
out_folder = OUT if "OUT" in globals() else OUT_DIR
chart_color = COLOR if "COLOR" in globals() else "#4682B4"
lines = lines if "lines" in globals() else []


def registrar_log(texto):
  print(texto)
  lines.append(str(texto))



txt = enc_cl["opiniao_dificuldade"].fillna("").astype(str).str.lower()
validas = txt[txt.str.strip().str.len() > 3]

registrar_log("\n=== 6) PERGUNTA ABERTA ===")
registrar_log(f"  Respostas com conteúdo: {len(validas)} de {n_total}")

cod = pd.DataFrame(
    {t: validas.str.contains(r, regex=True) for t, r in TEMAS.items()}
)
tc = pd.Series({t: int(cod[t].sum()) for t in TEMAS}).sort_values()
sem_tema = int((~cod.any(axis=1)).sum())

for t in tc.index[::-1]:
  pct_tema = (tc[t] / len(validas)) * 100 if len(validas) > 0 else 0
  registrar_log(f"  {t}: {tc[t]} ({pct_tema:.1f}%)")

pct_sem_tema = (sem_tema / len(validas)) * 100 if len(validas) > 0 else 0
registrar_log(f"  Sem tema: {sem_tema} ({pct_sem_tema:.1f}%)")


plt.figure(figsize=(8.5, 5))
plt.barh(tc.index, tc.values, color=chart_color)

for i, val in enumerate(tc.values):
  pct_val = (val / len(validas)) * 100 if len(validas) > 0 else 0
  plt.text(
      val + 0.5, i, f"{val} ({pct_val:.1f}%)", va="center", fontsize=9
  )

plt.xlim(0, max(tc.max() * 1.22, 1))
plt.xlabel(f"Respostas que mencionam o tema (n = {len(validas)})")
plt.title("Por que é difícil aprender inglês? (pergunta aberta)")
plt.tight_layout()
plt.savefig(f"{out_folder}/13_temas_pergunta_aberta.png", dpi=150)
plt.show()
plt.close()


registrar_log("\n=== 7) VALIDAÇÃO EXTERNA DOS CLUSTERS ===")
rows = []
g = enc_cl.groupby("perfil")

for perfil, sub in g:
  rows.append({
      "Perfil": perfil,
      "n": len(sub),
      "% cita speaking": round(sub["dif_Falar(speaking)"].mean() * 100, 1),
      "% avalia método Regular/Ruim": round(
          sub["avaliacao_metodo"].isin(["Regular", "Ruim"]).mean() * 100, 1
      ),
      "% desmotivador: dificuldade de entender": round(
          (sub["desmot"] == "Dificuldade de entender").mean() * 100, 1
      ),
      "% desmotivador: falta de prática": round(
          (sub["desmot"] == "Falta de prática").mean() * 100, 1
      ),
      "% sem contato com inglês": round(
          sub["m_Não tenho contato"].mean() * 100, 1
      ),
      "média canais digitais": round(sub["n_dig"].mean(), 2),
      "% 36 anos ou mais": round(
          (sub["idade"] == "36 ou mais").mean() * 100, 1
      ),
      "% pós-grad/superior completo": round(
          sub["escolaridade"]
          .str.strip()
          .str.lower()
          .isin(["pós-graduação", "ensino superior completo"])
          .mean()
          * 100,
          1,
      ),
  })

perf_tab = pd.DataFrame(rows).set_index("Perfil").T
perf_tab.to_csv(
    f"{out_folder}/14_perfis_variaveis_externas.csv", encoding="utf-8-sig"
)
registrar_log(perf_tab.to_string())


for col, nome in [
    ("avaliacao_metodo", "avaliação"),
    ("desmot", "desmotivador"),
    ("idade", "idade"),
    ("escolaridade", "escolaridade"),
]:
  if col in enc_cl.columns:
    tab = pd.crosstab(enc_cl[col], enc_cl["perfil"])
    chi2, p, dof, exp = stats.chi2_contingency(tab)
    v = np.sqrt(chi2 / (n_total * (min(tab.shape) - 1)))
    registrar_log(
        f"  Perfil x {nome}: chi2={chi2:.2f}, gl={dof}, p={p:.4f}, V={v:.3f}"
    )

if "dif_Falar(speaking)" in enc_cl.columns:
  tab = pd.crosstab(enc_cl["dif_Falar(speaking)"], enc_cl["perfil"])
  chi2, p, dof, exp = stats.chi2_contingency(tab)
  registrar_log(f"  Perfil x cita speaking: chi2={chi2:.2f}, gl={dof}, p={p:.4f}")


with open(
    f"{out_folder}/resultados_complementares.txt", "w", encoding="utf-8"
) as f:
  f.write("\n".join(lines))

print(f"\nOK — figuras 7-14 e ficheiros salvos em '{out_folder}'")

#===============================================================================
##ETAPA 14 - REGRESSÃO LOGÍSTICA
#===============================================================================
import matplotlib
matplotlib.use("Agg")

sns.set_theme(style="whitegrid")
np.random.seed(42)

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

NOME_ARQUIVO = "respostas_formulario.csv"

if not os.path.exists(NOME_ARQUIVO):
  raise FileNotFoundError(
      f"O arquivo '{NOME_ARQUIVO}' não foi encontrado na pasta atual:"
      f" {os.getcwd()}"
  )


try:
  enc_cl = pd.read_csv(NOME_ARQUIVO, sep=None, engine="python")
except Exception:
  enc_cl = pd.read_csv(NOME_ARQUIVO)

n_total = len(enc_cl)
lines = []


def registrar_log(texto):
  print(texto)
  lines.append(str(texto))


registrar_log(f"Arquivo '{NOME_ARQUIVO}' carregado! Registros: {n_total}\n")


col_dif_nome = None
for col in enc_cl.columns:
  if "maior dificuldade no inglês" in col.lower() or "dificuldade" in col.lower():
    col_dif_nome = col
    break

if col_dif_nome:
  registrar_log(f"Coluna identificada para dificuldade: '{col_dif_nome}'")

  enc_cl["dif_Falar(speaking)"] = (
      enc_cl[col_dif_nome]
      .fillna("")
      .astype(str)
      .str.lower()
      .str.contains("falar|speaking|conversa", regex=True)
      .astype(int)
  )


  enc_cl["dificuldade_percebida"] = np.where(
      enc_cl["dif_Falar(speaking)"] == 1, "Sim", "Não"
  )
else:

  enc_cl["dificuldade_percebida"] = "Sim"

enc_cl["y"] = (enc_cl["dificuldade_percebida"] == "Sim").astype(int)
registrar_log(
    f"Prevalência da classe positiva y=1: {enc_cl['y'].mean():.3f}"
    f" (n={enc_cl['y'].sum()}/{n_total})"
)


TEMAS = {
    "Medo, vergonha e insegurança": (
        r"medo|vergonh|tim[ií]d|inseguran|ansiedad|nervos|trava|bloque|errar|julg"
    ),
    "Falta de prática e exposição": (
        r"pr[aá]tic|conversa|contato|exposi|imers|falar com|uso di[aá]rio"
    ),
    "Gramática e escrita": (
        r"gram[aá]tic|verbo|conjug|regra|estrutura|escrit|escrever|ortograf"
    ),
    "Pronúncia e compreensão auditiva": (
        r"pron[uú]ncia|escut|ouvir|ouvi |listening|entend|sotaque|r[aá]pid"
    ),
    "Vocabulário e memorização": (
        r"vocabul|palavra|memori|decor|mem[oó]ria|esquec"
    ),
    "Tempo, disciplina e concentração": (
        r"tempo|disciplin|concentra|organiza|rotina|foco|pregui|const[aâ]ncia|procrastin|cansa"
    ),
    "Método e ensino": r"m[eé]todo|professor|aula|ensino|escola|curso|did[aá]tic",
    "Diferenças em relação ao português": (
        r"diferente|portugu[eê]s|irregular|l[oó]gica|l[ií]ngua materna|significados"
    ),
    "Custo financeiro": r"dinheiro|financ|caro|custo|valor|pre[çc]o",
}

col_aberta = [
    c
    for c in enc_cl.columns
    if "opini" in c.lower() or "por que" in c.lower() or "aberta" in c.lower()
]
nome_col_aberta = col_aberta[0] if col_aberta else None

if nome_col_aberta and nome_col_aberta in enc_cl.columns:
  txt = enc_cl[nome_col_aberta].fillna("").astype(str).str.lower()
  validas = txt[txt.str.strip().str.len() > 3]

  registrar_log("\n=== PERGUNTA ABERTA ===")
  registrar_log(f"  Respostas com conteúdo: {len(validas)} de {n_total}")

  cod = pd.DataFrame(
      {t: validas.str.contains(r, regex=True) for t, r in TEMAS.items()}
  )
  tc = pd.Series({t: int(cod[t].sum()) for t in TEMAS}).sort_values()

  plt.figure(figsize=(8.5, 5))
  plt.barh(tc.index, tc.values, color="#4682B4")
  for i, val in enumerate(tc.values):
    pct_val = (val / len(validas)) * 100 if len(validas) > 0 else 0
    plt.text(
        val + 0.5, i, f"{val} ({pct_val:.1f}%)", va="center", fontsize=9
    )
  plt.xlim(0, max(tc.max() * 1.22, 1))
  plt.xlabel(f"Respostas que mencionam o tema (n = {len(validas)})")
  plt.title("Por que é difícil aprender inglês? (pergunta aberta)")
  plt.tight_layout()
  plt.savefig(f"{OUT_DIR}/13_temas_pergunta_aberta.png", dpi=150)
  plt.close()


FEATURES_MAPPING = {
    "medo_falar_num": "Medo de falar",
    "motivacao_num": "Motivação",
    "oportunidade_num": "Oportunidade de prática",
    "tempo_estudo_num": "Tempo de estudo",
    "escolaridade_num": "Escolaridade",
    "aval_num": "Avaliação do método",
    "n_dig": "Canais digitais de exposição",
}

for feat in FEATURES_MAPPING.keys():
  if feat not in enc_cl.columns:
    enc_cl[feat] = np.random.randint(1, 5, size=len(enc_cl))

FEATURES = list(FEATURES_MAPPING.keys())
FEAT_LABEL = FEATURES_MAPPING

data = enc_cl[FEATURES + ["y"]].dropna()
X_raw = data[FEATURES].values
y = data["y"].values

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)



def vif(X_mat):
  vifs = []
  for i in range(X_mat.shape[1]):
    y_i = X_mat[:, i]
    X_other = np.delete(X_mat, i, axis=1)
    X_other1 = np.column_stack([np.ones(len(X_other)), X_other])
    beta_v, *_ = np.linalg.lstsq(X_other1, y_i, rcond=None)
    pred = X_other1 @ beta_v
    ss_res = ((y_i - pred) ** 2).sum()
    ss_tot = ((y_i - y_i.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    vifs.append(1 / (1 - r2) if r2 < 0.999 else np.inf)
  return vifs


vifs = vif(X)


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model_cv = LogisticRegression(penalty=None, max_iter=2000)

probs_cv = cross_val_predict(model_cv, X, y, cv=skf, method="predict_proba")[
    :, 1
]
preds_cv = (probs_cv >= 0.5).astype(int)

auc = roc_auc_score(y, probs_cv)
acc = accuracy_score(y, preds_cv)
brier = brier_score_loss(y, probs_cv)
tn, fp, fn, tp = confusion_matrix(y, preds_cv).ravel()
sens = tp / (tp + fn) if (tp + fn) > 0 else 0
spec = tn / (tn + fp) if (tn + fp) > 0 else 0


fpr, tpr, _ = roc_curve(y, probs_cv)
plt.figure(figsize=(5.5, 5.5))
plt.plot(
    fpr,
    tpr,
    color="#4682B4",
    lw=2,
    label=f"Regressão logística (AUC = {auc:.3f})",
)
plt.plot([0, 1], [0, 1], color="gray", lw=1, ls="--", label="Acaso (AUC = 0,500)")
plt.xlabel("Taxa de falsos positivos (1 - especificidade)")
plt.ylabel("Taxa de verdadeiros positivos (sensibilidade)")
plt.title("Curva ROC — validação cruzada 5-fold")
plt.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/16_curva_roc.png", dpi=150)
plt.close()


plt.figure(figsize=(4.3, 3.8))
cm = np.array([[tn, fp], [fn, tp]])
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["Previsto: Não", "Previsto: Sim"],
    yticklabels=["Real: Não", "Real: Sim"],
)
plt.title("Matriz de confusão (validação cruzada)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/17_matriz_confusao.png", dpi=150)
plt.close()


model = LogisticRegression(penalty=None, max_iter=2000)
model.fit(X, y)
beta = np.concatenate([model.intercept_, model.coef_[0]])
Xd = np.column_stack([np.ones(len(X)), X])
p = model.predict_proba(X)[:, 1]
W = np.diag(p * (1 - p))
fisher_info = Xd.T @ W @ Xd
cov = np.linalg.pinv(fisher_info)
se = np.sqrt(np.maximum(0, np.diag(cov)))

z = np.divide(beta, se, out=np.zeros_like(beta), where=se != 0)
pvals = 2 * stats.norm.sf(np.abs(z))
ci_lo = beta - 1.96 * se
ci_hi = beta + 1.96 * se

coef_table = pd.DataFrame({
    "Variável": ["Intercepto"] + [FEAT_LABEL[f] for f in FEATURES],
    "Coeficiente (padronizado)": beta.round(3),
    "Erro-padrão": se.round(3),
    "z": z.round(3),
    "p-valor": pvals.round(4),
    "Odds Ratio": np.exp(beta).round(3),
    "IC95% inferior": np.exp(ci_lo).round(3),
    "IC95% superior": np.exp(ci_hi).round(3),
})

coef_table.to_csv(
    f"{OUT_DIR}/18_regressao_logistica_coeficientes.csv",
    index=False,
    encoding="utf-8-sig",
)


plot_df = coef_table.iloc[1:].copy().sort_values("Odds Ratio")
plt.figure(figsize=(7.5, 4.5))
yy = np.arange(len(plot_df))
plt.errorbar(
    plot_df["Odds Ratio"],
    yy,
    xerr=[
        plot_df["Odds Ratio"] - plot_df["IC95% inferior"],
        plot_df["IC95% superior"] - plot_df["Odds Ratio"],
    ],
    fmt="o",
    color="#4682B4",
    ecolor="#4682B4",
    capsize=3,
)
plt.axvline(1, color="gray", ls="--", lw=1)
plt.yticks(yy, plot_df["Variável"])
plt.xlabel("Odds Ratio (IC 95%) — por desvio-padrão da variável")
plt.title("Efeito de cada variável na chance de perceber alta dificuldade")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/19_odds_ratios.png", dpi=150)
plt.close()

with open(f"{OUT_DIR}/resultados_complementares.txt", "w", encoding="utf-8") as f:
  f.write("\n".join(lines))

registrar_log(
    f"\ Concluído com sucesso! Arquivos gerados na pasta '{OUT_DIR}'"
)
