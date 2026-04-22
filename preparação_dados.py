import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

sns.set_theme(style="whitegrid")
os.makedirs("output", exist_ok=True)

# 1. Ler os dados
df = pd.read_csv("CVD_cleaned.csv")

print("\n--- SHAPE ORIGINAL ---")
print(df.shape)

print("\n--- INFO ORIGINAL ---")
df.info()

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- DUPLICADOS ---")
duplicados = df.duplicated().sum()
print(duplicados)
print((duplicados * 100) / len(df))

# 2. Remover duplicados
df_clean = df.drop_duplicates().copy()

print("\n--- SHAPE ORIGINAL ---")
print(df.shape)

print("\n--- SHAPE APÓS LIMPEZA ---")
print(df_clean.shape)

print("\n--- HEAD df_clean ---")
print(df_clean.head())

# 3. Dataset completamente categórico
df_cat = df_clean.copy()

df_cat["BMI_cat"] = pd.cut(
    df_cat["BMI"],
    bins=[0, 18.5, 25, 30, 40, 100],
    labels=["Baixo peso", "Normal", "Excesso de peso", "Obesidade", "Obesidade severa"]
)

df_cat["Height_cat"] = pd.cut(
    df_cat["Height_(cm)"],
    bins=[0, 150, 165, 180, 250],
    labels=["Baixa", "Média-baixa", "Média-alta", "Alta"]
)

df_cat["Weight_cat"] = pd.cut(
    df_cat["Weight_(kg)"],
    bins=[0, 60, 80, 100, 300],
    labels=["Leve", "Médio", "Pesado", "Muito pesado"]
)

df_cat["Alcohol_cat"] = pd.cut(
    df_cat["Alcohol_Consumption"],
    bins=[-1, 0, 5, 15, 30],
    labels=["Nenhum", "Baixo", "Moderado", "Alto"]
)

df_cat["Fruit_cat"] = pd.cut(
    df_cat["Fruit_Consumption"],
    bins=[-1, 10, 20, 40, 120],
    labels=["Baixo", "Médio", "Alto", "Muito alto"]
)

df_cat["GreenVeg_cat"] = pd.cut(
    df_cat["Green_Vegetables_Consumption"],
    bins=[-1, 5, 15, 30, 130],
    labels=["Baixo", "Médio", "Alto", "Muito alto"]
)

df_cat["FriedPotato_cat"] = pd.cut(
    df_cat["FriedPotato_Consumption"],
    bins=[-1, 2, 8, 20, 130],
    labels=["Baixo", "Médio", "Alto", "Muito alto"]
)

print("\n--- HEAD df_cat (BMI, Height e Weight) ---")
print(df_cat[[
    "BMI", "BMI_cat",
    "Height_(cm)", "Height_cat",
    "Weight_(kg)", "Weight_cat"
]].head())

print("\n--- HEAD df_cat (consumos alimentares) ---")
print(df_cat[[
    "Alcohol_Consumption", "Alcohol_cat",
    "Fruit_Consumption", "Fruit_cat",
    "Green_Vegetables_Consumption", "GreenVeg_cat",
    "FriedPotato_Consumption", "FriedPotato_cat"
]].head())

# 4. Dataset completamente numérico
df_num = df_clean.copy()
label_encoders = {}

for col in df_num.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df_num[col] = le.fit_transform(df_num[col])
    label_encoders[col] = le

print("\n--- HEAD df_num ---")
print(df_num.head())

# 5. Normalização MinMax
num_cols = df_clean.select_dtypes(exclude="object").columns
df_minmax = df_clean.copy()
minmax = MinMaxScaler()
df_minmax[num_cols] = minmax.fit_transform(df_minmax[num_cols])

print("\n--- HEAD df_minmax ---")
print(df_minmax.head())

# 6. Normalização Z-score
df_std = df_clean.copy()
standard = StandardScaler()
df_std[num_cols] = standard.fit_transform(df_std[num_cols])

print("\n--- HEAD df_std / Z-score ---")
print(df_std.head())

# 5. Normalização MinMax
df_minmax = df_clean.copy()
minmax = MinMaxScaler()
df_minmax[num_cols] = minmax.fit_transform(df_minmax[num_cols])

print("\n--- HEAD df_minmax ---")
print(df_minmax.head())

# 6. Normalização Z-score
df_zscore = df_clean.copy()
standard = StandardScaler()
df_zscore[num_cols] = standard.fit_transform(df_zscore[num_cols])

print("\n--- HEAD df_zscore / Z-score ---")
print(df_zscore.head())


# =========================
# VISUALIZAÇÕES
# =========================

# 1. Duplicados
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(["Original", "Sem duplicados"], [len(df), len(df_clean)], color=["steelblue", "seagreen"])
ax.set_title("Dimensão do dataset antes e depois da remoção de duplicados")
ax.set_ylabel("Número de linhas")
for i, v in enumerate([len(df), len(df_clean)]):
    ax.text(i, v + 500, f"{v:,}".replace(",", "."), ha="center", fontsize=10)
plt.tight_layout()
plt.savefig("output/01_duplicados.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()

# 2. Dataset categórico
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.countplot(
    data=df_cat,
    x="BMI_cat",
    hue="BMI_cat",
    order=df_cat["BMI_cat"].value_counts().index,
    palette="viridis",
    ax=axes[0, 0],
    legend=False
)
axes[0, 0].set_title("Distribuição de BMI categorizado")
axes[0, 0].tick_params(axis="x", rotation=30)

sns.countplot(
    data=df_cat,
    x="Height_cat",
    hue="Height_cat",
    order=df_cat["Height_cat"].value_counts().index,
    palette="viridis",
    ax=axes[0, 1],
    legend=False
)
axes[0, 1].set_title("Distribuição de Height categorizado")
axes[0, 1].tick_params(axis="x", rotation=20)

sns.countplot(
    data=df_cat,
    x="Weight_cat",
    hue="Weight_cat",
    order=df_cat["Weight_cat"].value_counts().index,
    palette="viridis",
    ax=axes[1, 0],
    legend=False
)
axes[1, 0].set_title("Distribuição de Weight categorizado")
axes[1, 0].tick_params(axis="x", rotation=20)

sns.countplot(
    data=df_cat,
    x="Alcohol_cat",
    hue="Alcohol_cat",
    order=df_cat["Alcohol_cat"].value_counts().index,
    palette="viridis",
    ax=axes[1, 1],
    legend=False
)
axes[1, 1].set_title("Distribuição de Alcohol categorizado")
axes[1, 1].tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.savefig("output/02_dataset_categorico.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()

cols_para_escala = num_cols

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
df_minmax[cols_para_escala].boxplot()
plt.title('Normalização Min-Max (0 a 1)')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
df_zscore[cols_para_escala].boxplot()
plt.title('Normalização Z-Score (Média 0, Desvio 1)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("output/07_boxplots_escala.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()

# Guardar a versão que escolheste para a modelação (Z-Score)
df_zscore.to_csv('CVD_numerico_zscore.csv', index=False)
print("Ficheiro 'CVD_numerico_zscore.csv' guardado. Pronto para a modelação!")

# 3. Dataset numérico
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, col in zip(axes, ["Exercise", "Heart_Disease", "Sex"]):
    x = np.arange(5)
    ax.plot(x, df_clean[col].head().astype(str), marker="o", label="Original")
    ax.plot(x, df_num[col].head(), marker="x", label="Codificado")
    ax.set_title(col)
    ax.set_xlabel("Amostras")
    ax.legend()
plt.tight_layout()
plt.savefig("output/03_dataset_numerico.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()

# 4. Histogramas BMI
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
df_clean["BMI"].hist(ax=axes[0], bins=30, color="steelblue")
axes[0].set_title("BMI original")
axes[0].set_xlabel("BMI")
axes[0].set_ylabel("Frequência")

df_minmax["BMI"].hist(ax=axes[1], bins=30, color="orange")
axes[1].set_title("BMI MinMax")
axes[1].set_xlabel("BMI")
axes[1].set_ylabel("Frequência")

df_std["BMI"].hist(ax=axes[2], bins=30, color="green")
axes[2].set_title("BMI Z-Score")
axes[2].set_xlabel("BMI")
axes[2].set_ylabel("Frequência")

plt.tight_layout()
plt.savefig("output/04_bmi_hist_norm.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()

# 5. Boxplots BMI
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

sns.boxplot(x=df_clean["BMI"], ax=axes[0], color="steelblue")
axes[0].set_title("BMI original")
axes[0].set_xlabel("BMI")

sns.boxplot(x=df_minmax["BMI"], ax=axes[1], color="orange")
axes[1].set_title("BMI MinMax")
axes[1].set_xlabel("BMI")

sns.boxplot(x=df_std["BMI"], ax=axes[2], color="green")
axes[2].set_title("BMI Z-Score")
axes[2].set_xlabel("BMI")

plt.tight_layout()
plt.savefig("output/05_bmi_box_norm.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()

# 6. Correlação
plt.figure(figsize=(12, 10))
corr = df_clean.select_dtypes(exclude="object").corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, linewidths=0.3)
plt.title("Correlação entre variáveis numéricas")
plt.tight_layout()
plt.savefig("output/06_correlacao.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()

# =========================
# COMPARAÇÕES BEFORE / AFTER
# =========================

with pd.option_context(
    'display.max_columns', None,
    'display.max_colwidth', None,
    'display.width', 1000,
    'display.expand_frame_repr', False
):
    print("\n--- BEFORE / AFTER df_num ---")
    print(pd.concat([
        df_clean.head(),
        df_num.head()
    ], axis=1, keys=["Original", "Codificado"]))

print("\n--- BEFORE / AFTER MinMax ---")
print(pd.concat([
    df_clean[["BMI"]].head(),
    df_minmax[["BMI"]].head()
], axis=1, keys=["Original", "MinMax"]))

print("\n--- BEFORE / AFTER Z-Score ---")
print(pd.concat([
    df_clean[["BMI"]].head(),
    df_std[["BMI"]].head()
], axis=1, keys=["Original", "Z-Score"]))



print("\n--- PREPARAÇÃO DOS DADOS CONCLUÍDA ---")
print("df_clean:", df_clean.shape)
print("df_cat:", df_cat.shape)
print("df_num:", df_num.shape)
print("df_minmax:", df_minmax.shape)
print("df_std:", df_std.shape)