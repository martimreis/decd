import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# 1. Ler os dados
df = pd.read_csv("CVD_cleaned.csv")

# 2. Visão geral
print("\n--- SHAPE ---")
print(df.shape)

print("\n--- INFO ---")
print(df.info())

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- DUPLICADOS ---")
duplicados = df.duplicated().sum()
print(duplicados)
print((duplicados * 100) / len(df))

# 3. Describe geral
print("\n--- DESCRIBE NUMÉRICAS ---")
print(df.describe())

print("\n--- DESCRIBE CATEGÓRICAS ---")
print(df.describe(include="object"))

# 4. Separar variáveis
cat_cols = df.select_dtypes(include="object").columns
num_cols = df.select_dtypes(exclude="object").columns

print("\n--- VARIÁVEIS CATEGÓRICAS ---")
print(list(cat_cols))

print("\n--- VARIÁVEIS NUMÉRICAS ---")
print(list(num_cols))

# 5. Variáveis prioritárias para o relatório
vars_prioritarias = [
    "Age_Category",
    "BMI",
    "Smoking_History",
    "Exercise",
    "Diabetes",
    "Heart_Disease"
]


print("\n--- DESCRIBE VARIÁVEIS PRIORITÁRIAS ---")
print(df[vars_prioritarias].describe(include='all').round(2))
for col in vars_prioritarias:
    if df[col].dtype == 'object':
        print(f"\n{col}: top={df[col].mode().iloc[0]}, freq={df[col].value_counts().max()}")

print("\n--- VALUE COUNTS VARIÁVEIS PRIORITÁRIAS ---")
for col in ["Age_Category", "Smoking_History", "Exercise", "Diabetes", "Heart_Disease"]:
    print(f"\n--- VALUE COUNTS: {col} ---")
    print(df[col].value_counts())

# 6. Frequências de todas as variáveis categóricas
for col in cat_cols:
    print(f"\n--- VALUE COUNTS: {col} ---")
    print(df[col].value_counts())

# 7. Correlações gerais
corr = df[num_cols].corr()
print("\n--- CORRELAÇÕES ---")
print(corr)

# 8. Correlações relevantes para o relatório
print("\n--- CORRELAÇÕES RELEVANTES ---")
corr_relevantes = df[["BMI", "Weight_(kg)", "Height_(cm)"]].corr()
print(corr_relevantes)

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlação")
plt.tight_layout()
plt.show()

# 9. Histogramas das variáveis numéricas
df[num_cols].hist(figsize=(14, 10), bins=30)
plt.suptitle("Distribuições das Variáveis Numéricas")
plt.tight_layout()
plt.show()

# 10. Boxplots para detetar outliers
plt.figure(figsize=(14, 10))
for i, col in enumerate(num_cols, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(y=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()

# 11. Gráficos de barras para variáveis prioritárias
for col in ["Age_Category", "Smoking_History", "Exercise", "Diabetes", "Heart_Disease"]:
    plt.figure(figsize=(8, 4))
    df[col].value_counts().plot(kind="bar")
    plt.title(f"Distribuição de {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 12. Relações com o target Heart_Disease
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Heart_Disease", y="BMI")
plt.title("BMI por Heart_Disease")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="Age_Category", hue="Heart_Disease")
plt.title("Heart_Disease por Age_Category")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 13. Outliers com IQR
print("\n--- OUTLIERS (IQR) ---")
for col in num_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col}: {len(outliers)} outliers")