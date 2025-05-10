# eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class EDA:
    def __init__(self, df):
        """
        Inicializa el objeto EDA con un DataFrame.
        """
        self.df = df

    def explore(self):
        """
        Muestra información general del DataFrame.
        """
        print("📋 Información general del dataset:")
        print(self.df.info())
        print("\n📊 Estadísticas descriptivas:")
        print(self.df.describe())
        print("\n❓ Valores nulos por columna:")
        print(self.df.isnull().sum())

    def plot_distributions(self, features):
        """
        Grafica distribuciones (histogramas) para las variables numéricas.

        Args:
            features (list): Lista de columnas numéricas a visualizar.
        """
        for feature in features:
            plt.figure(figsize=(8, 4))
            sns.histplot(self.df[feature], kde=True, bins=30)
            plt.title(f"Distribución de {feature}")
            plt.xlabel(feature)
            plt.ylabel("Frecuencia")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def plot_boxplots(self, features, target=None):
        """
        Muestra boxplots de variables numéricas, diferenciadas por una variable target si se desea.

        Args:
            features (list): Columnas numéricas.
            target (str): Variable categórica (como 'rain') para diferenciar.
        """
        for feature in features:
            plt.figure(figsize=(8, 4))
            if target:
                sns.boxplot(x=target, y=feature, data=self.df)
                plt.title(f"{feature} por {target}")
            else:
                sns.boxplot(y=feature, data=self.df)
                plt.title(f"Boxplot de {feature}")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def plot_correlation_matrix(self, features):
        """
        Dibuja un heatmap de correlaciones entre variables numéricas.

        Args:
            features (list): Columnas a analizar.
        """
        plt.figure(figsize=(10, 8))
        corr = self.df[features].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Matriz de correlaciones")
        plt.tight_layout()
        plt.show()
