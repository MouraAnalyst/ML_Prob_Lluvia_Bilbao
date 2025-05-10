# preprocessing.py

import pandas as pd


class Preprocessing:
    def __init__(self, df):
        """
        Inicializa la clase con un DataFrame.
        """
        self.df = df

    def drop_columns(self, columns):
        """
        Elimina las columnas especificadas del DataFrame.

        Args:
            columns (list): Lista de nombres de columnas a eliminar.
        """
        self.df.drop(columns=columns, inplace=True)
        print(f"Columnas eliminadas: {columns}")

    def fillna_value(self, column, value):
        """
        Rellena valores nulos en una columna con un valor específico.

        Args:
            column (str): Nombre de la columna.
            value: Valor con el que se rellenarán los nulos.
        """
        self.df[column] = self.df[column].fillna(value)
        print(f"Nulos en '{column}' rellenados con: {value}")

    def fillna_with_daily_mean(self, column):
        """
        Rellena nulos con la media diaria de la columna seleccionada.

        Args:
            column (str): Nombre de la columna a imputar.
        """
        self.df["date"] = pd.to_datetime(self.df["time"]).dt.date
        self.df[column] = self.df.groupby("date")[column].transform(
            lambda x: x.fillna(x.mean())
        )
        self.df.drop(columns=["date"], inplace=True)
        print(f"Nulos en '{column}' rellenados con la media diaria.")

    def create_rain_column(self, prcp_column="prcp", threshold=0.1):
        """
        Crea una columna binaria 'rain' a partir de 'prcp' (> threshold).

        Args:
            prcp_column (str): Nombre de la columna de precipitación.
            threshold (float): Umbral para considerar que ha llovido.
        """
        self.df[prcp_column] = self.df[prcp_column].fillna(0)
        self.df["rain"] = (self.df[prcp_column] > threshold).astype(int)
        print(
            f"Columna 'rain' creada a partir de '{prcp_column}' con umbral {threshold} mm."
        )

    def split_column(self, column, left_digits=0, right_digits=0):
        """
        Divide una columna string o datetime en dos nuevas columnas.
        Útil para separar fecha y hora o partes de un string numérico.

        Args:
            column (str): Nombre de la columna.
            left_digits (int): Cantidad de dígitos desde la izquierda.
            right_digits (int): Cantidad de dígitos desde la derecha.
        """
        if pd.api.types.is_datetime64_any_dtype(self.df[column]):
            self.df["fecha"] = self.df[column].dt.date
            self.df["hora"] = self.df[column].dt.hour
            self.df["day"] = self.df[column].dt.day
            self.df["month"] = self.df[column].dt.month
            self.df["year"] = self.df[column].dt.year
            print(f"Columna '{column}' dividida en 'fecha' y 'hora' (datetime).")
        else:
            self.df[f"{column}_left"] = self.df[column].astype(str).str[:left_digits]
            self.df[f"{column}_right"] = self.df[column].astype(str).str[-right_digits:]
            print(f"Columna '{column}' dividida en '{column}_left' y '{column}_right'.")

    def get_data(self):
        """
        Devuelve el DataFrame procesado.
        """
        return self.df

    def extract_date_features(self, column):
        self.df["day"] = self.df[column].dt.day
        self.df["month"] = self.df[column].dt.month
        self.df["year"] = self.df[column].dt.year
        print("✔ Variables temporales extraídas: hour, month, weekday, day, dayofyear")
