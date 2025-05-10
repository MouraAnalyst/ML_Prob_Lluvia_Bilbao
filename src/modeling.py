import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)


class Modeling:
    def __init__(self, train_df, test_df, features, target):
        self.features = features
        self.target = target
        self.X_train = train_df[features]
        self.y_train = train_df[target]
        self.X_test = test_df[features]
        self.y_test = test_df[target]
        self.models = {}

    def train_logistic_regression(self):
        """
        Entrena un modelo de Regresión Logística.
        """
        model = LogisticRegression(max_iter=1000)
        model.fit(self.X_train, self.y_train)
        self.models["Logistic Regression"] = model
        print("✔ Modelo Logistic Regression entrenado.")

    def train_random_forest(self):
        """
        Entrena un modelo de Random Forest.
        """
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(self.X_train, self.y_train)
        self.models["Random Forest"] = model
        print("✔ Modelo Random Forest entrenado.")

    def evaluate(self, model_name):
        """
        Evalúa el modelo especificado por su nombre.
        """
        if model_name not in self.models:
            print(f"❌ Modelo '{model_name}' no encontrado.")
            return

        model = self.models[model_name]
        y_pred = model.predict(self.X_test)
        y_proba = (
            model.predict_proba(self.X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        print(f"📈 Evaluación del modelo: {model_name}")
        print("Accuracy:", accuracy_score(self.y_test, y_pred))
        print("F1 Score:", f1_score(self.y_test, y_pred))
        print("\nClassification Report:\n", classification_report(self.y_test, y_pred))

        # Matriz de confusión
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Matriz de Confusión")
        plt.tight_layout()
        plt.show()

        # Curva ROC
        if y_proba is not None:
            auc = roc_auc_score(self.y_test, y_proba)
            fpr, tpr, _ = roc_curve(self.y_test, y_proba)
            plt.figure(figsize=(6, 4))
            plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.title("Curva ROC")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            print("⚠️ Este modelo no proporciona probabilidades (predict_proba).")

    def compare_models(self):
        """
        Compara los modelos entrenados en términos de Accuracy y F1 Score.
        """
        if not self.models:
            print("❌ No hay modelos entrenados para comparar.")
            return

        results = []
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            acc = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            results.append({"Modelo": name, "Accuracy": acc, "F1 Score": f1})

        results_df = pd.DataFrame(results)
        print(results_df)

        # Gráfico de comparación
        results_df.set_index("Modelo")[["Accuracy", "F1 Score"]].plot(
            kind="bar", figsize=(8, 5)
        )
        plt.title("Comparación de modelos")
        plt.ylabel("Puntuación")
        plt.grid(axis="y")
        plt.tight_layout()
        plt.show()
