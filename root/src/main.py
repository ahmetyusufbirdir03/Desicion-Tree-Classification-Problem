import tkinter as tk
from tkinter import filedialog
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from random_forest import run_random_forest 
from decision_tree import run_decision_tree  

def select_and_run():
    file_path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx")])
    if not file_path:
        return

    df = pd.read_excel(file_path)

    # Encode categorical
    label_encoders = {}
    for col in ["Rock_type"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df[["Inclination", "Latitude", "Rugosity1", "Rugosity2", "Species_cover"]]
    y = df["Rock_type"]
    class_labels = list(np.unique(y))

    # Görseller klasörü oluştur
    base_dir = r"./results"
    img_dir = os.path.join(base_dir, "gorseller")
    result_dir = os.path.join(base_dir, "results")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # Histogramlar
    for col in X.columns:
        plt.figure()
        sns.histplot(X[col], kde=True, bins=20, color="skyblue")
        if(col== "Species_cover"):
            plt.xticks(ticks=range(0, 301, 25))
        plt.title(f"{col} Dağılımı")
        plt.xlabel(col)
        plt.ylabel("Frekans")
        plt.tight_layout()
        plt.savefig(os.path.join(img_dir, f"{col}_histogram.png"))
        plt.close()

    # Boxplotlar
    for col in X.columns:
        plt.figure()
        sns.boxplot(x=y, y=X[col], palette="Set2")
        plt.title(f"{col} vs Rock_type")
        plt.xlabel("Rock_type")
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(os.path.join(img_dir, f"{col}_boxplot.png"))
        plt.close()

    # Korelasyon matrisi
    plt.figure(figsize=(8, 6))
    corr = X.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Korelasyon Matrisi")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "korelasyon_matrisi.png"))
    plt.close()

    # Rock_type dağılımı
    plt.figure()
    sns.countplot(x=y, palette="pastel")
    plt.title("Rock_type Sınıf Dağılımı")
    plt.xlabel("Rock_type")
    plt.ylabel("Adet")
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "rock_type_dagilimi.png"))
    plt.close()

    print(f"Görseller kaydedildi: {img_dir}")

    # Modeli çalıştır
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
    run_decision_tree(X, X_train, X_test, y_train, y_test, class_labels, result_dir)
    run_random_forest(X, X_train, X_test, y_train, y_test, class_labels, result_dir)


# GUI başlat
root = tk.Tk()
root.title("Rock Type Sınıflandırma Karşılaştırması")
root.geometry("400x200")
tk.Label(root, text="Veri dosyasını seçin ve modelleri başlatın", font=("Arial", 12)).pack(pady=20)
tk.Button(root, text="📂 Dosya Seç ve Başlat", font=("Arial", 11), command=select_and_run).pack()
root.mainloop()
