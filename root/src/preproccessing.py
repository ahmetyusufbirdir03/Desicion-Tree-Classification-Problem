import pandas as pd

def clean_data(df):
    
    # Çok eksik olduğu için analiz dışı bırakılan sütun
    df.drop(columns=["Lichens_cover"], inplace=True)

    # " Canopy_diameter" sütunu metin olarak geldiği için, sayısala çevriliyor
    df["Canopy_diameter"] = df["Canopy_diameter"].astype(str).str.replace(",", ".")
    df["Canopy_diameter"] = pd.to_numeric(df["Canopy_diameter"], errors="coerce")

    # Eksik değerleri uygun yöntemle doldurma
    # - Sayısal olanlar: Ortalama ile
    df["Height"] = df["Height"].fillna(df["Height"].mean())
    df["Canopy_diameter"] = df["Canopy_diameter"].fillna(df["Canopy_diameter"].mean())
    df["Rugosity2"] = df["Rugosity2"].fillna(df["Rugosity2"].mean())
    df["Inclination"] = df["Inclination"].fillna(df["Inclination"].mean())

    # - Kategorik olan (veya sınırlı değerli): Mod (en sık görülen değer) ile
    df["Aspect"] = df["Aspect"].fillna(df["Aspect"].mode()[0])
    
    return df

# Excel/CSV dosyasını okuma
df = pd.read_csv("raw-data-walentowitz-etal-REVISED.csv")

# Veri setinin bir kopyasını oluştur (orijinali korumak için)
df_cleaned = df.copy()

# Temizleme işlemini fonksiyonla yap
df_cleaned = clean_data(df_cleaned)

# Temizlendi mi kontrol et (0 olmalı)
print("Eksik veri kalan sütunlar:")
print(df_cleaned.isnull().sum()[df_cleaned.isnull().sum() > 0])

# Temizlenmiş veri setini bir CSV dosyasına kaydet
df_cleaned.to_csv("cleaned_data.csv", index=False)
print("Temizlenmiş veri seti 'cleaned_data.csv' olarak kaydedildi.")
