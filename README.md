# Hava Durumu Tahmini

Saatlik hava durumu verilerinden sonraki sıcaklığı tahmin etmek için hazırlanmış bir makine öğrenmesi projesidir. Projede zaman serisi verileri 24 saatlik pencerelere ayrılır ve Conv1D + LSTM tabanlı bir sinir ağı ile tahmin yapılır.

## Proje Akışı

1. `load_and_explore.py` veri setini yükler, temel bilgileri yazdırır ve sıcaklık dağılımı ile zaman serisi grafiklerini gösterir.
2. `preprocessing.py` veriyi temizler, aykırı değerleri filtreler, özellikleri ölçeklendirir ve 24 saatlik giriş dizileri oluşturur.
3. `training.py` hazırlanmış eğitim verileriyle modeli eğitir, eğitim grafiklerini gösterir ve modeli `weather_model.h5` olarak kaydeder.
4. `test.py` test verisiyle tahmin yapar, MSE ve RMSE metriklerini yazdırır ve gerçek/tahmin edilen sıcaklıkları karşılaştırır.

## Gereksinimler

- Python 3.9 veya üzeri
- TensorFlow
- NumPy
- Pandas
- Scikit-learn 1.4 veya üzeri
- SciPy
- Matplotlib
- Seaborn
- Joblib

Bağımlılıkları aşağıdaki komutla kurabilirsiniz:

```bash
pip install tensorflow numpy pandas scikit-learn scipy matplotlib seaborn joblib
```

## Veri Seti

Proje kök dizininde `weatherHistory.csv` dosyası bulunmalıdır. CSV dosyasında aşağıdaki sütunlar beklenir:

- `Formatted Date`
- `Temperature (C)`
- `Humidity`
- `Wind Speed (km/h)`
- `Pressure (millibars)`
- `Visibility (km)`
- `Apparent Temperature (C)`

Ön işleme sırasında `Summary`, `Precip Type` ve `Daily Summary` sütunları çıkarılır. Özellik adları küçük harfe dönüştürülür ve boşluklar `_` ile değiştirilir.

## Çalıştırma

Komutları proje kök dizininde ve aşağıdaki sırayla çalıştırın:

```bash
python load_and_explore.py
python preprocessing.py
python training.py
python test.py
```

Grafik pencerelerini kapattıktan sonra ilgili betik çalışmasını tamamlar. `training.py` tamamlandığında `weather_model.h5`, `test.py` çalıştığında ise MSE ve RMSE sonuçları üretilir.

## Model

Modelin girişinde son 24 saate ait şu özellikler kullanılır:

- Nem (`humidity`)
- Rüzgar hızı (`wind_speed_km/h`)
- Basınç (`pressure_millibars`)
- Görüş mesafesi (`visibility_km`)
- Hissedilen sıcaklık (`apparent_temperature_c`)

Model mimarisi:

- Conv1D: 64 filtre, kernel boyutu 3
- BatchNormalization
- Dropout: 0.2
- LSTM: 64 birim
- Dense: 1 çıktı

Model, Adam optimizer ve mean squared error kaybı ile 50 epoch boyunca eğitilir.

## Oluşturulan Dosyalar

Ön işleme ve eğitim sonrasında aşağıdaki dosyalar kullanılır veya oluşturulur:

- `x_train.npy`, `y_train.npy`: Eğitim verileri
- `x_test.npy`, `y_test.npy`: Test verileri
- `scaler.pkl`: Özellik ölçekleyici
- `weather_model.h5`: Eğitilmiş Keras modeli

## Notlar

- Veriler zaman sırası korunarak yüzde 80 eğitim ve yüzde 20 test olarak ayrılır.
- Aykırı değerler, seçilen özellikler ve hedef sıcaklık için Z-skoru 3 eşiği kullanılarak filtrelenir.
- Tahmin edilen değer, her 24 saatlik pencerenin ardından gelen sıcaklıktır.
- Eğitim ve değerlendirme betikleri grafik göstermek için Matplotlib kullanır.
