# GATE 1 LOG — PELSTM (Hafta 1-2)

Tekrar üretmek için: `.venv/bin/python scripts/run_gate1.py` (tam stdout: `outputs/gate1/run_gate1_stdout.log`,
trial bazlı survey: `outputs/gate1/gate1_survey.json`).
Ortam: Python 3.13.12 (pyenv), `.venv/`; `requirements.txt` → numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2, torch 2.14.0 (pinned).

---

## Bölüm A — Teknik Kararlar

### A.0 Spec ile veri arasındaki uyuşmazlıklar (FLAG)

| # | Spec varsayımı | Gerçek veri | Etki / karar |
|---|---|---|---|
| F1 | Tek `.mat`, `data/raw/` altında; `s.Data` uzunluğu `L` | 50 ayrı dosya: `data/Subject1.mat` … `Subject50.mat`, her birinde ayrı bir `s`. `L` denek başına: ör. Subject6 → `L = 45` (tüm task'lar), toplam 1615 trial | Loader denek numarası alıyor (`data_io.load_subject(k)`). Dosyalar taşınmadı. |
| F2 | `s.EMGVarName` | Alan adı **`s.EmgVarName`** | Kodda doğru ad kullanılıyor. |
| F3 | `.TimeStampEMG/Kin/Grf` zaman vektörleri; rate bunlardan türetilecek | Alan adı `.TimeStampEmg`; üçü de **skaler** (stride'ın orijinal kayıttaki başlangıç anı, ör. 3.59 s). Zaman vektörü yok | Rate, örnek sayılarından türetildi (A.3). Kin ve Grf timestamp'leri 183/844 trial'da farklı (Grf 1–42 ms geç başlıyor); EMG timestamp'i her trial'da Kin ile aynı. |
| F4 | `.EMG` 8×N "ham zaman serisi" (sürekli kayıt) | `.EMG` ham (rectify/filtre yok) ama **tek bir stride'a kırpılmış**: süresi = 120/cadence (oran medyanı 1.001, N örnek = stride + 1). `.Marker` da aynı stride'ı kapsıyor. `.Grf` yalnızca stance'ı kapsıyor (stride'ın ~%53–75'i) | Hizalama çok basitleşiyor (A.4), ama her stride'ın ilk ~150 ms'si için causal feature yok (A.5). |
| F5 | EMG, kinematikle aynı taraf | EMG her zaman tek taraftan: `s.EMGSide = 'RX'` (49 denek), **Subject18 → `'NO'`** (EMG verisi var ama taraf bilgisi yok). Walking trial'ları: 581 `Foot='RX'`, 263 `Foot='LX'`. `LX` trial'larında `.Ang/.Mom` **sol** ayak bileği, `.EMG` ise **sağ** bacak kasları (sol stride'a kırpılmış) | Sağ-taraf protez kontrolcüsü için tutarlı eşleşme yalnızca `Foot == EMGSide == 'RX'` → **572 trial** (+ S18'den 9 belirsiz trial). Gate 1'de kaydedilen tüm trial'lar RX. Hafta 3'te hangi alt kümenin kullanılacağına karar verilmeli. |
| F6 | EMG sıfır-ortalamalı ham sinyal | Birçok trial'da belirgin **DC offset / baseline drift** var (ör. S40 tüm kanallarda ~+0.037 mV; S19'da dalgalanan baseline). GMax/VM için |ortalama| / ortalama-mutlak-AC oranı: medyan ≈ 1.0, %90'lık dilim 4.6. Ayrıca düşük frekanslı hareket artefaktları var (S01 GMax) | IEMG = Σ\|x\| offset'i doğrudan içine alır (bias). Spec gereği feature'lar **ham** sinyal üzerinde, filtre/demean yok. Hafta 3 öncesi karar verilmeli: bandpass (ör. 20–450 Hz) veya causal high-pass/demean. |
| F7 | Rate'ler 800/960/1000 Hz | Doğrulandı: 3 konfigürasyon (Kin, Grf, EMG): (60, 960, 960) → 25 denek, (200, 800, 800) → 13 denek, (200, 800, 1000) → 12 denek. Walking trial'ları: 960 Hz → 324, 800 Hz → 273, 1000 Hz → 247 | Beklendiği gibi. Not: 25 denekte kinematik 60 Hz ile örneklenmiş (`.Ang/.Mom` 101 noktaya zaten normalize, bu nedenle hedefleri etkilemiyor). |
| F8 | — | Ayak bileği açısı (`AnkleFlx`) sistematik olarak negatif: heel strike'taki değerin medyanı −22.5° (%5–95: −28.3…−16.2°), ROM medyanı 32.8°. Tipik klinik konvansiyonda heel-strike ≈ 0° | Muhtemelen nötr/statik ofset çıkarılmamış bir konvansiyon. Hedefler **dokunulmadan** bırakıldı; Hafta 3'te denek-başı ofset ya da statik trial (`s.StandingData`) ile düzeltme düşünülmeli. Moment makul: tepe medyanı 1.28 Nm/kg, swing'de ≈ 0. |
| F9 | — | Walking task'ı çoklu hız içeriyor: stride süresi 0.683–2.945 s (medyan 1.078 s). En uzun: S08_D018, hız 0.39 m/s, cadence 41 adım/dk (gerçek bir yavaş yürüme, hata değil). S08'in hızları 0.29–1.51 m/s aralığında | Pipeline bu uç değerleri sorunsuz işliyor (A.6). Hız, Hafta 3 split/stratifikasyonu için önemli bir kovaryat. |
| F10 | — | 2 trial'da `cadence` alanı EMG uzunluğuyla tutarsız: S12_D016 (cadence'tan türetilen rate 785.2 Hz, nominal 800), S35_D000 (931.3 vs 960). Marker tabanlı türetme her ikisinde de nominal rate'i tam veriyor | Bozuk olan `cadence` metadata'sı, EMG değil. Trial'lar dışlanmadı, flag'lendi. |
| F11 | Görev: "GATE1_SPEC.md dosyasını oku" | Diskte `GATE1_SPEC.md` yok; spec, sohbet mesajında verildi | Spec mesajdan uygulandı. |

### A.1 Level-walking trial sayısı
`.Task == 'Walking'` (tam eşleşme; `ToeWalking`/`HeelWalking` hariç): **844** — beklenen ~844 ile **eşleşiyor**.
Denek başına 2 (S36) ile 39 (S11, S19) arası. Diğer task'lar: ToeWalking 278, HeelWalking 178, StepUp 169, StepDown 146 (toplam 1615).

### A.2 Kanal indeksleri (veriden ampirik çıkarıldı)
`s.EmgVarName` sırası **50 deneğin hepsinde aynı** (1 farklı sıra):
`0 Tibialis Anterior, 1 Soleus, 2 Gastrocnemius Medialis, 3 Peroneus Longus, 4 Rectus Femoris, 5 Vastus Medialis, 6 Biceps Femoris, 7 Gluteus Maximus`

- **GMax = indeks 7** (0-tabanlı; MATLAB'da 8)
- **VM = indeks 5** (0-tabanlı; MATLAB'da 6)

Kod bu indeksleri sabit yazmıyor; her denek için `data_io.channel_index(s, 'EmgVarName', name)` ile isimden buluyor.
Hedefler: `AngVarName` → `AnkleFlx` = 9, `MomVarName` → `AnkleFlxMom` = 6 (her ikisi de isimden bulunuyor; `AngVarName`/`MomVarName`/`GrfVarName` sırası da 50 denekte aynı).

Shape'ler (Subject6, `Data[0]`, Walking, RX): `.EMG` (8, 821) @1000 Hz, `.Ang` (12, 101), `.Mom` (9, 101), `.Grf` (9, 378) @800 Hz, `.Marker` (87, 165) @200 Hz.
844 walking trial'ın tümünde `.Ang` (12, 101), `.Mom` (9, 101), `.EMG` 8 satır, `.Grf` 9 satır; EMG/Ang/Mom'da NaN yok.
**Temel fark:** `.Ang/.Mom` %stride tabanında (101 nokta, 0–100 %), `.EMG/.Grf` kendi native rate'lerinde ham örnekler (EMG N = 569…2946).

### A.3 Örnekleme hızları (trial bazında türetildi)
TimeStamp'ler skaler olduğu için (F3) rate, trial başına üç bağımsız yoldan hesaplanıp birbiriyle karşılaştırılıyor (`data_io.derive_rates`):
1. Nominal: `s.EMGFreq` (denek header'ı)
2. Marker'dan: `KinFreq · (N_emg − 1) / (N_marker − 1)` (iki dizi aynı stride'ı kapsıyor)
3. Cadence'tan: `(N_emg − 1) / (120 / cadence)`

844 trial'ın 842'sinde üç tahmin %1 içinde tutarlı; cadence-tabanlı tahmin ile nominal oranı: medyan 1.0000, min 0.970 (F10'daki 2 trial). Downstream'de kullanılan rate = trial-başı doğrulanmış nominal değer (800/960/1000 Hz). Stride süresi = (N_emg − 1) / rate.

### A.4 Hizalama yönü — karar ve gerekçe
**Karar:** Önce feature dizisi native EMG zaman ekseninde çıkarılıyor, sonra feature dizisi 101 noktalık %stride grid'ine resample ediliyor. `.Ang/.Mom` hedefleri **hiç resample edilmiyor**.

**Eşleme:** EMG stride'a kırpılmış olduğu için (F4) N örneklik EMG'nin k. örneği → `100 · k / (N − 1)` % stride (k = 0 heel strike, k = N−1 bir sonraki heel strike). Bir pencerenin feature'ı, **son örneğinin** zamanına damgalanıyor (causal). Bu %stride değerlerinden 101-grid'e lineer interpolasyon yapılıyor (`align.resample_to_stride_grid`). Ekstrapolasyon yok: yalnızca [ilk pencere sonu, son pencere sonu] aralığındaki grid noktaları tutuluyor.

**Gerekçe:**
- Hedefler, kontrolcünün tahmin edeceği referans. Onları interpolasyonla bozmak, loss'u bizim yarattığımız bir artefakta göre optimize etmek olur. Orijinal yazarların 101 noktalık normalizasyonu olduğu gibi kalıyor.
- Ham EMG'yi önce 101 noktaya (≈ 7–29 örnekte 1) indirmek aliasing'e yol açar ve IEMG/WL'nin tanımını bozar (özellikle WL örnekleme hızına duyarlı). Feature'lar native rate'te, 150 ms'lik fiziksel pencerede hesaplanmalı.
- Causal (pencere-sonu) damgalama, gerçek zamanlı protez kontrolcüsünde feature'ın ne zaman hazır olacağını temsil ediyor. Merkeze damgalamak 75 ms gelecek bilgisi sızdırırdı.
- `extract_features()` hizalamadan bağımsız kalıyor. Aynı fonksiyon ileride sürekli (stride'a kırpılmamış) bir sinyale de uygulanabilir.

### A.5 Windowing parametreleri
- **Pencere: 150 ms** → 800 Hz: 120, 960 Hz: 144, 1000 Hz: 150 örnek (trial-başı rate'ten hesaplanıyor).
- **Hop (stride): 25 ms** → 20 / 24 / 25 örnek; **overlap: 125 ms (%83.3)**.
  - Gerekçe: 25 ms, üç rate'te de tam sayı örneğe denk gelen **en küçük** hop (1 / gcd(800, 960, 1000) = 1/40 s). Böylece zaman çözünürlüğü denekler arası birebir aynı ve yuvarlama kayması yok. 40 Hz feature güncellemesi, bir stride'ın (~1 s) yaklaşık her %2.5'ine bir örnek demek. Ayrıca gerçekçi bir kontrolcü güncelleme hızı.
  - `window_layout` pencere veya hop tam sayı örnek değilse hata fırlatıyor (ör. 1024 Hz).
- Yalnızca tam pencereler kullanılıyor (padding yok). Pencere sayısı T = ⌊(N − W) / H⌋ + 1.
- **Sonuç / sınırlama:** EMG stride'a kırpılmış olduğundan her stride'ın başındaki 150 ms için causal pencere yok. Hizalanmış grid 844 trial'da en erken %6 (yavaş yürüme) ile %22 (hızlı) arasında başlıyor (medyan %14), en geç %97–100'de bitiyor. Hizalı uzunluk T = 77…94 nokta (medyan 85). Bu noktalar uydurulmadı, düşürüldü. Hafta 3'te karar verilecek seçenekler: (a) bu kaybı kabul etmek, (b) aynı yürüyüşten ardışık stride'ları birleştirmek (timestamp'ler buna yetmeyebilir), (c) sıfır/yansıtma padding'i.

### A.6 Feature modülü ve Gate 1 doğrulaması
- `src/data/features.py::extract_features(emg_channels (C,N), rate, window_ms=150, hop_ms=25) -> (T, 2C)`. Sütun sırası `[IEMG_c0, WL_c0, IEMG_c1, WL_c1]`. Saf NumPy, framework bağımlılığı yok. İçindeki assertion'lar: shape == (⌊(N−W)/H⌋+1, 2C) ve NaN yok. NaN/inf girdi reddediliyor.
- IEMG = Σ|x_n|, WL = Σ|x_{n+1} − x_n| (pencere başına W−1 fark). Rastgele sinyallerde naive döngü referansıyla birebir eşleşiyor (800/960/1000 Hz, tek-pencere uç durumu dahil).
- **Birim notu (flag):** Her iki feature da ham toplam (mV·örnek). Aynı 150 ms'de 1000 Hz'te 800 Hz'e göre %25 daha fazla örnek toplanıyor. Ayrıca WL'nin ölçeği örnekleme hızıyla değişiyor. Rate'ler arası karşılaştırılabilirlik Hafta 3'te ele alınmalı (ör. örnek sayısına bölme ya da ortak rate'e resample). Gate 1'de spec'teki tanım aynen uygulandı.
- Baseline tensor (`scripts/run_gate1.py`): `X = [IEMG_GMax, WL_GMax, IEMG_VM, WL_VM]` → (T, 4); `Y = [ankle_flx_angle_deg, ankle_flx_moment_Nm/kg]` → (T, 2), aynı %stride grid noktalarında. Kaydedilen dosyalar: `outputs/gate1/trial_<id>_features.npy`, `outputs/gate1/trial_<id>_targets.npy`, ayrıca `trial_<id>_meta.json` (grid indeksleri, rate, pencere parametreleri). `<id>` = `S<denek>_D<s.Data 0-tabanlı indeks>`.
- **Doğrulama:** Pipeline 844 walking trial'ın **tamamında** çalıştırıldı (X/Y shape eşleşmesi, NaN/inf yok). 5 trial kaydedildi; üç rate'i ve uç stride sürelerini kapsıyor:

| Trial | Rate | N | Stride | Pencere | X / Y | Grid |
|---|---|---|---|---|---|---|
| S02_D001 | 800 Hz | 709 | 0.885 s | 30 | (82, 4) / (82, 2) | 17–98 % |
| S03_D000 | 960 Hz | 945 | 0.983 s | 34 | (84, 4) / (84, 2) | 16–99 % |
| S01_D000 | 1000 Hz | 921 | 0.920 s | 31 | (81, 4) / (81, 2) | 17–97 % |
| S41_D001 (en kısa RX stride) | 960 Hz | 673 | 0.700 s | 23 | (78, 4) / (78, 2) | 22–99 % |
| S08_D018 (en uzun stride) | 1000 Hz | 2946 | 2.945 s | 112 | (94, 4) / (94, 2) | 6–99 % |

- Görsel kontrol (`outputs/gate1/week2_iemg_wl.png`, `outputs/gate1/gate1_triplets.png`): ayak bileği momenti stance sonunda (~%45–50) tepe yapıyor ve swing'de ≈ 0. VM IEMG çoğu trial'da loading response (%0–20) ve geç swing'de yüksek. Bunlar fizyolojik olarak beklenen desenler. İstisna: S03_D000'da VM ve GMax aynı anda %60–75'te tepe yapıyor ve ham sinyalde büyük düşük frekanslı sapmalar var. Muhtemelen toe-off civarı hareket artefaktı (F6 ile ilişkili).

**GATE 1 durumu: KARŞILANDI.** Herhangi bir level-walking trial için hizalı (features, position, moment) üçlüsü üretilebiliyor. 844/844 trial'da doğrulandı, 5 trial kaydedildi ve incelendi.

### Deliverable haritası
```
src/data/data_io.py       yükleme, isimden kanal indeksi, walking filtresi, trial-başı rate türetme/doğrulama
src/data/features.py      window_layout, window_end_indices, extract_features (saf NumPy)
src/data/align.py         sample -> %stride, 101-grid'e resample (ekstrapolasyonsuz), align_trial
scripts/run_gate1.py keşif -> survey (50 denek) -> 844 trial doğrulama -> kayıt -> plot
outputs/gate1/week1_raw_vs_stride.png, week2_iemg_wl.png, gate1_triplets.png
outputs/gate1/trial_<id>_{features,targets}.npy, trial_<id>_meta.json, gate1_survey.json, run_gate1_stdout.log
```

---

## Bölüm B — Explain in Your Own Words

> HARD AI Policy: Bu bölüm kullanıcı tarafından, kendi kelimeleriyle doldurulacak. Boş bırakıldı.

### Hafta 1: Kinematics neden 101 nokta/stride olarak saklanıyor ama EMG raw time series? Bu, orijinal yazarların gait cycle vs. sinyal ilişkisini nasıl düşündüğünü gösteriyor? Neden basitçe `np.concatenate` edilemiyor?



### Hafta 2: Neden özellikle IEMG + WL, raw amplitude veya RMS değil? Her biri sinyal hakkında ne yakalıyor (enerji vs karmaşıklık/değişim-hızı)? Biyomekanik ortodoksi MVC'ye normalize demesine rağmen neden non-normalized burada işe yarıyor — hangi gerçek-dünya MVC normalizasyon başarısızlığından kaçınılıyor?



