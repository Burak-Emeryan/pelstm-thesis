# HAFTA 3 LOG — PELSTM (speed-stratified split + iki ayrı LSTM + uçtan uca eğitim)

Tekrar üretmek için: `.venv/bin/python scripts/train_baseline.py --run all`
(çıktılar: `outputs/week3/`, klasör açıklaması: `outputs/week3/README.md`).
Ortam Gate 1 ile aynı ve değiştirilmedi: Python 3.13.12, numpy 2.5.3, scipy 1.18.1,
matplotlib 3.11.2, torch 2.14.0 (`requirements.txt` pinned, yeni bağımlılık yok).
Cihaz: CPU (bu makinede CUDA yok). Tüm rastgelelik seed 42.

Hafta 3 teslimleri: (D1) hız aralığına göre katmanlı train/val bölmesi,
(D2) pozisyon ve moment için iki AYRI LSTM, (D3) uçtan uca çalışan eğitim script'i.
Faz 1 bunların dayandığı veri setini kurar (her stride'ın 101 noktasının tamamı).

---

## Bölüm A — Teknik Kararlar

### A.0 Spec ile veri/kod arasındaki uyuşmazlıklar ve sapmalar (FLAG)

| # | Spec | Gerçek durum | Karar |
|---|---|---|---|
| W1 | Trial id biçimi `f"S{subject}_D{data_idx}"` ve "Gate 1'deki gibi" | İkisi çelişiyor: literal biçim `S2_D1` üretir, Gate 1 ise sıfır dolgulu `S02_D001` kullanıyor. Spec'in kendi referansları da sıfır dolgulu (`S12_D016`, `S35_D000`, `S02_D001`) | Gate 1 biçimi (`S{sub:02d}_D{idx:03d}`) kullanıldı; aksi halde C5 Gate 1 dosyalarıyla eşleşmezdi |
| W2 | Smoke testi: 300 epoch, final train loss < 0.1 × ilk loss | Seed 42'de position modeli epoch 98'de gradyan patlaması yaşıyor (gradyan normu medyan 0.96 → 130.9), loss 0.27'den 1.00'a dönüyor; 300. epoch'ta ancak 0.1005 oranına iniyor (eşiğin hemen üstü) | Kullanıcı onayıyla smoke epoch 300 → **600**. Eşik (0.1), seed (42), clipping yasağı ve asıl koşuların 200 epoch'u değişmedi. 600'de: position 0.0300, moment 0.0097. Teşhis: boru hattı sağlam — 4 farklı seed'de oran 0.028–0.046, model 16 trial'ı r=0.99 ile ezberliyor |
| W3 | Smoke testi validation seti tanımlı değil | — | Smoke'ta validation seti eğitim setiyle **aynı** 16 trial (amaç öğrenme kapasitesini ölçmek, genelleme değil). `config.json` içinde `note` alanıyla işaretli |
| W4 | C3: naive referans "pre-interpolation feature matrix'in ilgili satırıyla" karşılaştırılacak | İstenen `k` değerlerinin bir kısmı (1, W−2, W−1) hop ızgarasında değil, yani karşılık gelen satır yok | Izgaradaki `k`'ler matris satırıyla, ızgara dışındakiler aynı belgelenmiş wrap kuralıyla kurulan pencerenin feature'ıyla karşılaştırıldı. Naive taraf her durumda bağımsız (modüler indeksleme + düz NumPy) |
| W5 | — | `.idea/` (PyCharm) klasörü repo kökünde duruyordu ve `git status`'u kirletiyordu | `.gitignore`'a eklendi (R1 ekleme yapmaya izin veriyor) |
| W6 | — | Subject18'in `EMGSide` alanı `'NO'`; 23 walking trial'ı var, 9'u `Foot='RX'` | Gate 1'deki gibi: bu 9 trial `rx_consistent` sayılmıyor, toplam 844'e dahil. Run B'nin dışında kalıyorlar |
| W7 | — | Run C'de val hız aralığı (0.586–2.166 m/s) train aralığından (0.292–2.294) dar | Denek bazlı bölmede bu kaçınılmaz: en yavaş/en hızlı trial'lar train deneklerinde. Hız grupları denek medyanıyla dengelenmiş durumda, müdahale edilmedi |
| W8 | — | Run B'nin bin başına val dağılımı eşit değil (29/23/27/24/21) | Beklenen: B, A'dan yalnızca RX-tutarsız trial'lar çıkarılarak türetiliyor (spec'in tanımı), yeniden kotalama yapılmıyor |

Gate 1'de işaretlenen ve hâlâ geçerli olan konular (DC offset, filtre yok, örnekleme hızına bağlı
feature ölçeği, ayak bileği açısının sistematik ofseti) bu hafta da **kasıtlı olarak** ele alınmadı;
spec R7 bunları açıkça yasaklıyor.

### A.1 Faz 1 — tam stride (101 nokta) feature veri seti

Gate 1 yalnızca stride'ın içine tam sığan pencereleri tutuyordu; bu yüzden her stride'ın ilk
%6–22'si düşüyor ve trial başına 77–94 nokta kalıyordu. Keleş 2023 sonuçlarını %0–100 aralığında
raporladığı için karşılaştırılabilir bir replikasyon her trial'da 101 noktanın tamamını gerektiriyor.

**Karar: periyodik (wrap-around) ön-padding.** `.EMG` tam bir gait cycle'a kırpılmış ve N−1. örnek,
0. örnekle aynı olayın bir döngü sonrası; dolayısıyla 0. örnekten önceki örnekler aynı stride'ın
sonundan alınıyor: `pre = x[:, N−W : N−1]` (uzunluk tam W−1), `x_ext = [pre, x]`.
Orijinal `k` indeksi `x_ext` içinde `k+W−1`'e denk geliyor; `k`'de **biten** pencere `x_ext[:, k : k+W]`.

- **Sıfır padding reddedildi:** IEMG'yi stride'ın %0–20 loading-response fazında yapay olarak
  düşürürdü — tam da VM'nin aktif olduğu yerde — ve her stride'ın başını sistematik olarak
  "sessizliğe" doğru yanlılardı.
- **Varsayım:** ardışık stride'lar benzer (steady-state yürüyüş). Ek tanı olarak her trial/kanal için
  seam oranı `|x[N−1] − x[0]| / median(|diff(x)|)` hesaplanıyor: medyan 2.76, p90 13.02, maks 81.23
  (kanal medyanları GMax 2.54, VM 3.00). Yani tipik ek yerinde sıçrama, sinyalin tipik örnekler-arası
  adımının birkaç katı; uç trial'larda belirgin.
- Pencere sonu damgalama (causal) Gate 1 ile aynı; `(N−1) % H ≠ 0` olan 603/844 trial'da tam olarak
  `k = N−1`'de biten bir ek pencere ekleniyor ki ızgaranın %100 ucu ekstrapolasyonsuz dolsun.
- Hedefler (`.Ang` / `.Mom`) **hiç** resample edilmiyor; kaydedilen satırların `.mat` içindeki
  satırlarla bit düzeyinde aynı olduğu her trial için yeniden yüklenip doğrulanıyor.

Yeni IEMG/WL matematiği yazılmadı: `features.extract_features` ve `align.resample_to_stride_grid`
aynen kullanıldı (Gate 1 dosyalarının hiçbiri değiştirilmedi).

**Veri seti doğrulaması (C1–C7, hepsi PASS).** 844 trial; Foot RX 581 / LX 263; rx_consistent 572;
EMG hızı 960→324, 800→273, 1000→247; yalnızca S12_D016 ve S35_D000 rate flag'li; hız 0.292–2.294 m/s
(medyan 1.204, izin verilen 0.25–2.4 aralığında); tüm X (101,4) ve Y (101,) sonlu; N 569–2946.
Bağımsız yazılmış naive referans (modüler indeksleme + düz NumPy) üç hızı kapsayan 6 trial × 6 pencere
sonunda implementasyonla 1e-12 toleransında eşleşiyor; ızgara uçları (X[0], X[100]) de naive
k=0 / k=N−1 pencereleriyle eşleşiyor. Gate 1'in 5 kayıtlı trial'ıyla ortak ızgara noktalarında
kolon bazlı Pearson r = 0.996–1.000 (eşik 0.98).

### A.2 Hız aralığına göre katmanlı bölme (D1)

**Global hız binleri (tek doğruluk kaynağı, `outputs/week3/splits/speed_bins.json`).**
844 trial `np.lexsort((arange, speed))` ile sıralanıp `np.array_split(..., 5)` ile eşit sayıda
trial içeren 5 bine ayrılıyor (169/169/169/169/168, bin 0 en yavaş):

| bin | n | hız aralığı (m/s) |
|---|---|---|
| 0 | 169 | 0.292 – 0.914 |
| 1 | 169 | 0.915 – 1.114 |
| 2 | 169 | 1.115 – 1.292 |
| 3 | 169 | 1.292 – 1.511 |
| 4 | 168 | 1.514 – 2.294 |

**Run A (birincil, replikasyon kapısı), 844 trial:** 675 train / 169 val. Bin başına val kotası
largest-remainder yöntemiyle → (34, 34, 34, 34, 33). `rng = default_rng(42)`, binler artan sırada
permütasyon. Sonuç: her binde 135 train / 34 (bin 4'te 33) val. Train hız aralığı 0.292–2.149,
val 0.390–2.294 m/s.

**Run B (ablasyon), RX-tutarlı 572 trial:** A'nın train ve val listelerinin `rx_consistent` ile
kesişimi — yani A'dan yalnızca 272 trial (263 LX + Subject18'in 9 RX trial'ı) çıkarılıyor, yeniden
bölme yapılmıyor. 448 train / 124 val.

**Run C (ablasyon), denek bazlı, 844 trial:** denekler medyan hızlarına (eşitlikte denek numarasına)
göre sıralanıp 10'arlı 5 gruba bölünüyor; her gruptan `default_rng(42)` ile 2 val deneği çekiliyor.
40 train / 10 val deneği (val: 6, 9, 10, 11, 12, 20, 23, 33, 36, 43), 651 train / 193 val trial.

Bölme dosyaları bir kez üretilip saklanıyor, içlerinde zaman damgası yok: iki kez çalıştırınca
4 dosyanın sha256'sı birebir aynı (determinizm doğrulandı). `--force` olmadan üzerine yazmıyor.
Her koşu bölmesini dosyadan okuyor ve dosyanın sha256'sını `config.json`'a yazıyor.

### A.3 Ölçekleme — yalnızca train istatistiği

X için 4 feature kolonunun her birinin ortalama/std'si **sadece train trial'ları × 101 adım**
üzerinden; hedef için train setinde skaler ortalama/std. Std tabanı 1e-8. Aynı dönüşüm val'e
uygulanıyor, tahminler metrik hesabından önce ters dönüştürülüyor. İstatistikler `scaler.json`'a
yazılıyor. Val verisi istatistiklere hiçbir şekilde girmiyor.

### A.4 İki AYRI LSTM (D2)

`AnkleLSTM`: `nn.LSTM(input_size=4, hidden_size, num_layers=1, batch_first=True)` (tek yönlü,
sıfır başlangıç durumu, dropout yok) + her zaman adımına uygulanan `nn.Linear(hidden, 1)`;
giriş (B, 101, 4) → çıkış (B, 101). **Pozisyon modeli 200 gizli birim, moment modeli 50**
(Keleş 2023). İki ağ birbirinden tamamen bağımsız: kendi optimizer'ı, kendi scaler'ı, kendi koşu
klasörü; ortak ağ ya da ortak loss yok.

Hiperparametreler (Keleş 2023 replikasyonu): Adam lr 1e-3, MSELoss (tüm adımların ortalaması),
batch 32, 200 epoch. Scheduler, early stopping, dropout, weight decay, gradient clipping ve
bidirectional **yok**. Değerlendirilen ve kaydedilen model **son epoch** modeli; en iyi val epoch'u
yalnızca bilgi amaçlı `metrics.json`'a yazılıyor.

### A.5 Üç koşunun rolü ve sonuçlar

Run A önceden ilan edilmiş birincil koşudur (replikasyon kapısı). Run B ve Run C önceden ilan
edilmiş ablasyonlardır. Hiçbir koşu validation performansına bakılarak seçilmedi.

| koşu | hedef | n_val | r (ort ± SD) | RMSE (ort ± SD) | pooled r |
|---|---|---|---|---|---|
| A (844) | position | 169 | 0.9159 ± 0.0909 | 4.4996 ± 2.0587 deg | 0.8633 |
| A (844) | moment | 169 | 0.9746 ± 0.0346 | 0.1154 ± 0.0499 Nm/kg | 0.9615 |
| B (572, RX) | position | 124 | 0.9180 ± 0.0665 | 4.1122 ± 1.4204 deg | 0.8843 |
| B (572, RX) | moment | 124 | 0.9655 ± 0.1101 | 0.1178 ± 0.0684 Nm/kg | 0.9534 |
| C (denek bazlı) | position | 193 | 0.8201 ± 0.2212 | 5.8619 ± 2.3674 deg | 0.7608 |
| C (denek bazlı) | moment | 193 | 0.9559 ± 0.1187 | 0.1327 ± 0.0869 Nm/kg | 0.9384 |
| Keleş 2023 (referans) | position | 169 | 0.9010 ± 0.0805 | 4.8927 ± 1.6140 deg | — |
| Keleş 2023 (referans) | moment | 169 | 0.9718 ± 0.0326 | 0.1193 ± 0.0501 Nm/kg | — |

Run A, hız bini başına (ayrıntı `outputs/week3/summary.md`): position r 0.891–0.934,
RMSE 4.30–4.95 deg; moment r 0.964–0.984, RMSE 0.099–0.131 Nm/kg.
Run A val setinde RX-tutarlı (n=124) vs LX (n=44): position r 0.9215 vs 0.8983,
moment r 0.9752 vs 0.9728. Run C'de RX-tutarlı (n=129) vs LX (n=64): position r 0.8108 vs 0.8389,
moment r 0.9447 vs 0.9784.

### A.6 Bilinen sınırlamalar

1. Wrap padding periyodik yürüyüş varsayıyor; Keleş'in stride başlangıcını nasıl ele aldığı
   makalede belirtilmemiş.
2. Hop = 25 ms bizim seçimimiz; makale bir hop değeri vermiyor.
3. lr, batch ve loss fonksiyonu bizim seçimimiz; makalede belirtilmemiş.
4. Keleş düz rastgele bölme kullanmış; buradaki bölme Hafta 3 planının gerektirdiği gibi hız
   aralığına göre katmanlı (Keleş her iki setin de aynı hız aralığını kapsadığını belirtiyor).
5. Keleş gizli birim sayısını validation setiyle seçmiş ve yine aynı setle raporlamış.
6. 101 noktalık ızgaraya lineer interpolasyon 25 ms'ye kadar ileriye bakabiliyor; kapalı-çevrim
   çalışmadan önce zero-order hold ile değiştirilmeli.

### A.7 Dosya haritası

```
src/data/fullstride.py      wrap padding + tam stride (101 nokta) feature'ları
src/data/dataset.py         844 trial'lık veri setini kurar/yükler, hedefleri bit düzeyinde doğrular
src/data/splits.py          hız binleri + A/B/C bölmeleri (seed 42, deterministik)
src/models/lstm_baseline.py AnkleLSTM, build_position_model (200), build_moment_model (50)
src/models/train.py         ölçekleme, eğitim döngüsü, config/scaler/loss_log/figürler, MANIFEST
src/models/metrics.py       trial başına Pearson r ve RMSE, bin/taraf kırılımları
scripts/build_week3_dataset.py   veri setini kurar
scripts/check_week3_data.py      C1–C7 kontrolleri + dataset_report.json + karşılaştırma figürü
scripts/make_splits.py           bölmeleri üretir
scripts/train_baseline.py        uçtan uca: veri → bölme → smoke → A → B → C → özet (D3)
```

---

## Bölüm B — Explain in Your Own Words

HARD AI Policy: Bu bölüm kullanıcı tarafından doldurulacak. Boş bırakıldı.

Plan'daki Week 3 istemleri Burak tarafından eklenecek.

### 1. Neden feature'ları tam stride (101 nokta) üzerinde üretmeyi seçtik; wrap padding sıfır padding'den neden daha az yanlı ve varsayımı ne?



### 2. Neden z-score ölçekleme yalnızca train istatistiğiyle yapılıyor; val istatistiği kullanılsaydı hangi sızıntı olurdu?



### 3. Neden hız aralığına göre katmanlı bölme; düz rastgele bölmeden farkı ne?



### 4. Run A (844) ile Run B (572) arasındaki fark neyi ölçüyor; LX trial'ları öğrenmeyi neden bozar?



### 5. Run C (denek bazlı) neden trial bazlı A'dan farklı bir soruya cevap veriyor?


