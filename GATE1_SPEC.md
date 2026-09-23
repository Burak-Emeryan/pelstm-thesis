# PELSTM — Gate 1 Execution Spec (Hafta 1-2)

> Bu dosya, Gate 1 çalışmasının başında verilen spec metninin birebir kaydıdır.
> Spec ile gerçek verinin uyuşmadığı noktalar `GATE1_LOG.md` Bölüm A.0'da (F1–F11) işaretlidir.

## Proje Bağlamı

Boğaziçi Üniversitesi yüksek lisans tezi — PELSTM (Physics-Embedded LSTM), sEMG-tabanlı akıllı ayak bileği protezi kontrolcüsü. Bu spec, 12 haftalık planın Faz I'ini (Hafta 1-2, "Gate 1") kapsıyor. Kapsam dışı: Hafta 3 ve sonrası (train/val split, LSTM modelleri, physics-loss). Bu spec'te SADECE Gate 1'i bitir, ileri gitme.

Veri seti: Lencioni ve ark. (2019) — 50 sağlıklı denekten, 8 kanallı sEMG + kinematik (eklem açıları) + kinetik (eklem momentleri) + GRF verisi. MATLAB `.mat` formatında, `data/raw/` altında.

## Ortam Kurulumu

* Python 3.11+, `numpy`, `scipy`, `matplotlib`, `torch` — versiyonları `requirements.txt`'e sabitle (pin), sonraki haftalarda (Hafta 3+) bu ortamın değişmemesi önemli.
* Bir sanal ortam (venv) kur.

## HAFTA 1 — `.mat` Yükleme, Struct'ı Anlama, Zaman-Tabanı Problemi

1. Dataset'i yükle: `scipy.io.loadmat('...mat', struct_as_record=False, squeeze_me=True)`. Struct `s`; trial'lar `s.Data` içinde (uzunluk `L`). `L`'yi yazdır, doğrula.
2. Değişken-adı alanlarını yazdır: `s.EMGVarName`, `s.AngVarName`, `s.MomVarName`, `s.GrfVarName`. GMax ve VM'nin EMG kanal sırasındaki tam indeksini kaydet — bunu ASLA varsayma, veriden ampirik olarak çıkar ve logla.
3. Bir Walking trial için shape'leri incele ve logla:
   * `.EMG` → 8×N (mV, native rate)
   * `.Ang` → 12×101
   * `.Mom` → 9×101
   * `.Grf` → 9×N (satır 1-3 kuvvet N, 4-6 CoP mm, 7-9 tork N·m)
   * `.Ang`/`.Mom` %stride bazında (101 nokta/döngü), `.EMG`/`.Grf` ham zaman serisi (N örnek). Bu farkı açıkça logla — bu farkın NEDEN önemli olduğu Hafta 1'in "Explain in your own words" adımının konusu (aşağıda, cevaplama, sadece yer aç).
4. Trial-başı örnekleme hızını oku. Hardcode etme — trial'lar arası değişiyor (EMG 800/960/1000 Hz). `.TimeStampEMG`, `.TimeStampKin`, `.TimeStampGrf` ve örnek sayılarından türet, her trial için ayrı hesapla.
5. Level walking'e filtrele: `.Task == 'Walking'`. Trial sayısını logla (beklenen ~844, ama gerçek sayıyı raporla, beklenen sayıyla eşleşmiyorsa flag'le).
6. Hizalama yönüne karar ver ve uygula: önerilen yön — önce feature dizisini (Hafta 2'de) çıkar, sonra feature dizisini 101-noktalık stride grid'ine resample et; `.Ang`/`.Mom` hedefleri dokunulmadan kalır. Bu kararı ve gerekçesini `GATE1_LOG.md`'ye yaz.
7. Hafta 1 done-when: bir trial için raw GMax EMG kendi zaman ekseninde, ankle angle %stride ekseninde plot edilebiliyor (`outputs/week1_raw_vs_stride.png`) ve ortak eksene nasıl getirileceği `GATE1_LOG.md`'de net ifade edilmiş.

## HAFTA 2 — Windowing + Tool-Agnostic Feature Modülü

1. Raw EMG üzerinde 150ms sliding window uygula. Window uzunluğunu trial-başı rate'ten örnek cinsinden hesapla (150ms × rate, her trial için ayrı). Stride/overlap'i belirle ve `GATE1_LOG.md`'ye belgele.
2. İki feature'ı implemente et:
   * `IEMG = Σ|x_n|` (window üzerinde mutlak değer toplamı)
   * `WL = Σ|x_{n+1} − x_n|` (kümülatif waveform length)
3. GMax ve VM üzerinde birkaç trial'da feature'ları çalıştır. Raw EMG vs windowed IEMG/WL'yi plot et (`outputs/week2_iemg_wl.png`).
4. Sarmala: `extract_features(emg_channels, rate, window_ms=150) -> feature_sequence`. Saf NumPy in/out — PyTorch'a veya herhangi bir framework'e bağımlı olmayacak. Assertion ekle: feature uzunluğu beklenen pencere sayısıyla eşleşiyor mu; NaN yok mu.
5. Torch olmadan tek bir trial için full baseline tensor'u kur: `X = [IEMG_GMax, WL_GMax, IEMG_VM, WL_VM]` per timestep → shape `(T, 4)`; hizalanmış eksende `y_pos`, `y_moment` hedefleri. `.npy` olarak kaydet (`outputs/trial_<id>_features.npy`, `outputs/trial_<id>_targets.npy`).
6. GATE 1 (done-when): herhangi bir level-walking trial için hizalanmış `(features, position, moment)` üçlüsü üretilebiliyor ve incelenebiliyor — bunu en az 3 farklı trial üzerinde çalıştırıp doğrula (tek trial'a güvenme).

## Deliverable Yapısı

```
src/
  data_io.py       # .mat yükleme, struct exploration, sampling-rate derivation
  features.py      # extract_features() — saf NumPy
  align.py          # resample/hizalama mantığı
scripts/
  run_gate1.py      # uçtan uca: trial seç → yükle → hizala → feature çıkar → kaydet → plot
outputs/
  week1_raw_vs_stride.png
  week2_iemg_wl.png
  trial_<id>_features.npy, trial_<id>_targets.npy   (birden fazla trial için)
requirements.txt
GATE1_LOG.md
```

## GATE1_LOG.md İçeriği — DİKKAT

Bu dosyayı oluştur ve iki ayrı bölüme ayır:

**Bölüm A — Teknik Kararlar (SEN doldur):** GMax/VM kanal indeksi, gözlemlenen örnekleme hızları, level-walking trial sayısı, hizalama yönü kararı ve gerekçesi, window/overlap parametreleri. Bunlar faktüel/teknik — sen yaz.

**Bölüm B — Explain in Your Own Words (BOŞ BIRAK, CEVAPLAMA):** Bu adımlar HARD AI Policy gereği kullanıcının kendi kelimeleriyle dolduracağı bir active recall adımı. Sadece soruları başlık olarak yaz, altını BOŞ bırak:

* Hafta 1: Kinematics neden 101 nokta/stride olarak saklanıyor ama EMG raw time series? Bu, orijinal yazarların gait cycle vs. sinyal ilişkisini nasıl düşündüğünü gösteriyor? Neden basitçe `np.concatenate` edilemiyor?
* Hafta 2: Neden özellikle IEMG + WL, raw amplitude veya RMS değil? Her biri sinyal hakkında ne yakalıyor (enerji vs karmaşıklık/değişim-hızı)? Biyomekanik ortodoksi MVC'ye normalize demesine rağmen neden non-normalized burada işe yarıyor — hangi gerçek-dünya MVC normalizasyon başarısızlığından kaçınılıyor?

## Genel Talimat

* Kod yazdıktan sonra gerçek veri üzerinde ÇALIŞTIR, hataları düzelt, sonucu doğrula — varsayımla ilerleme, her adımı gerçek çıktıyla teyit et.
* Zaman/token kısıtı yok — en az 3-5 farklı trial üzerinde test et, edge case'leri (farklı örnekleme hızları, kısa/uzun trial'lar) kontrol et, plot'ları görsel olarak makul mü diye kendi kendine değerlendir.
* Herhangi bir noktada checklist'teki bir varsayım (örn. ~844 trial, belirli kanal sırası) gerçek veriyle uyuşmuyorsa, sessizce devam etme — `GATE1_LOG.md`'ye flag'le.
