# outputs/week3 — klasör rehberi

Hafta 3 çıktıları. Teknik kararlar ve sonuç tabloları: depo kökündeki `WEEK3_LOG.md`.

## Yapı

```
data/
  dataset_walking844.npz    844 trial: X (844,101,4), Y_pos (844,101), Y_mom (844,101)
                            BÜYÜK, commit EDİLMİYOR (.gitignore) — script ile yeniden üretilir
  dataset_meta.json         trial başına meta: id, denek, foot, emg_side, rx_consistent, hız,
                            EMG hızı, pencere/hop, pencere sayısı, rate flag'leri, seam oranı
  dataset_report.json       C1–C7 kontrollerinin sonucu (PASS/FAIL + sayılar)
  fullstride_vs_gate1.png   Gate 1 (kırpılmış) vs yeni (tam stride) feature karşılaştırması
splits/
  speed_bins.json           5 hız bini (eşit trial sayısı) + trial_id → bin eşlemesi.
                            Hem bölmelerin hem de bin bazlı raporlamanın tek doğruluk kaynağı
  split_A.json              birincil koşu: 844 trial, 675 train / 169 val
  split_B.json              ablasyon: A'nın RX-tutarlı alt kümesi (572 trial)
  split_C.json              ablasyon: denek bazlı (40 train / 10 val deneği)
runs/
  smoke/                    16 trial'lık boru hattı testi (val seti = train seti)
  A_all844/ B_rx572/ C_subjectwise/
    position/  moment/      her biri AYRI bir LSTM (200 / 50 gizli birim):
      config.json           tüm hiperparametreler, seed, cihaz, sürümler, bölme dosyası + sha256,
                            veri seti sha256, git HEAD + working tree dirty, süreler
      scaler.json           z-score istatistikleri (yalnızca train'den)
      loss_log.csv          epoch başına train/val loss (normalize birim)
      metrics.json          trial başına r/RMSE, özet, hız bini ve RX/LX kırılımları
      curves.png            val setinde ölçülen (gri) vs tahmin (kırmızı), ortalama ± SD
      loss.png              train/val loss eğrisi (log y)
      model_final.pt        son epoch ağırlıkları — commit EDİLMİYOR (.gitignore)
MANIFEST.json               tamamlanan koşuların listesi (sonuç, git HEAD, cihaz, süre)
summary.md                  koşu × hedef tablosu, Keleş 2023 referans satırı, bin ve RX/LX kırılımı
```

## Yeniden üretme

Her şeyi tek komutla (veri seti → kontroller → bölmeler → smoke → A → B → C → özet):

```bash
.venv/bin/python scripts/train_baseline.py --run all
```

Tek tek:

```bash
.venv/bin/python scripts/build_week3_dataset.py      # data/dataset_walking844.npz
.venv/bin/python scripts/check_week3_data.py         # C1–C7 + dataset_report.json
.venv/bin/python scripts/make_splits.py              # splits/ (--force ile üzerine yazar)
.venv/bin/python scripts/train_baseline.py --run A   # tek koşu (--target position|moment)
.venv/bin/python scripts/train_baseline.py --summarize
```

Notlar:

- `data/Subject1.mat` … `Subject50.mat` mevcut olmalı (veri repoda değil).
- Tamamlanmış bir koşu klasörünün üzerine sessizce yazılmaz; yeniden çalıştırmak için `--force`.
  Koşu önce `<hedef>.partial/` olarak yazılır, ancak başarıyla bitince `<hedef>/` adını alır.
- Tüm rastgelelik seed 42; bölme dosyaları iki çalıştırmada bit düzeyinde aynı çıkar.
