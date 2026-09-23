# pelstm-thesis

PELSTM (Physics-Embedded LSTM) — sEMG tabanlı akıllı ayak bileği protezi kontrolcüsü.
Boğaziçi Üniversitesi yüksek lisans tezi. Bu repo şu an 12 haftalık planın **Faz I'ini (Hafta 1-2, "Gate 1")**
içeriyor: ham sEMG'den özellik çıkarımı ve bu özelliklerin eklem açısı/momenti hedefleriyle hizalanması.

## Veri

[Lencioni et al. (2019), *Scientific Data*](https://doi.org/10.1038/s41597-019-0323-z) —
50 sağlıklı denekten 8 kanallı sEMG + kinematik + kinetik + GRF verisi.
Veri figshare'den indirilmeli: <https://doi.org/10.6084/m9.figshare.c.4494755>

**Veri repoya dahil edilmiyor** (`.gitignore`'da). Çalıştırmadan önce `Subject1.mat` … `Subject50.mat`
dosyalarını `data/` klasörüne koy:

```
data/Subject1.mat
data/Subject2.mat
...
data/Subject50.mat
```

Kod dosya başına tek bir `s` struct'ı bekliyor (deneğin tüm trial'ları `s.Data` içinde).

## Kurulum ve çalıştırma

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/run_gate1.py
```

Script uçtan uca çalışır: 50 deneği tarar → 844 level-walking trial'ın tamamında pipeline'ı doğrular →
5 temsili trial için hizalı tensörleri kaydeder → plot üretir. Çıktılar `outputs/gate1/` altına yazılır
(bu klasör commit ediliyor, böylece sonuçlar veri olmadan da incelenebilir).

Belirli trial'ları çalıştırmak için (`denek:s.Data indeksi`, 0-tabanlı):

```bash
.venv/bin/python scripts/run_gate1.py --trials 6:0 41:1
```

Ortam pinlenmiş (`requirements.txt`): Python 3.13.12, numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2, torch 2.14.0.
Hafta 3+ boyunca bu ortamın sabit kalması bekleniyor.

## Yapı

```
data/                 .mat dosyaları (indirilecek, commit edilmiyor)
src/data/             data_io.py (yükleme, kanal indeksi, rate türetme)
                      features.py (extract_features — saf NumPy, framework bağımsız)
                      align.py (native zaman ekseni → 101 noktalık %stride grid'i)
src/models/           Hafta 3+ LSTM kodu (henüz boş)
scripts/run_gate1.py  uçtan uca Gate 1 pipeline'ı
outputs/gate1/        plot'lar, kaydedilen tensörler (.npy), survey ve çalıştırma log'u
GATE1_SPEC.md         Gate 1 için verilen spec
GATE1_LOG.md          teknik kararlar (Bölüm A) + kullanıcının dolduracağı sorular (Bölüm B)
```

## Gate 1 durumu

**Karşılandı.** Herhangi bir level-walking trial için hizalı `(features, position, moment)` üçlüsü
üretilebiliyor: özellikler (IEMG ve WL, GMax ve VM kanallarında, 150 ms pencere / 25 ms adım, nedensel
damgalama) EMG'nin kendi örnekleme hızında hesaplanıp 101 noktalık %stride ızgarasına taşınıyor; ayak
bileği açısı ve momenti hedeflerine hiç dokunulmuyor. Pipeline 844 level-walking trial'ın tamamında
hatasız çalıştı ve 5 trial (üç örnekleme hızı ile en kısa/en uzun stride dahil) kaydedilip görsel olarak
incelendi. Spec ile gerçek verinin uyuşmadığı noktalar ve Hafta 3 öncesi karar bekleyen konular —
EMG'nin tek stride'a kırpılmış olması, taraf eşleşmesi, DC offset, örnekleme hızına bağlı özellik
ölçeği — [GATE1_LOG.md](GATE1_LOG.md) Bölüm A'da işaretli. Hafta 3+ (train/val split, LSTM,
physics-loss) bu repoda henüz yok.
