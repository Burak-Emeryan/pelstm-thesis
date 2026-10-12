# Hafta 3 — baseline LSTM sonuçları

Run A is the pre-declared primary run (replication gate). Runs B and C are pre-declared ablations. No run was selected by validation performance.

## Run × target

| run | target | n_val | r (mean ± SD) | RMSE (mean ± SD) | pooled r |
|---|---|---|---|---|---|
| A | position | 169 | 0.9159 ± 0.0909 | 4.4996 ± 2.0587 deg | 0.8633 |
| A | moment | 169 | 0.9746 ± 0.0346 | 0.1154 ± 0.0499 Nm/kg | 0.9615 |
| B | position | 124 | 0.9180 ± 0.0665 | 4.1122 ± 1.4204 deg | 0.8843 |
| B | moment | 124 | 0.9655 ± 0.1101 | 0.1178 ± 0.0684 Nm/kg | 0.9534 |
| C | position | 193 | 0.8201 ± 0.2212 | 5.8619 ± 2.3674 deg | 0.7608 |
| C | moment | 193 | 0.9559 ± 0.1187 | 0.1327 ± 0.0869 Nm/kg | 0.9384 |
| Keleş 2023 (reference) | position | 169 | 0.9010 ± 0.0805 | 4.8927 ± 1.6140 deg | — |
| Keleş 2023 (reference) | moment | 169 | 0.9718 ± 0.0326 | 0.1193 ± 0.0501 Nm/kg | — |

Keleş 2023 used a plain random 675/169 split; the splits here are speed-stratified.

## Run A — per speed bin

| bin | target | n | r (mean) | RMSE (mean) |
|---|---|---|---|---|
| 0 (0.29–0.91 m/s) | position | 34 | 0.9014 | 4.4437 deg |
| 1 (0.91–1.11 m/s) | position | 34 | 0.9337 | 4.4825 deg |
| 2 (1.12–1.29 m/s) | position | 34 | 0.9284 | 4.2977 deg |
| 3 (1.29–1.51 m/s) | position | 34 | 0.9240 | 4.3385 deg |
| 4 (1.51–2.29 m/s) | position | 33 | 0.8912 | 4.9491 deg |
| 0 (0.29–0.91 m/s) | moment | 34 | 0.9660 | 0.1280 Nm/kg |
| 1 (0.91–1.11 m/s) | moment | 34 | 0.9836 | 0.0986 Nm/kg |
| 2 (1.12–1.29 m/s) | moment | 34 | 0.9817 | 0.1011 Nm/kg |
| 3 (1.29–1.51 m/s) | moment | 34 | 0.9774 | 0.1194 Nm/kg |
| 4 (1.51–2.29 m/s) | moment | 33 | 0.9642 | 0.1305 Nm/kg |

## Runs A and C — RX-consistent vs LX validation trials

| run | target | subset | n | r (mean) | RMSE (mean) |
|---|---|---|---|---|---|
| A | position | RX-consistent | 124 | 0.9215 | 4.4224 deg |
| A | position | Foot = LX | 44 | 0.8983 | 4.7937 deg |
| A | moment | RX-consistent | 124 | 0.9752 | 0.1155 Nm/kg |
| A | moment | Foot = LX | 44 | 0.9728 | 0.1155 Nm/kg |
| C | position | RX-consistent | 129 | 0.8108 | 5.9538 deg |
| C | position | Foot = LX | 64 | 0.8389 | 5.6767 deg |
| C | moment | RX-consistent | 129 | 0.9447 | 0.1348 Nm/kg |
| C | moment | Foot = LX | 64 | 0.9784 | 0.1285 Nm/kg |
