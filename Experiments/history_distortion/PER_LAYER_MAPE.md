# Per-layer average MAPE

Each row is one trainable parameter tensor, matching the normalization slices. MAPE pools all nonzero entries across clients 0, 10, and 20 in rounds 0–5; exact zeros are excluded and no epsilon is used. Within each group, rows are sorted by current MAPE, highest first.

Giving every one of the 62 tensors equal weight yields **0.26848% current** and **0.15544% round-shared** MAPE. The element-weighted global MAPEs were 0.11952% and 0.08874%.

| Group | Tensors | Equal-weight tensor-average MAPE: current (%) | Round-shared (%) |
| --- | ---: | ---: | ---: |
| At least one method ≥0.1% | 44 | 0.34496 | 0.18587 |
| Both methods below 0.1% | 18 | 0.08154 | 0.08104 |

The shared scheme increases MAPE in some tensors: `fc.weight` rises from 0.1127% to 0.2009%, and `layer1.1.bn1.bias` from 0.8611% to 0.9064%. This table exposes those regressions alongside improvements.

The earlier 44.1% figure was the worst single layer/client/round observation; the table below averages across all 18 client-rounds.

The common `model.` prefix is omitted below. Tiny is an explicit reporting cutoff (<0.1% in both methods), not a claim that a particular error is harmless for training.

## Higher MAPE

| Parameter tensor | Current MAPE (%) | Round-shared MAPE (%) |
| --- | ---: | ---: |
| `layer4.0.bn1.bias` | 2.6381 | 0.3438 |
| `layer4.1.bn2.weight` | 1.3682 | 0.3584 |
| `layer4.0.bn2.bias` | 1.2494 | 0.7559 |
| `layer4.0.downsample.1.bias` | 1.2494 | 0.7559 |
| `layer1.1.bn1.bias` | 0.8611 | 0.9064 |
| `layer4.0.downsample.1.weight` | 0.8172 | 0.2966 |
| `layer3.1.bn1.bias` | 0.6295 | 0.1523 |
| `layer4.0.bn2.weight` | 0.6020 | 0.1972 |
| `layer4.1.bn2.bias` | 0.3706 | 0.1314 |
| `layer2.0.bn1.bias` | 0.3133 | 0.3231 |
| `layer2.1.bn2.bias` | 0.2714 | 0.2530 |
| `layer1.1.bn2.bias` | 0.2255 | 0.1648 |
| `layer4.1.bn1.bias` | 0.2173 | 0.1099 |
| `layer3.1.bn2.bias` | 0.2169 | 0.1205 |
| `layer3.0.bn2.bias` | 0.2027 | 0.1074 |
| `layer3.0.downsample.1.bias` | 0.2027 | 0.1074 |
| `conv1.weight` | 0.2018 | 0.1349 |
| `layer2.0.bn2.bias` | 0.1847 | 0.2003 |
| `layer2.0.downsample.1.bias` | 0.1847 | 0.2003 |
| `layer2.1.bn1.bias` | 0.1797 | 0.0992 |
| `layer3.0.conv2.weight` | 0.1714 | 0.1004 |
| `layer4.0.downsample.0.weight` | 0.1665 | 0.1053 |
| `layer2.1.conv2.weight` | 0.1642 | 0.0955 |
| `layer3.0.bn1.bias` | 0.1480 | 0.1105 |
| `layer2.0.bn2.weight` | 0.1462 | 0.1501 |
| `layer3.1.conv2.weight` | 0.1387 | 0.0915 |
| `layer1.1.bn1.weight` | 0.1381 | 0.0887 |
| `layer3.0.downsample.1.weight` | 0.1365 | 0.0887 |
| `layer3.0.downsample.0.weight` | 0.1313 | 0.0964 |
| `layer2.0.downsample.0.weight` | 0.1290 | 0.0952 |
| `layer2.0.conv2.weight` | 0.1262 | 0.0914 |
| `layer1.1.conv2.weight` | 0.1234 | 0.0928 |
| `layer1.0.bn1.bias` | 0.1210 | 0.0966 |
| `layer4.1.conv2.weight` | 0.1191 | 0.0782 |
| `layer2.0.conv1.weight` | 0.1128 | 0.1004 |
| `fc.weight` | 0.1127 | 0.2009 |
| `layer1.1.conv1.weight` | 0.1122 | 0.0915 |
| `layer1.0.bn2.bias` | 0.1120 | 0.0983 |
| `layer3.0.conv1.weight` | 0.1109 | 0.0960 |
| `layer3.1.conv1.weight` | 0.1024 | 0.0877 |
| `layer1.0.conv1.weight` | 0.1024 | 0.0862 |
| `layer1.0.conv2.weight` | 0.0940 | 0.1033 |
| `bn1.bias` | 0.0923 | 0.1063 |
| `bn1.weight` | 0.0802 | 0.1077 |

## Tiny MAPE — listed last

| Parameter tensor | Current MAPE (%) | Round-shared MAPE (%) |
| --- | ---: | ---: |
| `layer1.0.bn2.weight` | 0.0970 | 0.0980 |
| `layer2.1.conv1.weight` | 0.0899 | 0.0906 |
| `layer1.1.bn2.weight` | 0.0892 | 0.0850 |
| `layer3.1.bn2.weight` | 0.0883 | 0.0799 |
| `layer4.0.conv1.weight` | 0.0874 | 0.0772 |
| `layer2.1.bn1.weight` | 0.0853 | 0.0903 |
| `layer3.0.bn2.weight` | 0.0845 | 0.0865 |
| `layer2.1.bn2.weight` | 0.0830 | 0.0845 |
| `layer4.0.bn1.weight` | 0.0809 | 0.0733 |
| `layer4.1.bn1.weight` | 0.0804 | 0.0740 |
| `layer3.1.bn1.weight` | 0.0792 | 0.0772 |
| `layer2.0.downsample.1.weight` | 0.0790 | 0.0765 |
| `layer3.0.bn1.weight` | 0.0779 | 0.0843 |
| `layer2.0.bn1.weight` | 0.0775 | 0.0759 |
| `layer1.0.bn1.weight` | 0.0758 | 0.0757 |
| `layer4.1.conv1.weight` | 0.0724 | 0.0713 |
| `layer4.0.conv2.weight` | 0.0719 | 0.0710 |
| `fc.bias` | 0.0683 | 0.0876 |

[Machine-readable values](results/svhn_seed44_round_shared_20261005/per_layer_mape.csv).
