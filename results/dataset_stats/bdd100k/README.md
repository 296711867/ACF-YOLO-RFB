# BDD100K 全量基础统计

> 生成日期：2026-07-25（Asia/Shanghai）  
> 来源：已审计 Lance 标注快照；排除 500 条 `synthetic_val_*`。

## 汇总

| split | images | instances | empty | instances/image | small | medium | large | small ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 70,000 | 1,288,009 | 0 | 18.400 | 710,395 | 414,421 | 163,193 | 55.15% |
| val | 10,000 | 185,526 | 0 | 18.553 | 102,729 | 59,669 | 23,128 | 55.37% |

## 分类别实例

| class | train | val |
|---|---:|---:|
| person | 91,405 | 13,262 |
| rider | 4,521 | 649 |
| car | 713,916 | 102,506 |
| truck | 30,003 | 4,245 |
| bus | 11,684 | 1,597 |
| train | 136 | 15 |
| motor | 3,002 | 452 |
| bike | 7,225 | 1,007 |
| traffic light | 186,224 | 26,885 |
| traffic sign | 239,893 | 34,908 |

## 天气

| value | train | val |
|---|---:|---:|
| clear | 37,412 | 5,346 |
| overcast | 8,784 | 1,239 |
| undefined | 8,134 | 1,157 |
| snowy | 5,571 | 769 |
| rainy | 5,083 | 738 |
| partly cloudy | 4,886 | 738 |
| foggy | 130 | 13 |

## 场景

| value | train | val |
|---|---:|---:|
| city street | 43,581 | 6,112 |
| highway | 17,414 | 2,499 |
| residential | 8,105 | 1,253 |
| parking lot | 378 | 49 |
| undefined | 366 | 53 |
| tunnel | 129 | 27 |
| gas stations | 27 | 7 |

## 时段

| value | train | val |
|---|---:|---:|
| daytime | 36,800 | 5,258 |
| night | 28,028 | 3,929 |
| dawn/dusk | 5,033 | 778 |
| undefined | 139 | 35 |

完整分辨率、遮挡、截断和类别图像数见同目录 `stats.json`。
