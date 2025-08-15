# Korean Hate Speech Detection Evaluation Report

## Evaluation Overview 

- **Dataset**: K-MHaS (Korean Multi-label Hate Speech Dataset)
- **Evaluation Target**: Azure OpenAI Content Filter vs Azure Content Safety
- **Metrics**: Precision, Recall, F1-Score, Accuracy
- **Created**: 2025-08-14

### Performance Comparison: Content Filter vs Content Safety

| Method | Blocking/Severity | Precision | Recall | F1-Score | Accuracy | False Positive Rate | False Negative Rate |
|--------|-----------|-----------|--------|----------|----------|--------------------|--------------------|
| **Content Filter** | Low | 0.771 | 0.043 | 0.081 | 0.578 | 0.010 | 0.957 |
| **Content Filter** | Medium | 0.679 | 0.491 | 0.570 | 0.677 | 0.179 | 0.509 |
| **Content Filter** | High | 0.568 | 0.878 | 0.690 | 0.657 | 0.513 | 0.122 |
| **Content Safety** | 1~2 | 0.545 | 0.905 | 0.681 | 0.630 | 0.581 | 0.095 |
| **Content Safety** | 3~4 | 0.675 | 0.522 | 0.588 | 0.682 | 0.194 | 0.478 |
| **Content Safety** | 5~6 | 0.782 | 0.049 | 0.093 | 0.581 | 0.011 | 0.951 |

### Content Filter Evaluation Results

#### Overall Performance Metrics

| Blocking threshold | Precision | Recall | F1-Score | Accuracy | TP | TN | FP | FN | Total |
|-----------|-----------|--------|----------|----------|----|----|----|----|-------|
| **Low** | 0.771 | 0.043 | 0.081 | 0.578 | 37 | 1119 | 11 | 833 | 2000 |
| **Medium** | 0.679 | 0.491 | 0.570 | 0.677 | 427 | 928 | 202 | 443 | 2000 |
| **High** | 0.568 | 0.878 | 0.690 | 0.657 | 764 | 550 | 580 | 106 | 2000 |

#### Detailed Category Analysis

|         |                                   |low||medium||high||
|---------------|-----------------------------------------|-------------|------|--------|------|--------------|------|
|category_big   |category                                 |filtered<br>count        |filtered<br>mean  |filtered<br>count   |filtered<br>mean  |filtered<br>count         |filtered<br>mean  |
|Hate Speech    |['Age', 'Gender', 'Religion']            |0            |0.000 |2       |1.000 |2             |1.000 |
|Hate Speech    |['Age', 'Gender']            |1            |0.111 |7       |0.778 |9             |1.000 |
|Hate Speech    |['Age', 'Profanity']            |0            |0.000 |0       |0.000 |2             |1.000 |
|Hate Speech    |['Age', 'Religion', 'Race']            |0            |0.000 |1       |1.000 |1             |1.000 |
|Hate Speech    |['Age', 'Religion']            |1            |0.062 |9       |0.562 |15             |0.938 |
|Hate Speech    |['Age']            |3            |0.018 |54       |0.320 |138             |0.817 |
|Hate Speech    |['Gender', 'Religion']            |0            |0.000 |8       |0.364 |20             |0.909 |
|Hate Speech    |['Gender']            |3            |0.042 |34       |0.472 |56             |0.778 |
|Hate Speech    |['Origin', 'Age', 'Religion']            |0            |0.000 |3       |0.750 |3             |0.750 |
|Hate Speech    |['Origin', 'Age']            |1            |0.032 |23       |0.742 |30             |0.968 |
|Hate Speech    |['Origin', 'Gender']            |0            |0.000 |0       |0.000 |1             |1.000 |
|Hate Speech    |['Origin', 'Physical', 'Age']            |0            |0.000 |3       |0.600 |5             |1.000 |
|Hate Speech    |['Origin', 'Physical']            |0            |0.000 |2       |0.667 |3             |1.000 |
|Hate Speech    |['Origin', 'Religion']            |1            |0.056 |9       |0.500 |15             |0.833 |
|Hate Speech    |['Origin']            |0            |0.000 |50       |0.510 |84             |0.857 |
|Hate Speech    |['Physical', 'Age', 'Gender']            |0            |0.000 |1       |1.000 |1             |1.000 |
|Hate Speech    |['Physical', 'Age', 'Race']            |0            |0.000 |1       |1.000 |1             |1.000 |
|Hate Speech    |['Physical', 'Age']            |1            |0.056 |12       |0.667 |17             |0.944 |
|Hate Speech    |['Physical', 'Gender']            |1            |0.067 |8       |0.533 |14             |0.933 |
|Hate Speech    |['Physical', 'Religion']            |1            |0.250 |3       |0.750 |4             |1.000 |
|Hate Speech    |['Physical']            |6            |0.053 |35       |0.310 |95             |0.841 |
|Hate Speech    |['Politics', 'Age', 'Gender']            |0            |0.000 |2       |1.000 |2             |1.000 |
|Hate Speech    |['Politics', 'Age', 'Religion']            |1            |0.333 |1       |0.333 |3             |1.000 |
|Hate Speech    |['Politics', 'Age']            |1            |0.050 |13       |0.650 |18             |0.900 |
|Hate Speech    |['Politics', 'Gender']            |0            |0.000 |1       |1.000 |1             |1.000 |
|Hate Speech    |['Politics', 'Origin', 'Age', 'Religion']            |0            |0.000 |1       |1.000 |1             |1.000 |
|Hate Speech    |['Politics', 'Origin', 'Age']            |0            |0.000 |0       |0.000 |1             |0.500 |
|Hate Speech    |['Politics', 'Origin', 'Physical']            |1            |1.000 |1       |1.000 |1             |1.000 |
|Hate Speech    |['Politics', 'Origin', 'Religion']            |0            |0.000 |1       |0.500 |2             |1.000 |
|Hate Speech    |['Politics', 'Origin']            |0            |0.000 |1       |1.000 |1             |1.000 |
|Hate Speech    |['Politics', 'Physical', 'Age']            |0            |0.000 |4       |0.800 |4             |0.800 |
|Hate Speech    |['Politics', 'Physical']            |1            |0.062 |13       |0.812 |16             |1.000 |
|Hate Speech    |['Politics', 'Religion']            |0            |0.000 |10       |0.667 |15             |1.000 |
|Hate Speech    |['Politics']            |10            |0.098 |70       |0.686 |98             |0.961 |
|Hate Speech    |['Profanity']            |2            |0.067 |14       |0.467 |26             |0.867 |
|Hate Speech    |['Race']            |0            |0.000 |1       |1.000 |1             |1.000 |
|Hate Speech    |['Religion', 'Race']            |0            |0.000 |1       |1.000 |1             |1.000 |
|Hate Speech    |['Religion']            |2            |0.032 |28       |0.452 |57             |0.919 |
|Not Hate Speech    |['Not Hate Speech']            |11            |0.010 |202       |0.179 |580             |0.513 |
|**Filtering Total**|                                         |             |      |        |      |              |      |
|**Hate Speech**    |-                                        |**37**          |**0.043** |**427**      |**0.491** |**764**            |**0.878** |
|**Not Hate Speech**|-                                        |**11**          |**0.010** |**202**      |**0.179** |**580**             |**0.513** |

### Content Safety Evaluation Results

#### Overall Performance Metrics

| Severity level | Precision | Recall | F1-Score | Accuracy | TP | TN | FP | FN | Total |
|-----------|-----------|--------|----------|----------|----|----|----|----|-------|
| **Low (1~2)** | 0.545 | 0.905 | 0.681 | 0.630 | 787 | 474 | 656 | 83 | 2000 |
| **Medium (3~4)** | 0.675 | 0.522 | 0.588 | 0.682 | 454 | 911 | 219 | 416 | 2000 |
| **High (5~6)** | 0.782 | 0.049 | 0.093 | 0.581 | 43 | 1118 | 12 | 827 | 2000 |

#### Detailed Category Analysis

|         |                                   |low<br>(1~2)||medium<br>(3~4)||high<br>(5~6)||
|---------------|-----------------------------------------|-------------|------|--------|------|--------------|------|
|category_big   |category                                 |filtered<br>count        |filtered<br>mean  |filtered<br>count   |filtered<br>mean  |filtered<br>count         |filtered<br>mean  |
|Hate Speech    |['Age', 'Gender', 'Religion']            |2            |1.000 |1       |0.500 |0             |0.000 |
|Hate Speech    |['Age', 'Gender']            |9            |1.000 |8       |0.889 |0             |0.000 |
|Hate Speech    |['Age', 'Profanity']            |2            |1.000 |2       |1.000 |0             |0.000 |
|Hate Speech    |['Age', 'Religion', 'Race']            |1            |1.000 |1       |1.000 |0             |0.000 |
|Hate Speech    |['Age', 'Religion']            |16            |1.000 |7       |0.438 |0             |0.000 |
|Hate Speech    |['Age']            |150            |0.888 |77       |0.456 |4             |0.024 |
|Hate Speech    |['Gender', 'Religion']            |20            |0.909 |10       |0.455 |1             |0.045 |
|Hate Speech    |['Gender']            |60            |0.833 |28       |0.389 |3             |0.042 |
|Hate Speech    |['Origin', 'Age', 'Religion']            |4            |1.000 |2       |0.500 |0             |0.000 |
|Hate Speech    |['Origin', 'Age']            |30            |0.968 |24       |0.774 |1             |0.032 |
|Hate Speech    |['Origin', 'Gender']            |0            |0.000 |0       |0.000 |0             |0.000 |
|Hate Speech    |['Origin', 'Physical', 'Age']            |5            |1.000 |5       |1.000 |0             |0.000 |
|Hate Speech    |['Origin', 'Physical']            |3            |1.000 |3       |1.000 |0             |0.000 |
|Hate Speech    |['Origin', 'Religion']            |17            |0.944 |9       |0.500 |1             |0.056 |
|Hate Speech    |['Origin']            |81            |0.827 |44       |0.449 |1             |0.010 |
|Hate Speech    |['Physical', 'Age', 'Gender']            |1            |1.000 |1       |1.000 |0             |0.000 |
|Hate Speech    |['Physical', 'Age', 'Race']            |1            |1.000 |1       |1.000 |0             |0.000 |
|Hate Speech    |['Physical', 'Age']            |18            |1.000 |13       |0.722 |1             |0.056 |
|Hate Speech    |['Physical', 'Gender']            |14            |0.933 |7       |0.467 |0             |0.000 |
|Hate Speech    |['Physical', 'Religion']            |4            |1.000 |3       |0.750 |1             |0.250 |
|Hate Speech    |['Physical']            |98            |0.867 |55       |0.487 |8             |0.071 |
|Hate Speech    |['Politics', 'Age', 'Gender']            |2            |1.000 |1       |0.500 |0             |0.000 |
|Hate Speech    |['Politics', 'Age', 'Religion']            |3            |1.000 |1       |0.333 |1             |0.333 |
|Hate Speech    |['Politics', 'Age']            |19            |0.950 |14       |0.700 |1             |0.050 |
|Hate Speech    |['Politics', 'Gender']            |1            |1.000 |1       |1.000 |0             |0.000 |
|Hate Speech    |['Politics', 'Origin', 'Age', 'Religion']            |1            |1.000 |1       |1.000 |0             |0.000 |
|Hate Speech    |['Politics', 'Origin', 'Age']            |1            |0.500 |1       |0.500 |0             |0.000 |
|Hate Speech    |['Politics', 'Origin', 'Physical']            |1            |1.000 |1       |1.000 |1             |1.000 |
|Hate Speech    |['Politics', 'Origin', 'Religion']            |2            |1.000 |0       |0.000 |0             |0.000 |
|Hate Speech    |['Politics', 'Origin']            |1            |1.000 |1       |1.000 |0             |0.000 |
|Hate Speech    |['Politics', 'Physical', 'Age']            |5            |1.000 |4       |0.800 |0             |0.000 |
|Hate Speech    |['Politics', 'Physical']            |15            |0.938 |10       |0.625 |0             |0.000 |
|Hate Speech    |['Politics', 'Religion']            |15            |1.000 |9       |0.600 |0             |0.000 |
|Hate Speech    |['Politics']            |97            |0.951 |70       |0.686 |13             |0.127 |
|Hate Speech    |['Profanity']            |28            |0.933 |16       |0.533 |4             |0.133 |
|Hate Speech    |['Race']            |1            |1.000 |0       |0.000 |0             |0.000 |
|Hate Speech    |['Religion', 'Race']            |1            |1.000 |1       |1.000 |1             |1.000 |
|Hate Speech    |['Religion']            |58            |0.935 |22       |0.355 |1             |0.016 |
|Not Hate Speech    |['Not Hate Speech']            |656            |0.581 |219       |0.194 |12             |0.011 |
|**Filtering Total**|                                         |             |      |        |      |              |      |
|**Hate Speech**    |-                                        |**787**          |**0.905** |**454**      |**0.522** |**43**            |**0.049** |
|**Not Hate Speech**|-                                        |**656**          |**0.581** |**219**      |**0.194** |**12**             |**0.011** |