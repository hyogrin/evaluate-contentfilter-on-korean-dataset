# Korean Hate Speech Detection Evaluation Report

## Evaluation Overview 

- **Dataset**: K-MHaS (Korean Multi-label Hate Speech Dataset)
- **Evaluation Target**: Azure OpenAI Content Filter vs Azure Content Safety
- **Metrics**: Precision, Recall, F1-Score, Accuracy
- **Created**: 2025-08-14

### Performance Comparison: Content Filter vs Content Safety

| Method | Blocking/Severity | Precision | Recall | F1-Score | Accuracy | False Positive Rate | False Negative Rate |
|--------|-----------|-----------|--------|----------|----------|--------------------|--------------------|
| **Content Filter** | Low | 0.500 | 0.029 | 0.056 | 0.660 | 0.015 | 0.971 |
| **Content Filter** | Medium | 0.548 | 0.500 | 0.523 | 0.690 | 0.212 | 0.500 |
| **Content Filter** | High | 0.429 | 0.882 | 0.577 | 0.560 | 0.606 | 0.118 |
| **Content Safety** | 1~2 | 0.408 | 0.912 | 0.564 | 0.520 | 0.682 | 0.088 |
| **Content Safety** | 3~4 | 0.581 | 0.529 | 0.554 | 0.710 | 0.197 | 0.471 |
| **Content Safety** | 5~6 | 0.500 | 0.029 | 0.056 | 0.660 | 0.015 | 0.971 |

### Content Filter Evaluation Results

#### Overall Performance Metrics

| Blocking threshold | Precision | Recall | F1-Score | Accuracy | TP | TN | FP | FN | Total |
|-----------|-----------|--------|----------|----------|----|----|----|----|-------|
| **Low** | 0.500 | 0.029 | 0.056 | 0.660 | 1 | 65 | 1 | 33 | 100 |
| **Medium** | 0.548 | 0.500 | 0.523 | 0.690 | 17 | 52 | 14 | 17 | 100 |
| **High** | 0.429 | 0.882 | 0.577 | 0.560 | 30 | 26 | 40 | 4 | 100 |

#### Detailed Category Analysis

|         |                                   |low||medium||high||
|---------------|-----------------------------------------|-------------|------|--------|------|--------------|------|
|category_big   |category                                 |filtered<br>count        |filtered<br>mean  |filtered<br>count   |filtered<br>mean  |filtered<br>count         |filtered<br>mean  |
|Hate Speech    |['Age', 'Religion']            |0            |0.000 |1       |1.000 |1             |1.000 |
|Hate Speech    |['Age']            |1            |0.125 |4       |0.500 |6             |0.750 |
|Hate Speech    |['Gender']            |0            |0.000 |0       |0.000 |2             |0.500 |
|Hate Speech    |['Origin', 'Age']            |0            |0.000 |3       |1.000 |3             |1.000 |
|Hate Speech    |['Origin', 'Religion']            |0            |0.000 |1       |0.500 |2             |1.000 |
|Hate Speech    |['Origin']            |0            |0.000 |1       |0.200 |5             |1.000 |
|Hate Speech    |['Physical']            |0            |0.000 |1       |1.000 |1             |1.000 |
|Hate Speech    |['Politics', 'Age']            |0            |0.000 |1       |1.000 |1             |1.000 |
|Hate Speech    |['Politics', 'Physical']            |0            |0.000 |0       |0.000 |1             |1.000 |
|Hate Speech    |['Politics', 'Religion']            |0            |0.000 |1       |1.000 |1             |1.000 |
|Hate Speech    |['Politics']            |0            |0.000 |2       |1.000 |2             |1.000 |
|Hate Speech    |['Profanity']            |0            |0.000 |1       |0.500 |2             |1.000 |
|Hate Speech    |['Race']            |0            |0.000 |1       |1.000 |1             |1.000 |
|Hate Speech    |['Religion']            |0            |0.000 |0       |0.000 |2             |1.000 |
|Not Hate Speech    |['Not Hate Speech']            |1            |0.015 |14       |0.212 |40             |0.606 |
|**Filtering Total**|                                         |             |      |        |      |              |      |
|**Hate Speech**    |-                                        |**1**          |**0.029** |**17**      |**0.500** |**30**            |**0.882** |
|**Not Hate Speech**|-                                        |**1**          |**0.015** |**14**      |**0.212** |**40**             |**0.606** |

### Content Safety Evaluation Results

#### Overall Performance Metrics

| Severity level | Precision | Recall | F1-Score | Accuracy | TP | TN | FP | FN | Total |
|-----------|-----------|--------|----------|----------|----|----|----|----|-------|
| **Low (1~2)** | 0.408 | 0.912 | 0.564 | 0.520 | 31 | 21 | 45 | 3 | 100 |
| **Medium (3~4)** | 0.581 | 0.529 | 0.554 | 0.710 | 18 | 53 | 13 | 16 | 100 |
| **High (5~6)** | 0.500 | 0.029 | 0.056 | 0.660 | 1 | 65 | 1 | 33 | 100 |

#### Detailed Category Analysis

|         |                                   |low<br>(1~2)||medium<br>(3~4)||high<br>(5~6)||
|---------------|-----------------------------------------|-------------|------|--------|------|--------------|------|
|category_big   |category                                 |filtered<br>count        |filtered<br>mean  |filtered<br>count   |filtered<br>mean  |filtered<br>count         |filtered<br>mean  |
|Hate Speech    |['Age', 'Religion']            |1            |1.000 |1       |1.000 |0             |0.000 |
|Hate Speech    |['Age']            |7            |0.875 |4       |0.500 |1             |0.125 |
|Hate Speech    |['Gender']            |2            |0.500 |0       |0.000 |0             |0.000 |
|Hate Speech    |['Origin', 'Age']            |3            |1.000 |2       |0.667 |0             |0.000 |
|Hate Speech    |['Origin', 'Religion']            |2            |1.000 |2       |1.000 |0             |0.000 |
|Hate Speech    |['Origin']            |5            |1.000 |3       |0.600 |0             |0.000 |
|Hate Speech    |['Physical']            |1            |1.000 |1       |1.000 |0             |0.000 |
|Hate Speech    |['Politics', 'Age']            |1            |1.000 |1       |1.000 |0             |0.000 |
|Hate Speech    |['Politics', 'Physical']            |1            |1.000 |0       |0.000 |0             |0.000 |
|Hate Speech    |['Politics', 'Religion']            |1            |1.000 |1       |1.000 |0             |0.000 |
|Hate Speech    |['Politics']            |2            |1.000 |2       |1.000 |0             |0.000 |
|Hate Speech    |['Profanity']            |2            |1.000 |1       |0.500 |0             |0.000 |
|Hate Speech    |['Race']            |1            |1.000 |0       |0.000 |0             |0.000 |
|Hate Speech    |['Religion']            |2            |1.000 |0       |0.000 |0             |0.000 |
|Not Hate Speech    |['Not Hate Speech']            |45            |0.682 |13       |0.197 |1             |0.015 |
|**Filtering Total**|                                         |             |      |        |      |              |      |
|**Hate Speech**    |-                                        |**31**          |**0.912** |**18**      |**0.529** |**1**            |**0.029** |
|**Not Hate Speech**|-                                        |**45**          |**0.682** |**13**      |**0.197** |**1**             |**0.015** |