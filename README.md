# Content Filter evaluation tool using Korean hate-speech dataset

## Overview

Azure OpenAI Service includes a content filtering system that works alongside LLM, including image generation models. This system works by running both the prompt and completion through an ensemble of classification models designed to detect and prevent the output of harmful content. This tool evaluates two different content safety approaches:

1. **Content Filter**: Azure OpenAI's built-in content filtering system
2. **Content Safety**: Azure Content Safety service for dedicated content analysis

The content filtering models for the hate, sexual, violence, and self-harm categories support English, German, Japanese, Spanish, French, Italian, Portuguese, and Chinese. The service can work in many other languages however, the quality may vary which means that testing is essential especially for non-supported language such as Korean. In addition, even if you set up the content filter for a supported language, you need to test it to ensure that your filter detects the content at the severity levels you set up for prompts and completions. This tool performs benchmarking on hate-speech dataset with minimal time and effort, allowing you to understand the current performance of your established content filter, what types of content have been filtered, and to configure appropriate levels of your content filter.

## Features

- **Multi-processing support**: Parallel execution for faster processing
- **Batch processing**: Configurable batch sizes for optimal performance
- **Multiple evaluation targets**: 
  - `content_filter`: Azure OpenAI Content Filter evaluation
  - `content_safety`: Azure Content Safety service evaluation
- **Multiple model configurations**: Support for different deployment configurations
- **Environment-based configuration**: Support for multiple `.env` files for different deployments

---

## Blocking Threshold in Content Filter VS Severity Level in Content Safety

> ⚠ **Important:**  
> The meaning of "higher" and "lower" levels is **opposite** between Content Filter's **Blocking Threshold** and Content Safety's **Severity Level**.
> 
> - **Content Filter (Blocking Threshold)**: Higher → blocks **more** content.  
> - **Content Safety (Severity Level)**: Lower → detects **more** harmful content.



### 📌 Easy Explanation
- **Content Filter**: **Low → blocks less**, **High → blocks more**  
- **Content Safety**: **Low (1–2) → highly sensitive**, **High (5–6) → only catches the most severe**  
- In other words, **"Low"** in Content Filter ≠ **"Low"** in Content Safety. They mean the opposite.



### 🔍 Comparison Table

| Content Filter<br>(Blocking Threshold) | Content Safety<br>(Severity Level) | Description |
|----------------------------------------|--------------------------------------|-------------|
| Low                                    | High (5–6)                                  | Content Filter: Blocks mild cases only.<br>Content Safety: Catches only the most severe harmful content. |
| Medium                                 | Medium (3–4)                                  | Content Filter: Balanced blocking.<br>Content Safety: Catches moderate harmful content. |
| High                                   | Low (1–2)                                  | Content Filter: Blocks almost everything suspicious.<br>Content Safety: Detects even mild or borderline harmful content. |

### 🎯 Visual Level Diagram
Content Filter (Blocking Threshold)
Low ──▢▢──────────── High
Less Blocking More Blocking

Content Safety (Severity Level)
Low ──■■──────────── High
More Sensitive Less Sensitive
(Detects mild cases) (Only severe cases)


**Legend:**  
- **▢▢** = Blocking intensity in Content Filter  
- **■■** = Detection sensitivity in Content Safety  


### 💡 Quick Memory Tip
- **Content Filter**: *Low → blocks less*  
- **Content Safety**: *Low → detects more*

---

## Usage

### Prerequisites

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment files using .env.sample:
   - `.env_gpt-4-nano-low`: Configuration for low filtering threshold
   - `.env_gpt-4-nano-high`: Configuration for high filtering threshold

### Command Line Arguments

```bash
python main.py [OPTIONS]

Options:
  --num_samples INT          Number of samples to process (default: 2000)
  --is_random BOOL          Whether to randomize samples (default: False) 
  --is_debug BOOL           Debug mode (default: False)
  --num_debug_samples INT   Number of debug samples (default: 15)
  --model_provider STR      Model provider (default: azureopenai)
  --hf_model_id STR         Hugging Face model ID (default: gpt-4-nano)
  --max_retries INT         Maximum retry attempts (default: 3)
  --max_tokens INT          Maximum tokens (default: 256)
  --temperature FLOAT       Temperature (default: 0)
  --batch_size INT          Batch size (default: 10)
  --evaluation_target STR   Evaluation target: content_filter or content_safety
  --use_multiprocessing BOOL Enable multiprocessing (default: True)
  --max_workers INT         Maximum worker processes (default: 3)
  --locale STR              Output language (default: ko-KR)
```



### Tunable parameters
```python
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--is_random", type=str2bool, default=False)
    parser.add_argument("--is_debug", type=str2bool, default=True)
    parser.add_argument("--num_debug_samples", type=int, default=30)
    parser.add_argument("--model_provider", type=str, default="azureopenai")
    parser.add_argument("--split", type=str, default="valid")
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--evaluation_target", type=str, default="content_filter", 
                       choices=["content_filter", "content_safety"],
                       help="Target evaluation: content_filter (Azure OpenAI Content Filter) or content_safety (Azure Content Safety)")
    
    parser.add_argument("--use_multiprocessing", type=str2bool, default=True, help="Enable multiprocessing")
    parser.add_argument("--max_workers", type=int, default=3, help="Maximum number of worker processes")
    
    parser.add_argument("--locale", type=str, default="ko-KR", 
                       choices=["ko-KR", "en-US"],
                       help="Output language: ko-KR (Korean) or en-US (English)")
```


### Running Evaluations

#### Single Evaluation
```bash
# Content Filter evaluation
DOTENV_PATH=.env_gpt-4-nano-low python main.py --evaluation_target content_filter

# Content Safety evaluation  
DOTENV_PATH=.env_gpt-4-nano-low python main.py --evaluation_target content_safety
```

#### Batch Evaluation
```bash
# Run all evaluations for multiple models and evaluation targets
./run_all_env.sh
```

### Environment Configuration

#### For Azure OpenAI Content Filter 
```bash
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4-nano-low
AZURE_OPENAI_API_VERSION=2024-02-15-preview
FILTER_NAME=defaultv2
BLOCKING_THRESHOLD_LEVEL=middle
```

#### For Azure Content Safety
```bash
CONTENT_SAFETY_ENDPOINT=https://your-content-safety-resource.cognitiveservices.azure.com/
CONTENT_SAFETY_KEY=your-content-safety-key-here
# Higher levels focus only on the most severe harmful content, while lower levels also detect milder or borderline cases.
# 1-2: low
# 3-4: medium
# 5-6: high
CONTENT_SAFETY_THRESHOLD=4
```

---

## Korean Hate Speech Detection Evaluation Report

## Evaluation Overview 

- **Dataset**: K-MHaS (Korean Multi-label Hate Speech Dataset), 100 samples
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

### 🏆 Best performance
#### based on F1-Score:
    1. Content Filter (high): F1=0.577 (P=0.429, R=0.882)
    2. Content Safety (1~2): F1=0.564 (P=0.408, R=0.912)
    3. Content Safety (3-4): F1=0.554 (P=0.581, R=0.529)

---

### The Korean Multi-label Hate Speech Dataset, K-MHaS 
The Korean Multi-label Hate Speech Dataset, K-MHaS, consists of 109,692 utterances from Korean online news comments, labelled with 8 fine-grained hate speech classes (labels: Politics, Origin, Physical, Age, Gender, Religion, Race, Profanity) or Not Hate Speech class. Each utterance provides from a single to four labels that can handles Korean language patterns effectively. For more details, please refer to our paper about K-MHaS, published at COLING 2022. 

- [Paper](https://aclanthology.org/2022.coling-1.311/), [Hugging Face](https://huggingface.co/datasets/nayohan/K-MHaS)


## References

[K-MHaS: A Multi-label Hate Speech Detection Dataset in Korean Online News Comment](https://aclanthology.org/2022.coling-1.311) (Lee et al., COLING 2022)
