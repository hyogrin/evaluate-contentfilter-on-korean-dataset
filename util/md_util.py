import pandas as pd
import os
import glob
from typing import List, Dict, Tuple
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

def calculate_metrics_from_csv(csv_path: str) -> Dict[str, float]:
    """
    원본 CSV 파일에서 precision, recall, f1, accuracy 계산
    
    Args:
        csv_path: 원본 결과 CSV 파일 경로
        
    Returns:
        메트릭들을 포함한 딕셔너리
    """
    try:
        df = pd.read_csv(csv_path)
        
        # 카테고리 분류 (여러 라벨이 있을 수 있으므로 문자열로 처리)
        df['category_big'] = df['category'].apply(lambda x: 'Not Hate Speech' if 'Not Hate Speech' in str(x) else 'Hate Speech')
        
        # 실제 라벨: Hate Speech = 1, Not Hate Speech = 0
        actual = df['category_big'].apply(lambda x: 1 if x == 'Hate Speech' else 0).values
        
        # 예측 라벨: filtered=True이면 Hate Speech로 예측 = 1
        predicted = df['filtered'].apply(lambda x: 1 if x else 0).values
        
        # 메트릭 계산
        precision = precision_score(actual, predicted, zero_division=0)
        recall = recall_score(actual, predicted, zero_division=0)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = accuracy_score(actual, predicted)
        
        # Confusion Matrix
        cm = confusion_matrix(actual, predicted)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'total_samples': len(df),
            'hate_speech_samples': sum(actual),
            'normal_samples': len(actual) - sum(actual)
        }
        
    except Exception as e:
        print(f"Error calculating metrics from {csv_path}: {e}")
        return {
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'accuracy': 0.0,
            'true_positives': 0, 'true_negatives': 0, 'false_positives': 0, 'false_negatives': 0,
            'total_samples': 0, 'hate_speech_samples': 0, 'normal_samples': 0
        }


def find_results_files(results_dir: str, evaluation_type: str) -> Dict[str, str]:
    """
    results 폴더에서 평가 결과 파일들을 자동으로 찾기
    """
    result_files = {}
    
    if not os.path.exists(results_dir):
        print(f"❌ {results_dir} 폴더가 존재하지 않습니다.")
        return result_files
    
    # 모든 CSV 파일 가져오기
    all_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    
    # evaluation_type에 맞는 파일들 필터링
    target_files = [f for f in all_files if f.startswith(f"[{evaluation_type}]")]
    
    print(f"Found {len(target_files)} files for {evaluation_type}: {target_files}")
    
    if evaluation_type == 'content_filter':
        for filename in target_files:
            # [content_filter]-{filter_name}-{threshold_level}-{date}.csv
            parts = filename.replace('.csv', '').split('-')
            
            if len(parts) >= 4:
                filter_name = parts[1]  # DefaultV2, lowContentFilter, highContentFilter
                threshold_level = parts[2]  # low, medium, high
                
                # 레벨 매핑
                level_mapping = {
                    'lowContentFilter': 'low',
                    'DefaultV2': 'medium',
                    'highContentFilter': 'high',
                    'low': 'low',
                    'medium': 'medium', 
                    'high': 'high'
                }
                
                mapped_level = level_mapping.get(filter_name, level_mapping.get(threshold_level, 'unknown'))
                if mapped_level != 'unknown':
                    result_files[mapped_level] = os.path.join(results_dir, filename)
                    
    elif evaluation_type == 'content_safety':
        for filename in target_files:
            if 'threshold_' in filename:
                # [content_safety]-threshold_{number}-{date}.csv
                try:
                    threshold_part = filename.split('threshold_')[1].split('-')[0]
                    threshold_num = int(threshold_part)
                    
                    # threshold 번호에 따른 레벨 분류
                    if threshold_num in [1, 2]:
                        level = 'low'
                    elif threshold_num in [3, 4]:
                        level = 'medium'
                    elif threshold_num in [5, 6]:
                        level = 'high'
                    else:
                        continue
                        
                    result_files[level] = os.path.join(results_dir, filename)
                except (ValueError, IndexError):
                    print(f"⚠️ 파일명 파싱 실패: {filename}")
                    continue
    
    print(f"✅ 매핑된 파일들: {result_files}")
    return result_files


def generate_category_analysis_from_csv(csv_path: str) -> pd.DataFrame:
    """
    원본 CSV에서 카테고리별 분석 데이터 생성
    """
    df = pd.read_csv(csv_path)
    df['category_big'] = df['category'].apply(lambda x: 'Not Hate Speech' if 'Not Hate Speech' in str(x) else 'Hate Speech')
    
    # 카테고리별 통계
    category_stats = df.groupby(['category_big', 'category']).agg(
        total_count=('filtered', 'count'),
        filtered_count=('filtered', 'sum'),
        filtered_mean=('filtered', 'mean')
    ).reset_index()
    
    return category_stats


def safe_division(numerator: int, denominator: int) -> float:
    """안전한 나눗셈 함수"""
    return numerator / denominator if denominator > 0 else 0.0


def generate_content_filter_md(results_dir: str) -> str:
    """
    Content Filter 결과들을 메트릭과 함께 마크다운 테이블로 생성
    
    Args:
        results_dir: results 폴더 경로
        
    Returns:
        마크다운 테이블 문자열
    """
    
    # 파일들 자동 찾기
    files = find_results_files(results_dir, 'content_filter')
    
    if len(files) < 3:
        return f"⚠️ Content Filter 파일이 부족합니다. 필요: 3개, 발견: {len(files)}개"
    
    # 메트릭 계산
    low_metrics = calculate_metrics_from_csv(f"{files['low']}") if 'low' in files else {}
    medium_metrics = calculate_metrics_from_csv(f"{files['medium']}") if 'medium' in files else {}
    high_metrics = calculate_metrics_from_csv(f"{files['high']}") if 'high' in files else {}
    
    # 카테고리별 분석
    low_categories = generate_category_analysis_from_csv(f"{files['low']}") if 'low' in files else pd.DataFrame()
    medium_categories = generate_category_analysis_from_csv(f"{files['medium']}") if 'medium' in files else pd.DataFrame()
    high_categories = generate_category_analysis_from_csv(f"{files['high']}") if 'high' in files else pd.DataFrame()
    
    # 모든 카테고리 조합 수집
    all_categories = set()
    for df in [low_categories, medium_categories, high_categories]:
        if not df.empty:
            all_categories.update(zip(df['category_big'], df['category']))
    
    # 정렬: Hate Speech 먼저, 그 다음 Not Hate Speech
    sorted_categories = sorted(all_categories, key=lambda x: (x[0] != 'Hate Speech', x[1]))
    
    # 마크다운 테이블 헤더
    md_lines = [
        "### Content Filter Evaluation Results",
        "",
        "#### Overall Performance Metrics",
        "",
        "| Blocking threshold | Precision | Recall | F1-Score | Accuracy | TP | TN | FP | FN | Total |",
        "|-----------|-----------|--------|----------|----------|----|----|----|----|-------|",
        f"| **Low** | {low_metrics.get('precision', 0):.3f} | {low_metrics.get('recall', 0):.3f} | {low_metrics.get('f1', 0):.3f} | {low_metrics.get('accuracy', 0):.3f} | {low_metrics.get('true_positives', 0)} | {low_metrics.get('true_negatives', 0)} | {low_metrics.get('false_positives', 0)} | {low_metrics.get('false_negatives', 0)} | {low_metrics.get('total_samples', 0)} |",
        f"| **Medium** | {medium_metrics.get('precision', 0):.3f} | {medium_metrics.get('recall', 0):.3f} | {medium_metrics.get('f1', 0):.3f} | {medium_metrics.get('accuracy', 0):.3f} | {medium_metrics.get('true_positives', 0)} | {medium_metrics.get('true_negatives', 0)} | {medium_metrics.get('false_positives', 0)} | {medium_metrics.get('false_negatives', 0)} | {medium_metrics.get('total_samples', 0)} |",
        f"| **High** | {high_metrics.get('precision', 0):.3f} | {high_metrics.get('recall', 0):.3f} | {high_metrics.get('f1', 0):.3f} | {high_metrics.get('accuracy', 0):.3f} | {high_metrics.get('true_positives', 0)} | {high_metrics.get('true_negatives', 0)} | {high_metrics.get('false_positives', 0)} | {high_metrics.get('false_negatives', 0)} | {high_metrics.get('total_samples', 0)} |",
        "",
        "#### Detailed Category Analysis",
        "",
        "|         |                                   |low||medium||high||",
        "|---------------|-----------------------------------------|-------------|------|--------|------|--------------|------|",
        "|category_big   |category                                 |filtered<br>count        |filtered<br>mean  |filtered<br>count   |filtered<br>mean  |filtered<br>count         |filtered<br>mean  |"
    ]
    
    # 데이터 행들
    hate_speech_totals = {'low': {'count': 0, 'total': 0}, 'medium': {'count': 0, 'total': 0}, 'high': {'count': 0, 'total': 0}}
    not_hate_speech_totals = {'low': {'count': 0, 'total': 0}, 'medium': {'count': 0, 'total': 0}, 'high': {'count': 0, 'total': 0}}
    
    for category_big, category in sorted_categories:
        # 각 threshold별 데이터 찾기
        low_row = low_categories[(low_categories['category_big'] == category_big) & (low_categories['category'] == category)] if not low_categories.empty else pd.DataFrame()
        medium_row = medium_categories[(medium_categories['category_big'] == category_big) & (medium_categories['category'] == category)] if not medium_categories.empty else pd.DataFrame()
        high_row = high_categories[(high_categories['category_big'] == category_big) & (high_categories['category'] == category)] if not high_categories.empty else pd.DataFrame()
        
        # 데이터 추출 (없으면 0으로 설정)
        low_count = int(low_row['filtered_count'].iloc[0]) if not low_row.empty else 0
        low_mean = float(low_row['filtered_mean'].iloc[0]) if not low_row.empty else 0.0
        low_total = int(low_row['total_count'].iloc[0]) if not low_row.empty else 0
        
        medium_count = int(medium_row['filtered_count'].iloc[0]) if not medium_row.empty else 0
        medium_mean = float(medium_row['filtered_mean'].iloc[0]) if not medium_row.empty else 0.0
        medium_total = int(medium_row['total_count'].iloc[0]) if not medium_row.empty else 0
        
        high_count = int(high_row['filtered_count'].iloc[0]) if not high_row.empty else 0
        high_mean = float(high_row['filtered_mean'].iloc[0]) if not high_row.empty else 0.0
        high_total = int(high_row['total_count'].iloc[0]) if not high_row.empty else 0
        
        # 총계 계산용
        if category_big == 'Hate Speech':
            hate_speech_totals['low']['count'] += low_count
            hate_speech_totals['low']['total'] += low_total
            hate_speech_totals['medium']['count'] += medium_count
            hate_speech_totals['medium']['total'] += medium_total
            hate_speech_totals['high']['count'] += high_count
            hate_speech_totals['high']['total'] += high_total
        else:
            not_hate_speech_totals['low']['count'] += low_count
            not_hate_speech_totals['low']['total'] += low_total
            not_hate_speech_totals['medium']['count'] += medium_count
            not_hate_speech_totals['medium']['total'] += medium_total
            not_hate_speech_totals['high']['count'] += high_count
            not_hate_speech_totals['high']['total'] += high_total
        
        # 테이블 행 생성
        md_lines.append(f"|{category_big}    |{category}            |{low_count}            |{low_mean:.3f} |{medium_count}       |{medium_mean:.3f} |{high_count}             |{high_mean:.3f} |")
    
    # 총계 비율 계산 (안전한 나눗셈 사용)
    hs_low_rate = safe_division(hate_speech_totals['low']['count'], hate_speech_totals['low']['total'])
    hs_medium_rate = safe_division(hate_speech_totals['medium']['count'], hate_speech_totals['medium']['total'])
    hs_high_rate = safe_division(hate_speech_totals['high']['count'], hate_speech_totals['high']['total'])
    
    nhs_low_rate = safe_division(not_hate_speech_totals['low']['count'], not_hate_speech_totals['low']['total'])
    nhs_medium_rate = safe_division(not_hate_speech_totals['medium']['count'], not_hate_speech_totals['medium']['total'])
    nhs_high_rate = safe_division(not_hate_speech_totals['high']['count'], not_hate_speech_totals['high']['total'])
    
    # 총계 행 추가
    md_lines.extend([
        "|**Filtering Total**|                                         |             |      |        |      |              |      |",
        f"|**Hate Speech**    |-                                        |**{hate_speech_totals['low']['count']}**          |**{hs_low_rate:.3f}** |**{hate_speech_totals['medium']['count']}**      |**{hs_medium_rate:.3f}** |**{hate_speech_totals['high']['count']}**            |**{hs_high_rate:.3f}** |",
        f"|**Not Hate Speech**|-                                        |**{not_hate_speech_totals['low']['count']}**          |**{nhs_low_rate:.3f}** |**{not_hate_speech_totals['medium']['count']}**      |**{nhs_medium_rate:.3f}** |**{not_hate_speech_totals['high']['count']}**             |**{nhs_high_rate:.3f}** |"
    ])
    
    return "\n".join(md_lines)


def generate_content_safety_md(results_dir: str) -> str:
    """
    Content Safety 결과들을 메트릭과 함께 마크다운 테이블로 생성
    
    Args:
        results_dir: results 폴더 경로
        
    Returns:
        마크다운 테이블 문자열
    """
    
    # 파일들 자동 찾기
    files = find_results_files(results_dir, 'content_safety')
    
    if len(files) < 3:
        return f"⚠️ Content Safety 파일이 부족합니다. 필요: 3개, 발견: {len(files)}개"
    
    # 메트릭 계산
    low_metrics = calculate_metrics_from_csv(files['low']) if 'low' in files else {}
    medium_metrics = calculate_metrics_from_csv(files['medium']) if 'medium' in files else {}
    high_metrics = calculate_metrics_from_csv(files['high']) if 'high' in files else {}
    
    # 카테고리별 분석
    low_categories = generate_category_analysis_from_csv(files['low']) if 'low' in files else pd.DataFrame()
    medium_categories = generate_category_analysis_from_csv(files['medium']) if 'medium' in files else pd.DataFrame()
    high_categories = generate_category_analysis_from_csv(files['high']) if 'high' in files else pd.DataFrame()
    
    # 모든 카테고리 조합 수집
    all_categories = set()
    for df in [low_categories, medium_categories, high_categories]:
        if not df.empty:
            all_categories.update(zip(df['category_big'], df['category']))
    
    # 정렬: Hate Speech 먼저, 그 다음 Not Hate Speech
    sorted_categories = sorted(all_categories, key=lambda x: (x[0] != 'Hate Speech', x[1]))
    
    # 마크다운 테이블 헤더
    md_lines = [
        "### Content Safety Evaluation Results",
        "",
        "#### Overall Performance Metrics",
        "",
        "| Severity level | Precision | Recall | F1-Score | Accuracy | TP | TN | FP | FN | Total |",
        "|-----------|-----------|--------|----------|----------|----|----|----|----|-------|",
        f"| **Low (1~2)** | {low_metrics.get('precision', 0):.3f} | {low_metrics.get('recall', 0):.3f} | {low_metrics.get('f1', 0):.3f} | {low_metrics.get('accuracy', 0):.3f} | {low_metrics.get('true_positives', 0)} | {low_metrics.get('true_negatives', 0)} | {low_metrics.get('false_positives', 0)} | {low_metrics.get('false_negatives', 0)} | {low_metrics.get('total_samples', 0)} |",
        f"| **Medium (3~4)** | {medium_metrics.get('precision', 0):.3f} | {medium_metrics.get('recall', 0):.3f} | {medium_metrics.get('f1', 0):.3f} | {medium_metrics.get('accuracy', 0):.3f} | {medium_metrics.get('true_positives', 0)} | {medium_metrics.get('true_negatives', 0)} | {medium_metrics.get('false_positives', 0)} | {medium_metrics.get('false_negatives', 0)} | {medium_metrics.get('total_samples', 0)} |",
        f"| **High (5~6)** | {high_metrics.get('precision', 0):.3f} | {high_metrics.get('recall', 0):.3f} | {high_metrics.get('f1', 0):.3f} | {high_metrics.get('accuracy', 0):.3f} | {high_metrics.get('true_positives', 0)} | {high_metrics.get('true_negatives', 0)} | {high_metrics.get('false_positives', 0)} | {high_metrics.get('false_negatives', 0)} | {high_metrics.get('total_samples', 0)} |",
        "",
        "#### Detailed Category Analysis",
        "",
        "|         |                                   |low<br>(1~2)||medium<br>(3~4)||high<br>(5~6)||",
        "|---------------|-----------------------------------------|-------------|------|--------|------|--------------|------|",
        "|category_big   |category                                 |filtered<br>count        |filtered<br>mean  |filtered<br>count   |filtered<br>mean  |filtered<br>count         |filtered<br>mean  |"
    ]
    
    # 나머지 로직은 content_filter와 동일
    hate_speech_totals = {'low': {'count': 0, 'total': 0}, 'medium': {'count': 0, 'total': 0}, 'high': {'count': 0, 'total': 0}}
    not_hate_speech_totals = {'low': {'count': 0, 'total': 0}, 'medium': {'count': 0, 'total': 0}, 'high': {'count': 0, 'total': 0}}
    
    for category_big, category in sorted_categories:
        # 각 threshold별 데이터 찾기
        low_row = low_categories[(low_categories['category_big'] == category_big) & (low_categories['category'] == category)] if not low_categories.empty else pd.DataFrame()
        medium_row = medium_categories[(medium_categories['category_big'] == category_big) & (medium_categories['category'] == category)] if not medium_categories.empty else pd.DataFrame()
        high_row = high_categories[(high_categories['category_big'] == category_big) & (high_categories['category'] == category)] if not high_categories.empty else pd.DataFrame()
        
        # 데이터 추출 (없으면 0으로 설정)
        low_count = int(low_row['filtered_count'].iloc[0]) if not low_row.empty else 0
        low_mean = float(low_row['filtered_mean'].iloc[0]) if not low_row.empty else 0.0
        low_total = int(low_row['total_count'].iloc[0]) if not low_row.empty else 0
        
        medium_count = int(medium_row['filtered_count'].iloc[0]) if not medium_row.empty else 0
        medium_mean = float(medium_row['filtered_mean'].iloc[0]) if not medium_row.empty else 0.0
        medium_total = int(medium_row['total_count'].iloc[0]) if not medium_row.empty else 0
        
        high_count = int(high_row['filtered_count'].iloc[0]) if not high_row.empty else 0
        high_mean = float(high_row['filtered_mean'].iloc[0]) if not high_row.empty else 0.0
        high_total = int(high_row['total_count'].iloc[0]) if not high_row.empty else 0
        
        # 총계 계산용
        if category_big == 'Hate Speech':
            hate_speech_totals['low']['count'] += low_count
            hate_speech_totals['low']['total'] += low_total
            hate_speech_totals['medium']['count'] += medium_count
            hate_speech_totals['medium']['total'] += medium_total
            hate_speech_totals['high']['count'] += high_count
            hate_speech_totals['high']['total'] += high_total
        else:
            not_hate_speech_totals['low']['count'] += low_count
            not_hate_speech_totals['low']['total'] += low_total
            not_hate_speech_totals['medium']['count'] += medium_count
            not_hate_speech_totals['medium']['total'] += medium_total
            not_hate_speech_totals['high']['count'] += high_count
            not_hate_speech_totals['high']['total'] += high_total
        
        # 테이블 행 생성
        md_lines.append(f"|{category_big}    |{category}            |{low_count}            |{low_mean:.3f} |{medium_count}       |{medium_mean:.3f} |{high_count}             |{high_mean:.3f} |")
    
    # 총계 비율 계산 (안전한 나눗셈 사용)
    hs_low_rate = safe_division(hate_speech_totals['low']['count'], hate_speech_totals['low']['total'])
    hs_medium_rate = safe_division(hate_speech_totals['medium']['count'], hate_speech_totals['medium']['total'])
    hs_high_rate = safe_division(hate_speech_totals['high']['count'], hate_speech_totals['high']['total'])
    
    nhs_low_rate = safe_division(not_hate_speech_totals['low']['count'], not_hate_speech_totals['low']['total'])
    nhs_medium_rate = safe_division(not_hate_speech_totals['medium']['count'], not_hate_speech_totals['medium']['total'])
    nhs_high_rate = safe_division(not_hate_speech_totals['high']['count'], not_hate_speech_totals['high']['total'])
    
    # 총계 행 추가
    md_lines.extend([
        "|**Filtering Total**|                                         |             |      |        |      |              |      |",
        f"|**Hate Speech**    |-                                        |**{hate_speech_totals['low']['count']}**          |**{hs_low_rate:.3f}** |**{hate_speech_totals['medium']['count']}**      |**{hs_medium_rate:.3f}** |**{hate_speech_totals['high']['count']}**            |**{hs_high_rate:.3f}** |",
        f"|**Not Hate Speech**|-                                        |**{not_hate_speech_totals['low']['count']}**          |**{nhs_low_rate:.3f}** |**{not_hate_speech_totals['medium']['count']}**      |**{nhs_medium_rate:.3f}** |**{not_hate_speech_totals['high']['count']}**             |**{nhs_high_rate:.3f}** |"
    ])
    
    return "\n".join(md_lines)


def generate_performance_comparison_md(results_dir: str) -> str:
    """
    Content Filter vs Content Safety 성능 비교 테이블 생성
    """
    
    md_lines = [
        "### Performance Comparison: Content Filter vs Content Safety",
        "",
        "| Method | Blocking/Severity | Precision | Recall | F1-Score | Accuracy | False Positive Rate | False Negative Rate |",
        "|--------|-----------|-----------|--------|----------|----------|--------------------|--------------------|"
    ]
    
    # Content Filter 메트릭들
    cf_files = find_results_files(results_dir, 'content_filter')
    for level in ['low', 'medium', 'high']:
        if level in cf_files:
            metrics = calculate_metrics_from_csv(cf_files[level])
            
            fp_rate = safe_division(metrics['false_positives'], metrics['false_positives'] + metrics['true_negatives'])
            fn_rate = safe_division(metrics['false_negatives'], metrics['false_negatives'] + metrics['true_positives'])
            
            md_lines.append(f"| **Content Filter** | {level.capitalize()} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1']:.3f} | {metrics['accuracy']:.3f} | {fp_rate:.3f} | {fn_rate:.3f} |")
    
    # Content Safety 메트릭들
    cs_files = find_results_files(results_dir, 'content_safety')
    for level in ['low', 'medium', 'high']:
        if level in cs_files:
            metrics = calculate_metrics_from_csv(cs_files[level])
            
            fp_rate = safe_division(metrics['false_positives'], metrics['false_positives'] + metrics['true_negatives'])
            fn_rate = safe_division(metrics['false_negatives'], metrics['false_negatives'] + metrics['true_positives'])
            
            threshold_label = "1~2" if level == 'low' else "3~4" if level == 'medium' else "5~6"
            md_lines.append(f"| **Content Safety** | {threshold_label} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1']:.3f} | {metrics['accuracy']:.3f} | {fp_rate:.3f} | {fn_rate:.3f} |")
    
    return "\n".join(md_lines)


def generate_complete_evaluation_report(results_dir: str) -> str:
    """
    전체 평가 보고서 생성 (단순화된 버전)
    
    Args:
        results_dir: results 폴더 경로
        
    Returns:
        완전한 마크다운 보고서
    """
    md_content = []
    
    # 성능 비교 테이블
    comparison_md = generate_performance_comparison_md(results_dir)
    md_content.append(comparison_md)
    md_content.append("\n\n")
    
    # Content Filter 상세 결과
    content_filter_md = generate_content_filter_md(results_dir)
    md_content.append(content_filter_md)
    md_content.append("\n\n")
    
    # Content Safety 상세 결과
    content_safety_md = generate_content_safety_md(results_dir)
    md_content.append(content_safety_md)
    
    return "".join(md_content)