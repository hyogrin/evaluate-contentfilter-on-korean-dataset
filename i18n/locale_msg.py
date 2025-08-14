# locale_msg.py

LOCALE_MESSAGES = {
    "ko-KR": {
        # 섹션 제목
        "detailed_category_analysis": "=== 세부 카테고리별 분석 ===",
        "confusion_matrix": "=== 혼동 행렬 (Confusion Matrix) ===",
        "performance_metrics": "=== 성능 지표 ===",
        "overall_summary": "📊 전체 결과 요약",
        "filtering_results": "🔍 필터링 결과:",
        "performance_interpretation": "🎪 모델 성능 해석:",
        "practical_analysis": "💡 실용적 분석:",
        "saved_files": "💾 결과 파일 저장 완료:",
        
        # 상세 메시지
        "filtering_rate": "필터링율",
        "filtered_items": "건 필터링됨",
        "actual_vs_predicted": "실제 vs 예측 (1=혐오표현, 0=일반텍스트)",
        "true_negatives": "일반텍스트를 올바르게 분류",
        "false_positives": "일반텍스트를 혐오표현으로 잘못 분류",
        "false_negatives": "혐오표현을 놓친 경우",
        "true_positives": "혐오표현을 올바르게 탐지",
        
        # 성능 지표 설명
        "precision_desc": "정밀도 - 혐오표현으로 예측한 것 중 실제 혐오표현 비율",
        "recall_desc": "재현율 - 실제 혐오표현 중 올바르게 탐지한 비율",
        "f1_desc": "F1-Score - 정밀도와 재현율의 조화평균",
        "accuracy_desc": "정확도 - 전체 예측 중 올바른 예측 비율",
        
        # 통계 라벨
        "total_samples": "전체 샘플 수",
        "hate_speech": "혐오표현",
        "normal_text": "일반텍스트",
        "filtered": "필터링됨",
        "passed": "통과함",
        
        # 성능 평가
        "excellent": "우수",
        "good": "보통",
        "needs_improvement": "개선 필요",
        "accurate_detection": "정확한 탐지",
        "some_false_positives": "일부 오탐지 있음",
        "many_false_positives": "많은 오탐지",
        "most_detected": "대부분 탐지",
        "some_missed": "일부 누락",
        "many_missed": "많은 누락",
        
        # 실용적 분석
        "false_positive_rate": "오탐지율 (정상 글을 혐오표현으로 분류)",
        "false_negative_rate": "누락율 (혐오표현을 놓침)",
        "high_false_positive_warning": "오탐지율이 높음 - 정상 사용자 경험에 영향 가능",
        "high_false_negative_warning": "누락율이 높음 - 혐오표현 차단 효과 제한적",
        
        # 파일 경로
        "original_results": "원본 결과",
        "detailed_analysis": "세부 분석",
        "summary_analysis": "요약 분석",
        "visualization": "시각화",
        
        # HTML 리포트
        "evaluation_report": "평가 리포트",
        "model_info": "모델 정보",
        "dataset_info": "데이터셋 정보",
        "performance_summary": "성능 요약",
        "category_breakdown": "카테고리별 분석",
        "confusion_matrix_chart": "혼동 행렬",
        "recommendations": "권장사항"
    },
    "en-US": {
        # 섹션 제목
        "detailed_category_analysis": "=== Detailed Category Analysis ===",
        "confusion_matrix": "=== Confusion Matrix ===",
        "performance_metrics": "=== Performance Metrics ===",
        "overall_summary": "📊 Overall Results Summary",
        "filtering_results": "🔍 Filtering Results:",
        "performance_interpretation": "🎪 Model Performance Interpretation:",
        "practical_analysis": "💡 Practical Analysis:",
        "saved_files": "💾 Results Saved Successfully:",
        
        # 상세 메시지
        "filtering_rate": "Filtering Rate",
        "filtered_items": "items filtered",
        "actual_vs_predicted": "Actual vs Predicted (1=Hate Speech, 0=Normal Text)",
        "true_negatives": "Correctly classified normal text",
        "false_positives": "Normal text incorrectly classified as hate speech",
        "false_negatives": "Missed hate speech",
        "true_positives": "Correctly detected hate speech",
        
        # 성능 지표 설명
        "precision_desc": "Precision - Ratio of actual hate speech among predicted hate speech",
        "recall_desc": "Recall - Ratio of correctly detected among actual hate speech",
        "f1_desc": "F1-Score - Harmonic mean of precision and recall",
        "accuracy_desc": "Accuracy - Ratio of correct predictions among all predictions",
        
        # 통계 라벨
        "total_samples": "Total Samples",
        "hate_speech": "Hate Speech",
        "normal_text": "Normal Text",
        "filtered": "Filtered",
        "passed": "Passed",
        
        # 성능 평가
        "excellent": "Excellent",
        "good": "Good",
        "needs_improvement": "Needs Improvement",
        "accurate_detection": "Accurate Detection",
        "some_false_positives": "Some False Positives",
        "many_false_positives": "Many False Positives",
        "most_detected": "Most Detected",
        "some_missed": "Some Missed",
        "many_missed": "Many Missed",
        
        # 실용적 분석
        "false_positive_rate": "False Positive Rate (Normal text classified as hate speech)",
        "false_negative_rate": "False Negative Rate (Missed hate speech)",
        "high_false_positive_warning": "High false positive rate - May impact normal user experience",
        "high_false_negative_warning": "High false negative rate - Limited hate speech blocking effectiveness",
        
        # 파일 경로
        "original_results": "Original Results",
        "detailed_analysis": "Detailed Analysis",
        "summary_analysis": "Summary Analysis",
        "visualization": "Visualization",
        
        # HTML 리포트
        "evaluation_report": "Evaluation Report",
        "model_info": "Model Information",
        "dataset_info": "Dataset Information",
        "performance_summary": "Performance Summary",
        "category_breakdown": "Category Breakdown",
        "confusion_matrix_chart": "Confusion Matrix",
        "recommendations": "Recommendations"
    }
}

def get_message(locale, key, default=None):
    """Get localized message by key"""
    return LOCALE_MESSAGES.get(locale, {}).get(key, default or key)