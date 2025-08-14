import base64
from i18n.locale_msg import get_message
from datetime import datetime
import pytz
import os

def generate_html_report(html_path, data):
    """HTML 평가 리포트 생성"""
    locale = data['locale']
    
    # 이미지를 base64로 인코딩
    plot_base64 = ""
    if os.path.exists(data['plot_path']):
        with open(data['plot_path'], "rb") as img_file:
            plot_base64 = base64.b64encode(img_file.read()).decode('utf-8')
    
    html_content = f"""
<!DOCTYPE html>
<html lang="{locale[:2]}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{get_message(locale, 'evaluation_report')} - {data['filename']}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            margin-bottom: 15px;
            padding: 10px 0;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        h3 {{
            color: #7f8c8d;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .metric-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .confusion-matrix {{
            text-align: center;
            margin: 20px 0;
        }}
        .confusion-matrix img {{
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        tr:hover {{
            background-color: #e8f4f8;
        }}
        .warning {{
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .info {{
            background-color: #d1ecf1;
            border: 1px solid #bee5eb;
            color: #0c5460;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        .summary-stats {{
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
            flex-wrap: wrap;
        }}
        .stat-item {{
            text-align: center;
            margin: 10px;
        }}
        .stat-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .stat-label {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 {get_message(locale, 'evaluation_report')}</h1>
        
        <div class="info">
            <strong>{get_message(locale, 'model_info')}:</strong> {data['filename']}<br>
            <strong>{get_message(locale, 'dataset_info')}:</strong> Korean Hate Speech Dataset (K-MHaS)<br>
            <strong>Generated:</strong> {datetime.now(tz=pytz.timezone("Asia/Seoul")).strftime('%Y-%m-%d %H:%M:%S')}
        </div>

        <h2>🎯 {get_message(locale, 'performance_summary')}</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{data['precision']:.3f}</div>
                <div class="metric-label">Precision</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{data['recall']:.3f}</div>
                <div class="metric-label">Recall</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{data['f1']:.3f}</div>
                <div class="metric-label">F1-Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{data['accuracy']:.3f}</div>
                <div class="metric-label">Accuracy</div>
            </div>
        </div>

        <h2>📈 {get_message(locale, 'dataset_info')}</h2>
        <div class="summary-stats">
            <div class="stat-item">
                <div class="stat-value">{data['total_samples']:,}</div>
                <div class="stat-label">{get_message(locale, 'total_samples')}</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{data['hate_speech_samples']:,}</div>
                <div class="stat-label">{get_message(locale, 'hate_speech')}</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{data['normal_samples']:,}</div>
                <div class="stat-label">{get_message(locale, 'normal_text')}</div>
            </div>
        </div>

        <h2>📊 {get_message(locale, 'confusion_matrix_chart')}</h2>
        <div class="confusion-matrix">
            {"<img src='data:image/png;base64," + plot_base64 + "' alt='Confusion Matrix'/>" if plot_base64 else "<p>Confusion matrix image not available</p>"}
        </div>

        <h2>📋 {get_message(locale, 'category_breakdown')}</h2>
        <table>
            <thead>
                <tr>
                    <th>Category Type</th>
                    <th>Category Detail</th>
                    <th>Total Count</th>
                    <th>Filtered Count</th>
                    <th>Filtering Rate</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for _, row in data['category_count'].iterrows():
        html_content += f"""
                <tr>
                    <td>{row['category_big']}</td>
                    <td>{row['category']}</td>
                    <td>{row['total_count']:,}</td>
                    <td>{row['filtered_count']:,}</td>
                    <td>{row['filtered_mean']:.1%}</td>
                </tr>
        """
    
    html_content += f"""
            </tbody>
        </table>

        <h2>⚠️ {get_message(locale, 'recommendations')}</h2>
    """
    
    if data['false_positive_rate'] > 0.1:
        html_content += f'<div class="warning">⚠️ {get_message(locale, "high_false_positive_warning")} (FPR: {data["false_positive_rate"]:.1%})</div>'
    
    if data['false_negative_rate'] > 0.2:
        html_content += f'<div class="warning">⚠️ {get_message(locale, "high_false_negative_warning")} (FNR: {data["false_negative_rate"]:.1%})</div>'
    
    if data['precision'] > 0.8 and data['recall'] > 0.8:
        if locale == "ko-KR":
            html_content += '<div class="info">✅ 모델이 우수한 성능을 보이고 있습니다. 현재 설정을 유지하는 것을 권장합니다.</div>'
        else:
            html_content += '<div class="info">✅ The model shows excellent performance. We recommend maintaining the current settings.</div>'
    
    html_content += f"""
        <div class="footer">
            <p>Report generated by Content Filter Evaluation System</p>
            <p>© 2025 - Automated Evaluation Report</p>
        </div>
    </div>
</body>
</html>
    """
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
