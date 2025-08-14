#!/bin/bash

### Parallel execution version of run_all.sh with resume capability
env_files=(.env_gpt-oss-120b) 
is_debug=False
batch_size=30
max_tokens=256
temperature=0.01
max_parallel_jobs=2
num_debug_samples=100
categories=""  #별도로 실행할 카테고리를 넣으려면 여기에 카테고리명 입력, 기존의 데이터는 삭제되므로 주의 필요

echo "Found the following .env files:"
for env_file in "${env_files[@]}"; do
    echo "$env_file"
done

# 함수: 단일 모델의 모든 벤치마크 실행
run_model() {
    local env_file=$1
    local model_provider=$2
    
    echo "Starting evaluation for $env_file"
    
    # CLIcK
    DOTENV_PATH="$env_file" python click_main.py \
        --is_debug "$is_debug" \
        --model_provider "$model_provider" \
        --batch_size "$batch_size" \
        --max_tokens "$max_tokens" \
        --temperature "$temperature" \
        --template_type gpt5 \
        --categories "$categories" \
        --num_debug_samples 100 &
    
    # HAERAE 1.0
    # DOTENV_PATH="$env_file" python haerae_main.py \
    #     --is_debug "$is_debug" \
    #     --model_provider "$model_provider" \
    #     --batch_size "$batch_size" \
    #     --max_tokens "$max_tokens" \
    #     --temperature "$temperature" \
    #     --template_type basic \
    #     --categories "$categories" \
    #     --num_debug_samples 100  &
    
    # # KMMLU - 재시작 지원
    # DOTENV_PATH="$env_file" python kmmlu_main.py \
    #     --is_debug "$is_debug" \
    #     --model_provider "$model_provider" \
    #     --batch_size "$batch_size" \
    #     --max_tokens "$max_tokens" \
    #     --temperature "$temperature" \
    #     --template_type basic \
    #     --is_hard False \
    #     --use_few_shot False \
    #     --categories "$categories" &
    
    # # KMMLU (HARD) - 재시작 지원
    # DOTENV_PATH="$env_file" python kmmlu_main.py \
    #     --is_debug "$is_debug" \
    #     --model_provider "$model_provider" \
    #     --batch_size "$batch_size" \
    #     --max_tokens "$max_tokens" \
    #     --temperature "$temperature" \
    #     --template_type basic \
    #     --is_hard True \
    #     --use_few_shot False \
    #     --categories "$categories" &
    
    wait  # 해당 모델의 모든 작업이 완료될 때까지 대기
    echo "Completed evaluation for $env_file"
}

# 재시작 기능 안내
if [[ -n "$start_category" ]]; then
    echo "Resuming from category: $start_category"
else
    echo "Starting from the beginning (or skipping completed categories)"
fi

# 병렬 실행 관리
job_count=0
for env_file in "${env_files[@]}"; do
    if [[ "$env_file" == .env_gpt* ]]; then
        model_provider="azureopenai"
    else
        model_provider="azureml"
    fi
    
    # 백그라운드에서 실행
    run_model "$env_file" "$model_provider" &
    
    ((job_count++))
    
    # 최대 병렬 작업 수에 도달하면 일부 작업이 완료될 때까지 대기
    if (( job_count >= max_parallel_jobs )); then
        wait -n  # 하나의 작업이 완료될 때까지 대기
        ((job_count--))
    fi
done

wait  # 모든 작업 완료 대기
echo "All evaluations completed!"