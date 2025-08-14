#!/bin/bash

### Parallel execution version of run_all_gpts.sh with resume capability
env_files=(.env_gpt-4-nano-default, .env_gpt-4-nano-low, .env_gpt-4-nano-high) 
is_debug=True
batch_size=10
max_tokens=256
temperature=0.01
max_parallel_jobs=2
num_debug_samples=15
evaluation_targets=("content_filter" "content_safety")  # 두 가지 평가 방식

echo "Found the following .env files:"
for env_file in "${env_files[@]}"; do
    echo "$env_file"
done

echo "Evaluation targets: ${evaluation_targets[@]}"

# 함수: 단일 모델의 모든 벤치마크 실행
run_model() {
    local env_file=$1
    local model_provider=$2
    local evaluation_target=$3
    
    echo "Starting evaluation for $env_file with $evaluation_target"
    
    # HateSpeech Content Filter/Safety Evaluation
    DOTENV_PATH="$env_file" python main.py \
        --is_debug "$is_debug" \
        --model_provider "$model_provider" \
        --batch_size "$batch_size" \
        --max_tokens "$max_tokens" \
        --temperature "$temperature" \
        --evaluation_target "$evaluation_target" \
        --num_debug_samples "$num_debug_samples" \
        --use_multiprocessing True \
        --max_workers 2 &
    
    wait  # 해당 모델의 모든 작업이 완료될 때까지 대기
    echo "Completed evaluation for $env_file with $evaluation_target"
}

# 병렬 실행 관리
job_count=0
for env_file in "${env_files[@]}"; do
    if [[ "$env_file" == .env_gpt* ]]; then
        model_provider="azureopenai"
    else
        model_provider="azureopenai"  # 이 스크립트는 azureopenai만 지원
    fi
    
    # 각 evaluation_target에 대해 실행
    for evaluation_target in "${evaluation_targets[@]}"; do
        # 백그라운드에서 실행
        run_model "$env_file" "$model_provider" "$evaluation_target" &
        
        ((job_count++))
        
        # 최대 병렬 작업 수에 도달하면 일부 작업이 완료될 때까지 대기
        if (( job_count >= max_parallel_jobs )); then
            wait -n  # 하나의 작업이 완료될 때까지 대기
            ((job_count--))
        fi
    done
done

wait  # 모든 작업 완료 대기
echo "All evaluations completed!"
