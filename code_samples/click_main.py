import os
import json
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

import openai
from openai import RateLimitError
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import load_dataset

from prompts import TYPE_1, TYPE_2, TYPE_3, TYPE_4
from util.custom_parser import MultipleChoicesFiveParser

from util.common_helper import (
    str2bool,
    format_timespan,
    get_prompt_template,
    get_llm_client,
)

from logger import logger


def get_prompt(x) -> str:
    num_choices = len(x["choices"])
    if num_choices == 4:
        if x["paragraph"] != "":  # Use Type 1 Prompt
            return TYPE_1.format(
                CONTEXT=x["paragraph"],
                QUESTION=x["question"],
                A=x["choices"][0],
                B=x["choices"][1],
                C=x["choices"][2],
                D=x["choices"][3],
            )
        else:
            return TYPE_2.format(
                QUESTION=x["question"],
                A=x["choices"][0],
                B=x["choices"][1],
                C=x["choices"][2],
                D=x["choices"][3],
            )
    elif num_choices == 5:
        if x["paragraph"] != "":
            return TYPE_3.format(
                CONTEXT=x["paragraph"],
                QUESTION=x["question"],
                A=x["choices"][0],
                B=x["choices"][1],
                C=x["choices"][2],
                D=x["choices"][3],
                E=x["choices"][4],
            )
        else:
            return TYPE_4.format(
                QUESTION=x["question"],
                A=x["choices"][0],
                B=x["choices"][1],
                C=x["choices"][2],
                D=x["choices"][3],
                E=x["choices"][4],
            )
    else:
        raise ValueError(f"Invalid number of choices: {num_choices} (ID: {x['id']})")


def get_answer(x) -> str:
    answer_idx = [xx.strip() for xx in x["choices"]].index(x["answer"].strip())
    if answer_idx == -1:
        raise ValueError(f"Answer not found in choices: {x['answer']} (ID: {x['id']})")
    return chr(0x41 + answer_idx)  # answer_idx = 0 -> answer = "A"


def get_category_from_id(item_id):
    """ID에서 카테고리 추출"""
    with open("id_to_category.json", "r") as json_file:
        id_to_category = json.load(json_file)
    return id_to_category.get(str(item_id), "Unknown")


def load_existing_results(csv_path):
    """기존 결과 파일이 있으면 로드"""
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                logger.info(f"Found existing results: {len(df)} records in {csv_path}")
                return df
        except Exception as e:
            logger.warning(f"Error loading existing results: {e}")
    return pd.DataFrame()


def _save_results_safely(responses, csv_path):
    """실제 저장 로직"""
    df_new = pd.DataFrame(responses)
    
    # 파일 경로를 절대 경로로 변환
    abs_csv_path = os.path.abspath(csv_path)
    
    # 기존 파일이 있으면 로드하고 합치기
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        # ID 기준으로 중복 제거
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=['id'], keep='last')
        
        logger.info(f"📁 Updated existing file: {abs_csv_path}")
        logger.info(f"   - Added {len(df_new)} new records")
        logger.info(f"   - Total records after merge: {len(df_combined)}")
    else:
        df_combined = df_new
        logger.info(f"📁 Created new file: {abs_csv_path}")
        logger.info(f"   - Initial records: {len(df_combined)}")
    
    # 디렉토리 생성 확인
    os.makedirs(os.path.dirname(abs_csv_path), exist_ok=True)
    
    # 파일 저장
    df_combined.to_csv(csv_path, index=False)
    
    # 파일 크기 확인
    file_size = os.path.getsize(csv_path)
    file_size_mb = file_size / (1024 * 1024)
    
    logger.info(f"✅ Successfully saved CSV file:")
    logger.info(f"   - File path: {abs_csv_path}")
    logger.info(f"   - File size: {file_size:,} bytes ({file_size_mb:.2f} MB)")
    
    # 카테고리별 분포 로깅 (CLIcK의 경우)
    if 'id' in df_combined.columns:
        try:
            with open("id_to_category.json", "r") as json_file:
                id_to_category = json.load(json_file)
            df_combined["category"] = df_combined["id"].astype(str).map(id_to_category)
            category_counts = df_combined['category'].value_counts()
            logger.info(f"   - Categories saved: {list(category_counts.index)}")
            logger.info(f"   - Records per category: {dict(category_counts)}")
        except Exception as e:
            logger.warning(f"Could not analyze category distribution: {e}")


def save_results_incremental(responses, csv_path, lock=None):
    """결과를 점진적으로 저장 (멀티프로세싱 안전)"""
    try:
        if lock:
            with lock:
                _save_results_safely(responses, csv_path)
        else:
            _save_results_safely(responses, csv_path)
    except Exception as e:
        logger.error(f"❌ Error saving results to {os.path.abspath(csv_path)}: {e}")
        raise


def get_completed_categories(csv_path, min_records=10):
    """완료된 카테고리 목록 반환"""
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                # ID를 통해 카테고리 매핑
                with open("id_to_category.json", "r") as json_file:
                    id_to_category = json.load(json_file)
                
                df["category"] = df["id"].astype(str).map(id_to_category)
                category_counts = df['category'].value_counts()
                completed = []
                for category, count in category_counts.items():
                    logger.info(f"Found category {category} with {count} records")
                    if count >= min_records:
                        completed.append(category)
                logger.info(f"Completed categories: {completed}")
                return completed
        except Exception as e:
            logger.warning(f"Error reading completed categories: {e}")
    return []


def get_category_data_ranges(click_ds):
    """카테고리별 데이터 범위 계산"""
    with open("id_to_category.json", "r") as json_file:
        id_to_category = json.load(json_file)
    
    category_ranges = {}
    current_category = None
    start_idx = 0
    
    for idx, item in enumerate(click_ds):
        item_category = id_to_category.get(str(item["id"]), "Unknown")
        
        if current_category != item_category:
            if current_category is not None:
                category_ranges[current_category] = (start_idx, idx)
            current_category = item_category
            start_idx = idx
    
    # 마지막 카테고리 처리
    if current_category is not None:
        category_ranges[current_category] = (start_idx, len(click_ds))
    
    return category_ranges


def process_batch_streaming(batch_data, model_config, template_type="basic"):
    """스트리밍 방식으로 배치 처리"""
    try:
        # 각 프로세스에서 별도의 LLM 클라이언트 생성
        llm, _ = get_llm_client(
            model_config['provider'], 
            model_config.get('hf_model_id', 'microsoft/Phi-3.5-mini-instruct'),
            model_config['temperature'], 
            model_config['max_tokens'], 
            model_config['max_retries']
        )
        
        prompt_template = get_prompt_template(template_type)
        chain = prompt_template | llm | MultipleChoicesFiveParser()
        
        results = []
        batch_size = model_config['batch_size']
        max_retries = model_config['max_retries']
        
        for i in range(0, len(batch_data), batch_size):
            mini_batch = batch_data[i:i + batch_size]
            retries = 0
            
            while retries <= max_retries:
                try:
                    preds = chain.batch(mini_batch, {"max_concurrency": batch_size})
                    
                    for qna, pred in zip(mini_batch, preds):
                        results.append({
                            "id": qna["id"],
                            "trial": 0,
                            "answer": qna["answer"],
                            "pred": pred[0],
                            "response": pred[1],
                        })
                    break
                    
                except RateLimitError as e:
                    delay = (retries + 1) * 30
                    logger.warning(f"Rate limit error, retrying in {delay} seconds...")
                    time.sleep(delay)
                    retries += 1
                    
                    if retries > max_retries:
                        logger.error(f"Max retries reached for batch")
                        for qna in mini_batch:
                            results.append({
                                "id": qna["id"],
                                "trial": 0,
                                "answer": qna["answer"],
                                "pred": "FAILED",
                                "response": "RATE_LIMIT_ERROR",
                            })
                        break
                        
                except openai.BadRequestError as e:
                    logger.error(f"BadRequestError: {e}. Adding failed responses for this batch.")
                    logger.info(f"Question sample: {batch_data[i]['question'][:100]}..." if batch_data else "No question data")
                    # 실패한 질문들에 대해 기본값으로 추가
                    for qna in mini_batch:
                        results.append({
                            "id": qna["id"],
                            "trial": 0,
                            "answer": qna["answer"],
                            "pred": "FAILED",
                            "response": "BAD_REQUEST_ERROR",
                        })
                    break
                        
                except KeyError as e:
                    # 핵심 'choices' KeyError 처리 추가
                    if "'choices'" in str(e):
                        logger.warning(f"OpenAI API response missing 'choices' field - processing individually")
                        # 개별 처리로 fallback
                        for qna in mini_batch:
                            try:
                                pred = chain.invoke(qna["question"])
                                results.append({
                                    "id": qna["id"],
                                    "trial": 0,
                                    "answer": qna["answer"],
                                    "pred": pred[0],
                                    "response": pred[1],
                                })
                            except Exception as individual_error:
                                logger.error(f"Individual processing failed: {individual_error}")
                                results.append({
                                    "id": qna["id"],
                                    "trial": 0,
                                    "answer": qna["answer"],
                                    "pred": "FAILED",
                                    "response": f"ERROR: {str(individual_error)}",
                                })
                        break
                    else:
                        # 다른 KeyError는 일반 처리
                        logger.error(f"KeyError in batch processing: {e}")
                        retries += 1
                        if retries > max_retries:
                            for qna in mini_batch:
                                results.append({
                                    "id": qna["id"],
                                    "trial": 0,
                                    "answer": qna["answer"],
                                    "pred": "FAILED",
                                    "response": f"KEYERROR: {str(e)}",
                                })
                            break
                        time.sleep(2 ** retries)
                        
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    retries += 1
                    if retries > max_retries:
                        for qna in mini_batch:
                            results.append({
                                "id": qna["id"],
                                "trial": 0,
                                "answer": qna["answer"],
                                "pred": "FAILED",
                                "response": f"ERROR: {str(e)}",
                            })
                        break
                    time.sleep(2 ** retries)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in process_batch_streaming: {e}")
        return []


def process_category_streaming(category_info):
    """단일 카테고리를 스트리밍 방식으로 처리"""
    category, data_range, click_ds, model_config, is_debug, num_debug_samples, template_type, csv_path = category_info
    
    logger.info(f"Processing category {category} in process {os.getpid()}")
    
    try:
        start_idx, end_idx = data_range
        category_ds = click_ds.select(range(start_idx, end_idx))
        
        if is_debug:
            category_ds = category_ds.select(range(min(num_debug_samples, len(category_ds))))
        
        # 스트리밍 처리를 위한 청크 크기 설정
        chunk_size = model_config['batch_size'] * 10
        total_items = len(category_ds)
        category_responses = []
        
        logger.info(f"Processing {total_items} items for category {category} in chunks of {chunk_size}")
        
        # 청크별 진행률 표시 추가
        total_chunks = (total_items + chunk_size - 1) // chunk_size
        with tqdm(total=total_chunks, desc=f"Processing {category}", position=0, leave=True) as pbar:
            # 청크별로 스트리밍 처리
            for start_chunk in range(0, total_items, chunk_size):
                end_chunk = min(start_chunk + chunk_size, total_items)
                chunk_ds = category_ds.select(range(start_chunk, end_chunk))
                
                # 청크를 배치 데이터로 변환
                chunk_batch = [
                    {
                        "id": x["id"], 
                        "question": get_prompt(x), 
                        "answer": get_answer(x)
                    }
                    for x in chunk_ds
                ]
                
                # 청크 처리
                chunk_results = process_batch_streaming(chunk_batch, model_config, template_type)
                category_responses.extend(chunk_results)
                
                # 진행률 업데이트
                pbar.update(1)
                pbar.set_postfix({
                    'items': f"{len(category_responses)}/{total_items}",
                    'chunk_size': len(chunk_results)
                })
                
                # 중간 저장 (메모리 절약)
                if len(category_responses) >= 1000:
                    save_results_incremental(category_responses, csv_path)
                    logger.info(f"Intermediate save for category {category}: {len(category_responses)} items")
                    category_responses = []
        
        # 마지막 배치 저장
        if category_responses:
            save_results_incremental(category_responses, csv_path)
        
        logger.info(f"Completed category {category}")
        return category, "completed"
        
    except Exception as e:
        logger.error(f"Error processing category {category}: {e}")
        return category, f"error: {str(e)}"


def benchmark_multiprocess(args):
    """멀티프로세싱을 사용한 벤치마크 실행"""
    
    is_debug = args.is_debug
    
    # 모델 설정
    model_name = os.getenv("MODEL_NAME", "gpt-5-mini")
    model_version = os.getenv("MODEL_VERSION", "2025-08-08")
    
    model_config = {
        'provider': args.model_provider,
        'hf_model_id': args.hf_model_id,
        'batch_size': args.batch_size,
        'max_tokens': args.max_tokens,
        'temperature': args.temperature,
        'max_retries': args.max_retries,
    }
    
    # CSV 파일 경로 설정
    os.makedirs("results", exist_ok=True)
    csv_path = f"results/[CLIcK] {model_name}-{model_version}.csv"
    abs_csv_path = os.path.abspath(csv_path)
    
    logger.info(f"🎯 Target output file: {abs_csv_path}")
    
    # 기존 파일 상태 확인
    if os.path.exists(csv_path):
        file_size = os.path.getsize(csv_path)
        existing_df = pd.read_csv(csv_path)
        logger.info(f"📋 Found existing file with {len(existing_df)} records ({file_size:,} bytes)")
    else:
        logger.info(f"📋 No existing file found - will create new file")
    
    # 데이터셋 로드
    click_ds = load_dataset("EunsuKim/CLIcK")["train"]
    
    if is_debug:
        click_ds = click_ds.select(range(args.num_debug_samples))
    
    # 카테고리별 데이터 범위 계산
    category_ranges = get_category_data_ranges(click_ds)
    
    # 모든 CLIcK 카테고리 목록
    all_click_categories = list(category_ranges.keys())
    
    # 실행할 카테고리 결정
    if args.categories:
        # 사용자가 지정한 카테고리들 검증
        invalid_categories = [c for c in args.categories if c not in all_click_categories]
        if invalid_categories:
            logger.error(f"Invalid categories specified: {invalid_categories}")
            logger.error(f"Available categories: {all_click_categories}")
            return
        selected_categories = args.categories
        logger.info(f"🎯 Processing user-specified categories: {selected_categories}")
        
        # 선택된 카테고리들만 포함하는 새로운 category_ranges 생성
        filtered_category_ranges = {cat: category_ranges[cat] for cat in selected_categories}
        category_ranges = filtered_category_ranges
    else:
        selected_categories = all_click_categories
        logger.info(f"🎯 Processing all categories: {selected_categories}")
    
    # 완료된 카테고리 확인
    completed_categories = get_completed_categories(csv_path)
    
    # 남은 카테고리 필터링 (start_category 로직 제거)
    remaining_categories = [c for c in selected_categories if c not in completed_categories]
    
    if not remaining_categories:
        logger.info("All specified categories already completed!")
        return
    
    logger.info(f"Processing {len(remaining_categories)} remaining categories: {remaining_categories}")
    logger.info(f"Using multiprocessing with {args.max_workers} workers")
    
    # 멀티프로세싱 작업 준비
    category_tasks = [
        (category, category_ranges[category], click_ds, model_config, 
         is_debug, args.num_debug_samples, args.template_type, csv_path)
        for category in remaining_categories
    ]
    
    start_time = time.time()
    
    # 멀티프로세싱 실행 - tqdm 개선
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_category = {
            executor.submit(process_category_streaming, task): task[0] 
            for task in category_tasks
        }
        
        completed_count = 0
        # 진행률 표시 개선
        with tqdm(total=len(remaining_categories), desc="Categories", position=1, leave=True) as category_pbar:
            for future in future_to_category:
                category = future_to_category[future]
                try:
                    result_category, status = future.result()
                    completed_count += 1
                    logger.info(f"Category {result_category} completed: {status} ({completed_count}/{len(remaining_categories)})")
                    
                    # 카테고리 진행률 업데이트
                    category_pbar.update(1)
                    category_pbar.set_postfix({
                        'current': result_category,
                        'status': status
                    })
                    
                except Exception as e:
                    logger.error(f"Category {category} failed with exception: {e}")
                    completed_count += 1
                    category_pbar.update(1)
                    category_pbar.set_postfix({
                        'current': category,
                        'status': 'failed'
                    })

    end_time = time.time()
    total_time = format_timespan(end_time - start_time)
    
    logger.info(f"====== [DONE] All specified categories processed in {total_time} =====")
    
    # 최종 파일 상태 확인
    if os.path.exists(csv_path):
        final_df = pd.read_csv(csv_path)
        final_file_size = os.path.getsize(csv_path)
        logger.info(f"🏁 Final output file status:")
        logger.info(f"   - Path: {abs_csv_path}")
        logger.info(f"   - Records: {len(final_df)}")
        logger.info(f"   - Size: {final_file_size:,} bytes ({final_file_size/(1024*1024):.2f} MB)")
    else:
        logger.error(f"❌ Final output file not found: {abs_csv_path}")
    
    # 최종 평가
    logger.info(f"====== [START] Final Evaluation - CSV_PATH: {csv_path} =====")
    evaluate(csv_path)
    logger.info(f"====== [END] Evaluation completed =====")


def benchmark_sequential(args):
    """기존 순차 처리 방식"""
    is_debug = args.is_debug
    max_retries = args.max_retries
    delay_increment = 30

    num_debug_samples = args.num_debug_samples
    batch_size = args.batch_size
    max_tokens = args.max_tokens
    temperature = args.temperature
    llm, model_name = get_llm_client(
        args.model_provider, args.hf_model_id, temperature, max_tokens, max_retries
    )
    model_version = (
        os.getenv("MODEL_VERSION")
        if args.model_provider == "azureopenai"
        else None
    )

    click_ds = load_dataset("EunsuKim/CLIcK")["train"]

    if is_debug:
        click_ds = click_ds.select(range(num_debug_samples))

    all_batch = [
        {"id": x["id"], "question": get_prompt(x), "answer": get_answer(x)}
        for x in tqdm(click_ds)
    ]
    responses = []
    prompt_template = get_prompt_template(args.template_type)
    chain = prompt_template | llm | MultipleChoicesFiveParser()

    logger.info(f"====== [START] Generate answers to questions given by LLM. =====")
    logger.info(
        f"====== deployment name: {model_name}, model version: {model_version} ====="
    )
    t0 = time.time()

    with tqdm(total=len(all_batch), desc="Processing Answers") as pbar:

        for i in range(0, len(all_batch), batch_size):
            mini_batch = all_batch[i : i + batch_size]
            retries = 0

            while retries <= max_retries:
                try:
                    preds = chain.batch(mini_batch, {"max_concurrency": batch_size})
                    # If no exception, add questions and answers to all_answers
                    
                    for qna, pred in zip(mini_batch, preds):
                        responses.append(
                            {
                                "id": qna["id"],
                                "trial": 0,
                                "answer": qna["answer"],
                                "pred": pred[0],
                                "response": pred[1],
                            }
                        )
                    break  # Exit the retry loop once successful
                except RateLimitError as rate_limit_error:
                    delay = (retries + 1) * delay_increment
                    logger.warning(
                        f"{rate_limit_error}. Retrying in {delay} seconds..."
                    )
                    time.sleep(delay)
                    retries += 1

                    if retries > max_retries:
                        logger.error(
                            f"Max retries reached this batch. Adding failed responses for this batch."
                        )
                        # 실패한 질문들에 대해 기본값으로 추가
                        for qna in mini_batch:
                            responses.append(
                                {
                                    "id": qna["id"],
                                    "trial": 0,
                                    "answer": qna["answer"],
                                    "pred": "FAILED",
                                    "response": "RATE_LIMIT_ERROR",
                                }
                            )
                        break
                except openai.BadRequestError as e:
                    logger.error(f"BadRequestError: {e}. Adding failed responses for this batch.")
                    logger.info(f"Question sample: {mini_batch[0]['question'][:100]}...")
                    # 실패한 질문들에 대해 기본값으로 추가
                    for qna in mini_batch:
                        responses.append(
                            {
                                "id": qna["id"],
                                "trial": 0,
                                "answer": qna["answer"],
                                "pred": "FAILED",
                                "response": "BAD_REQUEST_ERROR",
                            }
                        )
                    break
                except Exception as e:
                    logger.error(f"Error in process_inputs: {e}. Adding failed responses for this batch.")
                    # 실패한 질문들에 대해 기본값으로 추가
                    for qna in mini_batch:
                        responses.append(
                            {
                                "id": qna["id"],
                                "trial": 0,
                                "answer": qna["answer"],
                                "pred": "FAILED",
                                "response": f"ERROR: {str(e)}",
                            }
                        )
                    break

            pbar.set_postfix(
                {
                    "current_batch": f"{i//batch_size + 1}/{(len(all_batch) + (batch_size-1))//batch_size}"
                }
            )
            pbar.update(len(mini_batch))

    t1 = time.time()
    timespan = format_timespan(t1 - t0)
    logger.info(f"===== [DONE] Generating Answer dataset took {timespan}")

    if not responses:
        logger.error("No successful responses were generated. Skipping evaluation.")
        return

    df = pd.DataFrame(responses)
    os.makedirs("results", exist_ok=True)
    csv_path = f"results/[CLIcK] {model_name}-{model_version}.csv"
    abs_csv_path = os.path.abspath(csv_path)
    
    df.to_csv(csv_path, index=False)
    
    file_size = os.path.getsize(csv_path)
    logger.info(f"✅ Successfully saved CSV file:")
    logger.info(f"   - File path: {abs_csv_path}")
    logger.info(f"   - Records: {len(df)}")
    logger.info(f"   - File size: {file_size:,} bytes ({file_size/(1024*1024):.2f} MB)")
    
    logger.info(f"====== [START] Evaluation start - CSV_PATH: {csv_path} =====")
    evaluate(csv_path)
    logger.info(f"====== [START] Evaluation end =====")


def evaluate(csv_path):
    abs_csv_path = os.path.abspath(csv_path)
    
    if not os.path.exists(csv_path):
        logger.error(f"❌ CSV file does not exist: {abs_csv_path}")
        return
    
    logger.info(f"📊 Starting evaluation of: {abs_csv_path}")
    
    result = pd.read_csv(csv_path)
    if result.empty:
        logger.error(f"❌ CSV file is empty: {abs_csv_path}")
        return
    
    logger.info(f"📊 Loaded {len(result)} records for evaluation")
    
    # FAILED 응답 필터링 및 로깅
    original_count = len(result)
    failed_count = len(result[result["pred"] == "FAILED"])
    if failed_count > 0:
        logger.warning(f"Found {failed_count} FAILED responses out of {original_count} total responses")
        logger.info(f"Excluding FAILED responses from accuracy calculation")
        result = result[result["pred"] != "FAILED"]
        logger.info(f"Evaluating on {len(result)} valid responses")
    
    with open("id_to_category.json", "r") as json_file:
        id_to_category = json.load(json_file)

    result["category"] = result["id"].astype(str).map(id_to_category)
    
    # 매핑되지 않은 ID들 확인 및 제거
    missing_ids = result[result["category"].isna()]["id"].unique()
    if len(missing_ids) > 0:
        logger.warning(f"Found IDs without category mapping: {missing_ids[:10]}...")
        logger.warning(f"Total missing IDs: {len(missing_ids)}")
        result = result.dropna(subset=["category"])
    
    result["correct"] = result["answer"] == result["pred"]
    result["category_big"] = result["category"].apply(
        lambda x: (
            "Culture"
            if x
            in [
                "Economy",
                "Geography",
                "History",
                "Law",
                "Politics",
                "Popular",
                "Society",
                "Tradition",
                "Pop Culture",
            ]
            else ("Language" if x in ["Functional", "Textual", "Grammar"] else "Other")
        )
    )

    category_avg = (
        result.groupby(["category_big", "category"])
        .agg(correct_mean=("correct", "mean"), correct_count=("correct", "size"))
        .reset_index()
    )
    print(category_avg)

    category_big_avg = (
        result.groupby("category_big")
        .agg(correct_mean=("correct", "mean"), correct_count=("correct", "size"))
        .reset_index()
    )
    print(category_big_avg)

    # 전체 평균 계산
    overall_avg = result["correct"].mean()
    print(f"Overall Average: {overall_avg}")

    os.makedirs("evals", exist_ok=True)
    filename = csv_path.split("/")[-1].split(".")[0]
    
    eval_file1 = f"evals/{filename}-eval.csv"
    eval_file2 = f"evals/{filename}-eval-avg.csv"
    
    category_avg.to_csv(eval_file1, index=False)
    category_big_avg.to_csv(eval_file2, index=False)
    
    abs_eval_file1 = os.path.abspath(eval_file1)
    abs_eval_file2 = os.path.abspath(eval_file2)
    
    logger.info(f"✅ Evaluation results saved:")
    logger.info(f"   - Detailed results: {abs_eval_file1}")
    logger.info(f"   - Summary results: {abs_eval_file2}")


if __name__ == "__main__":
    dotenv_path = os.getenv('DOTENV_PATH', '.env')
    load_dotenv(dotenv_path, override=True)
   
    parser = argparse.ArgumentParser(description="CLIcK Benchmark with Multiprocessing and Streaming")
    parser.add_argument("--is_debug", type=str2bool, default=True)
    parser.add_argument("--num_debug_samples", type=int, default=20)
    parser.add_argument("--model_provider", type=str, default="azureopenai")
    parser.add_argument("--hf_model_id", type=str, default="microsoft/Phi-3.5-MoE-instruct")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--template_type", type=str, default="basic")
    
    # 특정 카테고리들만 실행하기 위한 인수
    parser.add_argument(
        "--categories", 
        type=str, 
        nargs='*', 
        default=None, 
        help='Specific categories to process (e.g., --categories "Economy" "Geography")'
    )
    
    # 새로운 멀티프로세싱 관련 인수
    parser.add_argument("--use_multiprocessing", type=str2bool, default=True, help="Enable multiprocessing")
    parser.add_argument("--max_workers", type=int, default=3, help="Maximum number of worker processes")
    parser.add_argument("--streaming_chunk_size", type=int, default=1000, help="Chunk size for streaming processing")

    args = parser.parse_args()
    valid_providers = ["azureopenai", "openai", "azureml", "azureai", "huggingface"]
    assert (
        args.model_provider in valid_providers
    ), f"Invalid 'model_provider' value. Please choose from {valid_providers}."

    valid_template_types = ["basic", "chat", "gpt5"]
    assert (
        args.template_type in valid_template_types
    ), f"Invalid 'template_type' value. Please choose from {valid_template_types}."

    # 카테고리 인수 로깅
    if args.categories:
        logger.info(f"🎯 User specified categories: {args.categories}")
    else:
        logger.info(f"🎯 Will process all available categories")

    logger.info(args)
    
    # 멀티프로세싱 사용 여부에 따라 실행 방식 선택
    if args.use_multiprocessing and args.max_workers > 1:
        benchmark_multiprocess(args)
    else:
        benchmark_sequential(args)