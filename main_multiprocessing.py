import os
import json
import random
import time
import argparse

import openai
from openai import AzureOpenAI
from openai import RateLimitError
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import Dataset, load_dataset
from typing import Tuple

from logger import logger
import multiprocessing

from main import benchmark

    


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--is_random", type=bool, default=False)
    parser.add_argument("--is_debug", type=bool, default=False)
    parser.add_argument("--num_debug_samples", type=int, default=15)
    parser.add_argument("--model_provider", type=str, default="azureopenai")
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0)
    
    args = parser.parse_args()
    valid_providers = ["azureopenai"] # azureopenai supports contentfilter
    assert args.model_provider in valid_providers, f"azureopenai only supports contentfilter. Please choose from {valid_providers}."

    logger.info(args)
    
    # Create a multiprocessing pool, check your CPU cores and set the number of processes accordingly
    pool = multiprocessing.Pool(processes=8)
    
    # Run the benchmark function in parallel using the multiprocessing pool
    pool.apply(benchmark, (args,))
    
    # Close the multiprocessing pool
    pool.close()
    pool.join()
