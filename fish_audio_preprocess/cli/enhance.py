import torchaudio
import os
import time
import concurrent.futures
import multiprocessing
import click
from loguru import logger
from loguru import logger
import tqdm

input_dir = "test"
# semi_tone = 2

def process(input, semi_list):

    # 预加载音频
    waveform, sample_rate = torchaudio.load(input)

    for semi in semi_list:
        # 大概就是 nahida_+1_key
        output_dir = f"{input_dir}_{'+' if semi > 0 else ''}{semi}_key/"
        y = torchaudio.functional.pitch_shift(waveform, sample_rate, semi)
        torchaudio.save(output_dir+os.path.basename(input), y, sample_rate)
        
@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--semi-tone",
    default=8,
    help="Range of semi tone",
)
@click.option(
    "--num-workers",
    default=5,
    help="Number of workers for parallel processing",
)
def enhance(input_dir,semi_tone,num_workers):
    tasks = []
    semi_list = [*list(-i for i in range(1,semi_tone+1)),*range(1,semi_tone+1)]
    for semi in semi_list:
        # 大概就是 nahida_+1_key
        output_dir = f"{input_dir}_{'+' if semi > 0 else ''}{semi}_key/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 处理 example 文件夹里面的所有 wav 文件
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith(".wav"):
                    tasks.append(executor.submit(process, os.path.join(root, file))) 
                    
        logger.info("tasks:", len(tasks))
        for task in tqdm.tqdm(tasks):
            task.result()