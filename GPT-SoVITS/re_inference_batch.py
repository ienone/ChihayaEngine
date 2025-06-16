#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import json
import traceback
import logging
import warnings
from time import time as ttime
import random
import re # 用于正则表达式处理空格和标点

# --- 项目根目录设置 ---
try:
    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
except NameError:
    PROJECT_ROOT = os.getcwd()

sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "GPT_SoVITS")) # Ensure GPT_SoVITS modules are found

# 基本的日志和警告设置
logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)
logging.getLogger("multipart.multipart").setLevel(logging.ERROR)
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

import torch
import soundfile as sf

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config, NO_PROMPT_ERROR
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method as get_segmentation_method
from GPT_SoVITS.process_ckpt import get_sovits_version_from_path_fast

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
is_half = False
tts_pipeline: TTS = None

i18n_mock = lambda x: x

dict_language_v1_cli = {
    "中文": "all_zh", "英文": "en", "日文": "all_ja",
    "中英混合": "zh", "日英混合": "ja", "多语种混合": "auto",
    "all_zh": "all_zh", "en": "en", "all_ja": "all_ja", "zh": "zh", "ja": "ja", "auto": "auto"
}
dict_language_v2_cli = {
    **dict_language_v1_cli,
    "粤语": "all_yue", "韩文": "all_ko",
    "粤英混合": "yue", "韩英混合": "ko", "多语种混合(粤语)": "auto_yue",
    "all_yue": "all_yue", "all_ko": "all_ko", "yue": "yue", "ko": "ko", "auto_yue": "auto_yue"
}

cut_method_cli = {
    "不切": "cut0", "凑四句一切": "cut1", "凑50字一切": "cut2",
    "按中文句号切": "cut3", "按英文句号切": "cut4", "按标点符号切": "cut5",
    "cut0":"cut0", "cut1":"cut1", "cut2":"cut2", "cut3":"cut3", "cut4":"cut4", "cut5":"cut5"
}

# --- 文本预处理函数 (新) ---
def preprocess_text_file_for_batching(filepath, target_line_length=40):
    """
    读取包含单篇长文本的文件，去除空格，按标点切分，
    然后重新组合成每行约 target_line_length 字符的文本，并覆盖写回原文件。
    """
    print(f"开始预处理文本文件: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            original_text = f.read()
    except Exception as e:
        print(f"错误: 读取文件 {filepath} 失败: {e}")
        raise

    # 1. 去除所有空格 (包括全角和半角)
    text_no_spaces = re.sub(r'\s+', '', original_text)
    if not text_no_spaces:
        print("警告: 文本去除空格后为空。")
        with open(filepath, 'w', encoding='utf-8') as f: # 覆盖写入空内容
            f.write("")
        return

    # 2. 按标点符号切分成句子/短语列表
    # 使用正则表达式匹配中英文标点作为分隔符，并保留分隔符在切分后的句末
    # (?<=[。？！，、；：……「」『』《》．“”‘’﹃﹄〔〕〖〗【】（）〈〉〖〗])  -> 匹配这些标点后面的位置
    # |(?<=[.?!,;:()'"\-]) -> 或匹配这些英文标点后面的位置
    # 确保不误切小数点等
    #segments = re.split(r'(?<=[。？！；：……「」『』《》．“”‘’﹃﹄〔〕〖〗【】（）〈〉〖〗])|(?<=(?<!\d)\(?!\d)|[?!;:()\'"\-])', text_no_spaces)
    segments = re.split(r'(?<=[。？！；：……「」『』《》．“”‘’﹃﹄〔〕〖〗【】（）〈〉〖〗])|(?<=(?<!\d)\.(?!\d))|[?!;:()\'"\-]', text_no_spaces)    
    processed_segments = []
    temp_segment = ""
    for part in segments:
        if part is None: # re.split 可能会产生 None
            continue
        temp_segment += part
        if part and part[-1] in "。？！，、；：……．.?!,;:": # 如果这部分以标点结尾
            processed_segments.append(temp_segment.strip())
            temp_segment = ""
    if temp_segment.strip(): # 添加最后剩余的部分
        processed_segments.append(temp_segment.strip())

    if not processed_segments:
        print("警告: 按标点切分后没有有效文本段落。")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text_no_spaces) # 如果切分失败，则写回无空格的原文
        return
        
    # 3. 重新组合成每行约 target_line_length 字符
    output_lines = []
    current_line = ""
    for seg in processed_segments:
        if not seg: continue
        if not current_line:
            current_line = seg
        elif len(current_line) + len(seg) <= target_line_length + 10: # 允许一些余量以避免过多短行
            current_line += seg
        else:
            output_lines.append(current_line)
            current_line = seg
    if current_line: # 添加最后一行
        output_lines.append(current_line)

    # 4. 覆盖写回原文件
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            for line in output_lines:
                f.write(line + '\n')
        print(f"文本文件预处理完成，已覆盖保存至: {filepath}")
        print(f"  共生成 {len(output_lines)} 行。")
    except Exception as e:
        print(f"错误: 写回预处理后的文本到 {filepath} 失败: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="GPT-SoVITS 命令行批量推理工具 (基于 inference_webui_fast, 自动文本预处理)")
    # Model paths
    parser.add_argument("--gpt_model_path", type=str, required=True, help="GPT模型的.ckpt文件路径")
    parser.add_argument("--sovits_model_path", type=str, required=True, help="SoVITS模型的.pth文件路径")
    parser.add_argument("--bert_model_dir", type=str, required=True, help="BERT预训练模型目录路径")
    parser.add_argument("--ssl_model_dir", type=str, required=True, help="SSL (Hubert/ContentVec)预训练模型目录路径")
    
    # Reference info
    parser.add_argument("--ref_wav_path", type=str, required=True, help="参考音频的.wav文件路径")
    parser.add_argument("--prompt_text", type=str, default="", help="参考音频对应的文本 (可选)")
    parser.add_argument("--prompt_language", type=str, default="中文", help=f"参考文本的语种 (可选: {', '.join(list(dict_language_v2_cli.keys())[:len(dict_language_v2_cli)//2])} 或其代码)")
    parser.add_argument("--ref_text_free", action="store_true", help="开启无参考文本模式")

    # Target text input (single long text file, will be preprocessed)
    parser.add_argument("--text_path", type=str, required=True, help="包含待合成的单篇长文本的文件路径 (UTF-8编码, 将被预处理)")
    parser.add_argument("--text_language", type=str, default="中文", help=f"目标文本的语种 (可选: {', '.join(list(dict_language_v2_cli.keys())[:len(dict_language_v2_cli)//2])} 或其代码)")
    parser.add_argument("--target_line_length_after_preprocessing", type=int, default=40, help="文本预处理后每行的目标字符数")

    # Output (batch mode)
    parser.add_argument("--output_dir_path", type=str, required=True, help="合成后音频的保存目录")
    
    # Device and precision
    parser.add_argument("--gpu_id", type=str, default="0", help="使用的GPU ID (例如 '0', 'cpu')")
    parser.add_argument("--use_half_precision", action="store_true", help="是否使用半精度 (FP16) 进行推理")
    
    # Core TTS parameters
    parser.add_argument("--top_k", type=int, default=5, help="GPT推理的top_k参数")
    parser.add_argument("--top_p", type=float, default=1.0, help="GPT推理的top_p参数")
    parser.add_argument("--temperature", type=float, default=1.0, help="GPT推理的temperature参数")
    parser.add_argument("--text_split_method", type=str, default="凑四句一切", help=f"TTS pipeline内部文本切分方法 (可选: {', '.join(list(cut_method_cli.keys())[:len(cut_method_cli)//2])} 或其代码)")
    parser.add_argument("--batch_size", type=int, default=10, help="TTS pipeline内部的批处理大小")
    parser.add_argument("--speed", type=float, default=1.0, help="SoVITS推理的语速参数")
    parser.add_argument("--pause_between_chunks_s", type=float, default=0.3, help="TTS pipeline内部切分片段间的默认停顿")
    parser.add_argument("--seed", type=int, default=-1, help="随机种子 (-1表示随机)")
    parser.add_argument("--keep_random", action="store_true", help="如果seed为-1, 是否在每次生成时都保持随机")
    parser.add_argument("--parallel_infer", action="store_true", default=True, help="是否启用并行推理")
    parser.add_argument("--split_bucket", action="store_true", default=True, help="是否使用数据分桶")
    
    # V3/V4 specific parameters
    parser.add_argument("--repetition_penalty", type=float, default=1.35, help="重复惩罚 (V3/V4)")
    parser.add_argument("--sample_steps", type=int, default=30, help="采样步数 (V3/V4)")
    parser.add_argument("--super_sampling", action="store_true", help="音频超采样")

    args = parser.parse_args()

    global device, is_half, tts_pipeline
    print(f"项目根目录已确定为: {PROJECT_ROOT}")

    if args.gpu_id.lower() == "cpu" or not torch.cuda.is_available():
        device = "cpu"
        is_half = False
        print("使用CPU进行推理。")
    else:
        try:
            gpu_idx = int(args.gpu_id.split(",")[0])
            if not (0 <= gpu_idx < torch.cuda.device_count()):
                print(f"错误: GPU ID {gpu_idx} 无效。可用 GPUs: {list(range(torch.cuda.device_count()))}.")
                sys.exit(1)
            device = f"cuda:{gpu_idx}"
            is_half = args.use_half_precision
            print(f"使用 GPU:{gpu_idx} 并 {'启用半精度 (FP16)' if is_half else '使用全精度 (FP32)'}。")
        except ValueError:
            print(f"错误: 无效的GPU ID格式 '{args.gpu_id}'。请输入单个数字,例如 '0'。")
            sys.exit(1)

    paths_to_check = [args.gpt_model_path, args.sovits_model_path, args.bert_model_dir, args.ssl_model_dir, args.ref_wav_path, args.text_path]
    for p_orig in paths_to_check:
        p_abs = os.path.abspath(p_orig)
        if not os.path.exists(p_abs):
            print(f"错误: 输入路径 '{p_orig}' (解析为 '{p_abs}') 不存在。")
            sys.exit(1)
    
    os.makedirs(args.output_dir_path, exist_ok=True)

    # --- 新增：预处理输入文本文件 ---
    try:
        preprocess_text_file_for_batching(os.path.abspath(args.text_path), args.target_line_length_after_preprocessing)
    except Exception as e:
        print(f"文本文件预处理失败: {e}")
        traceback.print_exc()
        sys.exit(1)
    # --- 预处理结束 ---

    try:
        symbols_version_inferred, model_version_inferred, _ = get_sovits_version_from_path_fast(args.sovits_model_path)
        print(f"从SoVITS模型路径推断: 符号版本 '{symbols_version_inferred}', 模型架构版本 '{model_version_inferred}'")
    except Exception as e:
        print(f"错误: 无法从SoVITS模型路径推断版本: {e}")
        traceback.print_exc()
        sys.exit(1)

    tts_config = TTS_Config(os.path.join(PROJECT_ROOT, "GPT_SoVITS", "configs", "tts_infer.yaml"))
    tts_config.device = device
    tts_config.is_half = is_half
    tts_config.t2s_weights_path = args.gpt_model_path
    tts_config.vits_weights_path = args.sovits_model_path
    tts_config.bert_base_path = args.bert_model_dir
    tts_config.cnhuhbert_base_path = args.ssl_model_dir
    tts_config.version = model_version_inferred

    print("TTS 配置:")
    print(f"  Device: {tts_config.device}")
    # ... (其他配置打印可以保留)

    try:
        print("正在初始化TTS Pipeline...")
        tts_pipeline = TTS(tts_config)
        print("TTS Pipeline 初始化成功。")
    except Exception as e:
        print(f"TTS Pipeline 初始化失败: {e}")
        traceback.print_exc()
        sys.exit(1)

    current_dict_language = dict_language_v1_cli if symbols_version_inferred == "v1" else dict_language_v2_cli
    try:
        prompt_lang_code = current_dict_language.get(args.prompt_language, args.prompt_language)
        text_lang_code = current_dict_language.get(args.text_language, args.text_language)
        if prompt_lang_code not in current_dict_language.values():
             raise KeyError(f"无效的参考语言: {args.prompt_language}")
        if text_lang_code not in current_dict_language.values():
             raise KeyError(f"无效的目标语言: {args.text_language}")
    except KeyError as e:
        print(f"错误: {e}. 可用选项: {list(current_dict_language.keys())[:len(current_dict_language)//2]}")
        sys.exit(1)
    
    try:
        text_split_method_code = cut_method_cli.get(args.text_split_method, args.text_split_method)
        if text_split_method_code not in cut_method_cli.values():
            raise KeyError(f"无效的文本切分方法: {args.text_split_method}")
    except KeyError as e:
        print(f"错误: {e}. 可用选项: {list(cut_method_cli.keys())[:len(cut_method_cli)//2]}")
        sys.exit(1)

    texts_to_process = []
    try:
        # 读取的是预处理后的文件
        with open(os.path.abspath(args.text_path), 'r', encoding='utf-8') as f:
            for line in f:
                stripped_line = line.strip()
                if stripped_line:
                    texts_to_process.append(stripped_line)
        if not texts_to_process:
            print(f"警告: (预处理后的) 输入文本文件 '{args.text_path}' 为空或只包含空行。")
            sys.exit(0)
    except Exception as e:
        print(f"读取 (预处理后的) 目标文本文件 '{args.text_path}' 时发生错误: {e}")
        sys.exit(1)

    total_texts = len(texts_to_process)
    print(f"共找到 {total_texts} 行文本待处理 (来自预处理后的文件)。")

    for i, text_line in enumerate(texts_to_process):
        print(f"\n--- 正在处理文本行 {i+1}/{total_texts} ---")
        print(f"输入文本: \"{text_line}\"")
        t_start_line = ttime()

        current_seed = args.seed
        if args.keep_random and args.seed == -1:
            current_seed = random.randint(0, 2**32 - 1)
        elif args.seed == -1:
             current_seed = random.randint(0, 2**32 - 1)

        inputs = {
            "text": text_line,
            "text_lang": text_lang_code,
            "ref_audio_path": args.ref_wav_path,
            "aux_ref_audio_paths": [],
            "prompt_text": args.prompt_text if not args.ref_text_free else "",
            "prompt_lang": prompt_lang_code,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
            "text_split_method": text_split_method_code, # 这是 TTS pipeline 内部的切分
            "batch_size": args.batch_size,
            "speed_factor": args.speed,
            "split_bucket": args.split_bucket,
            "return_fragment": False,
            "fragment_interval": args.pause_between_chunks_s,
            "seed": current_seed,
            "parallel_infer": args.parallel_infer,
            "repetition_penalty": args.repetition_penalty,
            "sample_steps": args.sample_steps,
            "super_sampling": args.super_sampling,
        }

        try:
            output_audio_data = None
            output_sampling_rate = None
            
            for sr, audio_data in tts_pipeline.run(inputs):
                output_sampling_rate = sr
                output_audio_data = audio_data 
            
            if output_audio_data is not None and output_sampling_rate is not None:
                output_filename = f"{i:04d}_output.wav" # 按行号命名
                output_path = os.path.join(args.output_dir_path, output_filename)
                sf.write(output_path, output_audio_data, output_sampling_rate)
                t_end_line = ttime()
                print(f"音频已保存至: {output_path}, 耗时: {t_end_line - t_start_line:.3f} 秒")
            else:
                print(f"警告: 未能为文本 \"{text_line}\" 生成音频数据。")

        except NO_PROMPT_ERROR as e:
            print(f"错误 (NO_PROMPT_ERROR): {e} - 对于文本 \"{text_line}\"")
        except Exception as e:
            print(f"处理文本 \"{text_line}\" 时发生错误: {e}")
            traceback.print_exc()

    print("\n所有文本处理完毕。")

if __name__ == "__main__":
    main()