#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import os
import sys
import re
import json
import traceback
import logging
import warnings
from time import time as ttime

# --- 项目根目录设置 ---
try:
    # 当脚本放在项目根目录时:
    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
except NameError:
    # 如果直接在解释器中运行或打包后，__file__ 可能不存在
    PROJECT_ROOT = os.getcwd()
    
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
import torchaudio
import librosa
import numpy as np
import soundfile as sf # 用于保存音频

from GPT_SoVITS.text.LangSegmenter import LangSegmenter # 用于自动语言分割
from GPT_SoVITS.text import cleaned_text_to_sequence, chinese # 确保chinese模块被导入
from GPT_SoVITS.text.cleaner import clean_text
from transformers import AutoModelForMaskedLM, AutoTokenizer

# 假设这些模块在PYTHONPATH中，或者与脚本在同一项目结构下
from GPT_SoVITS.feature_extractor import cnhubert
from GPT_SoVITS.module.models import SynthesizerTrn, SynthesizerTrnV3, Generator
from GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule
from GPT_SoVITS.process_ckpt import get_sovits_version_from_path_fast, load_sovits_new # 关键的辅助函数
from peft import LoraConfig, get_peft_model
from GPT_SoVITS.module.mel_processing import mel_spectrogram_torch, spectrogram_torch


# --- 全局变量，用于存储模型和配置 ---
if torch.cuda.is_available():
    device = "cuda"  # 使用 CUDA 设备
else:
    device = "cpu"   # 使用 CPU 设备
is_half = False      # 是否使用半精度浮点数 (FP16)
tokenizer = None     # BERT 分词器
bert_model = None    # BERT 模型
ssl_model = None     # SSL (如 Hubert) 模型
vq_model = None      # SoVITS 声码器/合成器模型
t2s_model = None     # GPT (Text-to-Semantic) 模型
hps = None           # SoVITS 模型配置 (来自 checkpoint)
gpt_config = None    # GPT 模型配置 (来自 checkpoint)
symbols_version = None # 符号版本 (例如 "v1", "v2") - 从SoVITS模型推断
sovits_model_version = None # SoVITS模型架构版本 (例如 "v1", "v2", "v3", "v4")
if_lora_v3v4 = False    # SoVITS v3/v4 是否为LoRA模型
hz = 50                 # GPT 推理中的参数，通常固定为50
max_sec = 0             # GPT 推理中生成语义的最大时长 (来自GPT配置)

# 声码器模型 (针对 SoVITS v3/v4)
bigvgan_model = None # BigVGAN 声码器 (v3)
hifigan_model = None # HiFiGAN 声码器 (v4)

# 动态创建的梅尔频谱函数
# 将在加载 SoVITS 模型后根据其版本和配置动态设置
mel_fn_dynamic = None

# 标点符号集 (来自 inference_webui.py)
splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", '\n'} 
# 用于过滤的标点符号 (来自 inference_webui.py)
punctuation = set(["!", "?", "…", ",", ".", "-", " ", "\n"]) 


# 辅助类：递归地将字典转换为属性访问对象
class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"属性 {item} 未找到")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"属性 {item} 未找到")

# 音频重采样函数及其缓存字典
# 音频重采样函数及其缓存字典
resample_transform_dict = {}
def resample(audio_tensor, sr0, sr1):
    """对音频张量进行重采样"""
    global resample_transform_dict, device, is_half # 添加 is_half

    # 确保重采样操作在 float32 上进行，以避免类型不匹配错误
    input_dtype = audio_tensor.dtype
    if input_dtype == torch.float16:
        # print(f"  [调试 resample] 输入为 float16, 临时转换为 float32 进行重采样。")
        audio_tensor_float32 = audio_tensor.float() 
    else:
        audio_tensor_float32 = audio_tensor

    key="%s-%s"%(sr0,sr1)
    if key not in resample_transform_dict:
        # Resample 变换本身会根据输入张量移动到相应设备，但其内部核的类型是固定的 (float32)
        resample_transform_dict[key] = torchaudio.transforms.Resample(sr0, sr1).to(device)
    
    resampled_audio = resample_transform_dict[key](audio_tensor_float32)

    # 如果原始输入是 float16 并且我们启用了半精度，则将结果转换回 float16
    if input_dtype == torch.float16 and is_half:
        # print(f"  [调试 resample] 重采样完成，将结果转换回 float16。")
        return resampled_audio.half()
    else:
        return resampled_audio

# 加载BERT分词器、BERT模型和SSL模型
def load_tokenizer_bert_ssl(bert_model_dir_abs, ssl_model_dir_abs):
    global tokenizer, bert_model, ssl_model, device, is_half
    
    print(f"从 {bert_model_dir_abs} 加载BERT模型...")
    tokenizer = AutoTokenizer.from_pretrained(bert_model_dir_abs)
    bert_model = AutoModelForMaskedLM.from_pretrained(bert_model_dir_abs)
    if is_half:
        bert_model = bert_model.half().to(device)
    else:
        bert_model = bert_model.to(device)
    bert_model.eval() # 设置为评估模式
    print("BERT模型加载成功。")

    print(f"从 {ssl_model_dir_abs} 加载SSL模型 (Hubert/ContentVec)...")
    cnhubert.cnhubert_base_path = ssl_model_dir_abs # 设置 cnhubert 模块的路径
    ssl_model = cnhubert.get_model()
    if is_half:
        ssl_model = ssl_model.half().to(device)
    else:
        ssl_model = ssl_model.to(device)
    ssl_model.eval() # 设置为评估模式
    print("SSL模型加载成功。")

# 加载SoVITS模型
def load_sovits_model(sovits_path_str_abs, project_root_for_models):
    global vq_model, hps, symbols_version, sovits_model_version, if_lora_v3v4, device, is_half
    global mel_fn_dynamic # 用于动态配置梅尔频谱函数
    
    # 从SoVITS模型路径快速推断版本信息
    _symbols_version, _sovits_model_version, _if_lora_v3v4 = get_sovits_version_from_path_fast(sovits_path_str_abs)
    symbols_version, sovits_model_version, if_lora_v3v4 = _symbols_version, _sovits_model_version, _if_lora_v3v4
    
    print(f"SoVITS模型路径: {sovits_path_str_abs}, 推断符号版本: {symbols_version}, 架构版本: {sovits_model_version}, 是否LoRA: {if_lora_v3v4}")

    # 加载SoVITS模型权重和配置
    dict_s2 = load_sovits_new(sovits_path_str_abs)
    hps_dict = dict_s2["config"]
    
    hps = DictToAttrRecursive(hps_dict) # 将配置字典转换为属性对象
    hps.model.semantic_frame_rate = "25hz" # 固定语义帧率为25Hz

    # 根据权重再次确认符号版本 (更可靠)
    if "enc_p.text_embedding.weight" not in dict_s2["weight"]:
        final_symbols_version = "v2" # 如果没有文本嵌入权重，通常是v2符号 (例如纯音频微调的模型)
    elif dict_s2["weight"]["enc_p.text_embedding.weight"].shape[0] == 322:
        final_symbols_version = "v1" # v1符号集大小为322
    else:
        final_symbols_version = "v2" # 其他情况为v2符号
    
    if symbols_version != final_symbols_version:
        print(f"警告: get_sovits_version_from_path_fast推断的符号版本 ({symbols_version}) 与根据权重判断的版本 ({final_symbols_version}) 不一致。将使用后者。")
        symbols_version = final_symbols_version
    hps.model.version = symbols_version # 设置SoVITS模型内部使用的符号版本 (如果其内部文本处理器被调用)
    print(f"最终使用的符号版本 (用于文本处理): {symbols_version}")
    print(f"最终使用的SoVITS架构版本 (用于模型加载): {sovits_model_version}")

    # 根据SoVITS架构版本选择合适的模型类
    v3v4set = {"v3", "v4"}
    if sovits_model_version not in v3v4set: # v1 或 v2
        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        )
    else: # v3 或 v4
        # 对于 v3/v4, hps.model.version 应反映架构版本以正确初始化 SynthesizerTrnV3
        hps.model.version = sovits_model_version 
        vq_model = SynthesizerTrnV3(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        )
    
    # 如果不是预训练模型且存在enc_q (通常用于训练)，则移除它
    if "pretrained" not in sovits_path_str_abs.lower() and hasattr(vq_model, 'enc_q'):
        try: del vq_model.enc_q
        except Exception as e: print(f"移除enc_q时发生警告: {e}")

    # 将模型移至设备并设置精度
    if is_half: vq_model = vq_model.half().to(device)
    else: vq_model = vq_model.to(device)
    
    # 加载权重
    if not if_lora_v3v4: # 非LoRA模型
        print(f"加载SoVITS {sovits_model_version}权重:", vq_model.load_state_dict(dict_s2["weight"], strict=False))
    else: # LoRA模型 (v3/v4)
        # LoRA模型需要先加载底模权重，再加载LoRA权重
        s2g_base_map_relative = { # 底模G的相对路径
            "v3": "GPT_SoVITS/pretrained_models/s2Gv3.pth",
            "v4": "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth"
        }
        base_model_path_relative = s2g_base_map_relative.get(sovits_model_version)
        base_model_path_abs = os.path.join(project_root_for_models, base_model_path_relative) if base_model_path_relative else None
        
        if not base_model_path_abs or not os.path.exists(base_model_path_abs):
            error_msg = f"SoVITS {sovits_model_version} LoRA底模 '{base_model_path_abs or '路径未定义'}' 缺失。项目根目录被识别为 '{project_root_for_models}'。"
            print(f"错误: {error_msg}")
            raise FileNotFoundError(error_msg)

        print(f"加载SoVITS {sovits_model_version} LoRA底模 G 权重 ({base_model_path_abs}):",
              vq_model.load_state_dict(load_sovits_new(base_model_path_abs)["weight"], strict=False))
        
        lora_rank = dict_s2.get("lora_rank", 128) # 获取LoRA秩
        # 配置PEFT LoRA
        lora_config_peft = LoraConfig(
            target_modules=["to_k", "to_q", "to_v", "to_out.0"], r=lora_rank, lora_alpha=lora_rank, init_lora_weights=True,
        )
        vq_model.cfm = get_peft_model(vq_model.cfm, lora_config_peft) # 将LoRA应用到cfm模块
        print(f"加载SoVITS {sovits_model_version} LoRA (rank {lora_rank}) 权重:")
        vq_model.load_state_dict(dict_s2["weight"], strict=False) # 加载LoRA增量权重
        print("合并LoRA权重..."); vq_model.cfm = vq_model.cfm.merge_and_unload(); print("LoRA权重合并完成。")

    vq_model.eval(); print("SoVITS模型加载并设置为评估模式。")

    # # --- 添加以下打印 ---
    # print(f"[DEBUG] load_sovits_model: SoVITS Model Version (from path inference): {sovits_model_version}")
    # print(f"[DEBUG] load_sovits_model: HPS sampling_rate: {hps.data.sampling_rate}")
    
    # 根据SoVITS模型版本及其HPS配置mel_fn_dynamic (梅尔频谱计算函数)
    if sovits_model_version == "v4":

        # v4使用特定的n_fft, win_size, hop_size等参数，可能与通用的hps.data不同
        # WebUI中的典型值: n_fft=1280, win_size=1280, hop_size=320, num_mels=100, sampling_rate=32000
        # 这些值理想情况下应来自hps.data (如果已正确设置)
        v4_mel_params = {
            "n_fft": getattr(hps.data, 'filter_length', 1280), 
            "win_size": getattr(hps.data, 'win_length', 1280),
            "hop_size": getattr(hps.data, 'hop_length', 320), 
            #"num_mels": getattr(hps.data, 'n_mel_channels', 100),
            "num_mels":100,
            "sampling_rate": 32000,
            "fmin": getattr(hps.data, 'fmin', 0), 
            "fmax": getattr(hps.data, 'fmax', None), 
            "center": False, # 保持与WebUI一致
        }
        mel_fn_dynamic = lambda x, sr: mel_spectrogram_torch(x,  **v4_mel_params)
        # print(f"[DEBUG] load_sovits_model: V4 Mel Params used: {v4_mel_params}")
    elif sovits_model_version == "v3":
        # v3使用特定的n_fft, win_size, hop_size等参数
        # WebUI中的典型值: n_fft=1024, win_size=1024, hop_size=256, num_mels=100, sampling_rate=24000
        v3_mel_params = {
            "n_fft": getattr(hps.data, 'filter_length', 1024),
            "win_size": getattr(hps.data, 'win_length', 1024),
            "hop_size": getattr(hps.data, 'hop_length', 256),
            #"num_mels": getattr(hps.data, 'n_mel_channels', 100),
            "num_mels":100,
            "sampling_rate":24000,
            "fmin": getattr(hps.data, 'fmin', 0),
            "fmax": getattr(hps.data, 'fmax', None),
            "center": False, # 保持与WebUI一致
        }
        mel_fn_dynamic = lambda x, sr: mel_spectrogram_torch(x,  **v3_mel_params)
        # print(f"[DEBUG] load_sovits_model: V3 Mel Params used: {v3_mel_params}")
    else: # v1/v2
        # v1/v2直接使用hps.data中的参数计算梅尔频谱
        mel_fn_dynamic = lambda x, sr: mel_spectrogram_torch(
            x, n_fft=hps.data.filter_length, num_mels=hps.data.n_mel_channels, sampling_rate=sr,
            hop_size=hps.data.hop_length, win_size=hps.data.win_length, 
            fmin=hps.data.fmin, fmax=hps.data.fmax, center=False, # 保持与WebUI一致
        )
        # print(f"[DEBUG] load_sovits_model: V1/V2 Mel Params: n_fft={hps.data.filter_length}, num_mels={hps.data.n_mel_channels}, hop_size={hps.data.hop_length}, win_size={hps.data.win_length}")


# 加载GPT模型
def load_gpt_model(gpt_path_str_abs):
    global t2s_model, gpt_config, hz, max_sec, device, is_half
    hz = 50 # 固定值
    print(f"从 {gpt_path_str_abs} 加载GPT模型...")
    dict_s1 = torch.load(gpt_path_str_abs, map_location="cpu") # 先加载到CPU
    gpt_config = dict_s1["config"]
    max_sec = gpt_config["data"]["max_sec"] # 获取最大生成时长
    
    # 初始化Text-to-Semantic模型
    t2s_model = Text2SemanticLightningModule(gpt_config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"]) # 加载权重
    if is_half: t2s_model = t2s_model.half() # 设置精度
    t2s_model = t2s_model.to(device); t2s_model.eval() # 移至设备并设为评估模式
    print("GPT模型加载成功。")

# 初始化外部声码器模型 (BigVGAN for v3, HiFiGAN for v4)
def init_vocoder_model(project_root_for_models):
    global bigvgan_model, hifigan_model, sovits_model_version, device, is_half, hps
    if sovits_model_version == "v3":
        if bigvgan_model is None: # 仅当未加载时加载
            try:
                from GPT_SoVITS.BigVGAN import bigvgan # 修正导入路径
                # BigVGAN预训练模型相对路径
                vocoder_path_relative = "GPT_SoVITS/pretrained_models/models--nvidia--bigvgan_v2_24khz_100band_256x"
                vocoder_path_abs = os.path.join(project_root_for_models, vocoder_path_relative)
                
                print(f"尝试从以下路径加载BigVGAN (v3声码器): {vocoder_path_abs}")
                if not os.path.exists(os.path.join(vocoder_path_abs, "config.json")): # 检查配置文件是否存在
                     print(f"错误: BigVGAN v3声码器模型在 {vocoder_path_abs} 未找到config.json。"); return 

                bigvgan_model = bigvgan.BigVGAN.from_pretrained(vocoder_path_abs, use_cuda_kernel=False) # 加载模型
                bigvgan_model.remove_weight_norm(); bigvgan_model = bigvgan_model.eval() # 移除权重归一化并设为评估模式
                if is_half: bigvgan_model = bigvgan_model.half().to(device) # 设置精度和设备
                else: bigvgan_model = bigvgan_model.to(device)
                print("BigVGAN (v3声码器)加载成功。")
            except Exception as e: print(f"加载BigVGAN时出错: {e}"); traceback.print_exc()
    elif sovits_model_version == "v4":
        if hifigan_model is None: # 仅当未加载时加载
            try:
                print("初始化HiFiGAN (v4声码器) 使用预设参数...")
                # HiFiGAN (Generator) 参数来自 WebUI
                hifigan_model = Generator( 
                    initial_channel=100, 
                    resblock="1", 
                    resblock_kernel_sizes=[3, 7, 11], 
                    resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]], 
                    upsample_rates=[10, 6, 2, 2, 2], # 适配32kHz输出
                    upsample_initial_channel=512, 
                    upsample_kernel_sizes=[20, 12, 4, 4, 4], # 对应upsample_rates
                    gin_channels=0, is_bias=True # SoVITS的HiFiGAN通常不使用gin_channels
                )
                hifigan_model.eval() # 设置为评估模式
                hifigan_model.remove_weight_norm() # 移除权重归一化
                
                # HiFiGAN权重文件相对路径
                vocoder_pth_relative = "GPT_SoVITS/pretrained_models/gsv-v4-pretrained/vocoder.pth"
                vocoder_pth_abs = os.path.join(project_root_for_models, vocoder_pth_relative)

                print(f"尝试加载HiFiGAN (v4声码器)权重: {vocoder_pth_abs}")
                if not os.path.exists(vocoder_pth_abs): # 检查权重文件是否存在
                    print(f"错误: HiFiGAN v4声码器权重 {vocoder_pth_abs} 未找到。"); return

                state_dict_g = torch.load(vocoder_pth_abs, map_location="cpu") # 加载权重
                print("加载HiFiGAN (v4声码器)权重:", hifigan_model.load_state_dict(state_dict_g))
                
                if is_half: hifigan_model = hifigan_model.half().to(device) # 设置精度和设备
                else: hifigan_model = hifigan_model.to(device)
                print("HiFiGAN (v4声码器)加载成功。")
            except Exception as e: print(f"加载HiFiGAN时出错: {e}"); traceback.print_exc()
    else: # v1/v2
        print(f"SoVITS {sovits_model_version} 使用集成声码器, 无需加载外部声码器。")

# --- 文本处理和特征提取函数 ---
# 获取BERT特征 (命令行版本，与WebUI内部调用类似)
def get_bert_feature_cli(text, word2ph):
    global tokenizer, bert_model, device, is_half
    with torch.no_grad(): # 无需计算梯度
        inputs = tokenizer(text, return_tensors="pt") # 分词并转换为PyTorch张量
        for i_name in inputs: inputs[i_name] = inputs[i_name].to(device) # 移至设备
        res = bert_model(**inputs, output_hidden_states=True) # BERT前向传播
        # 取倒数第二层和倒数第三层隐状态的拼接，并移除CLS和SEP token
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    
    phone_level_feature = []
    for i in range(len(word2ph)):
        if i < res.shape[0]: # 确保索引在范围内
            repeat_feature = res[i].repeat(word2ph[i], 1) # 根据每个词对应的音素数量复制特征
            phone_level_feature.append(repeat_feature)
        # else: # 此情况理论上不应发生，如果word2ph和文本长度与分词器输出匹配
            # print(f"警告: BERT特征提取时发生不匹配。文本: '{text}', word2ph长度: {len(word2ph)}, res.shape[0]: {res.shape[0]}, 索引i: {i}")

    if not phone_level_feature: # 如果没有生成特征 (例如空文本或全部不匹配)
        num_total_phones = sum(word2ph) if word2ph else 0
        # print(f"警告: 文本 '{text}' 的BERT特征列表为空。为 {num_total_phones} 个音素返回零值。")
        # 返回正确形状的零张量 (1024特征维度, num_total_phones时间步)
        return torch.zeros((1024, num_total_phones), dtype=torch.float16 if is_half else torch.float32)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T # 转置为 (特征维度, 时间步)


# 清理文本并转换为音素序列 (推理版本)
def clean_text_inf(text, language_str):
    global symbols_version # 此处的 symbols_version 至关重要
    # 使用clean_text进行文本清洗和音素转换 (来自GPT_SoVITS.text.cleaner)
    phones, word2ph, norm_text = clean_text(text, language_str, symbols_version)
    # 将音素字符串转换为ID序列 (来自GPT_SoVITS.text)
    phones = cleaned_text_to_sequence(phones, symbols_version)
    return phones, word2ph, norm_text

# 根据是否使用半精度设置PyTorch数据类型
dtype_torch = torch.float16 if is_half else torch.float32

# 获取音素对应的BERT特征 (推理版本)
def get_bert_inf(phones, word2ph, norm_text, language_str):
    global device, is_half, dtype_torch # dtype_torch将根据is_half设置
    if language_str == "zh": # 中文文本使用BERT提取特征
        bert = get_bert_feature_cli(norm_text, word2ph).to(device)
    else: # 其他语言 (如英文、日文) 使用零向量作为BERT特征
        bert = torch.zeros((1024, len(phones)), dtype=dtype_torch,).to(device)
    return bert

# 获取文本的音素序列和BERT特征 (综合处理函数)
def get_phones_and_bert(text, language_code, text_is_final=False):
    global symbols_version, dtype_torch # symbols_version 和 dtype_torch 是全局变量
    
    # 处理 'all_zh' (全中文模式) 下包含英文字符的情况，转换为 'zh' (中英混合模式)
    if language_code == "all_zh" and re.search(r"[A-Za-z]", text):
        text = re.sub(r"[a-z]", lambda x: x.group(0).upper(), text) # 小写转大写
        text = chinese.mix_text_normalize(text) # 使用 mix_text_normalize (来自 GPT_SoVITS.text import chinese)
        return get_phones_and_bert(text, "zh", text_is_final) # 递归调用，使用 'zh' 模式

    if language_code in {"en", "all_zh", "all_ja"}: # 直接处理这些语言模式
        _text = text
        while "  " in _text: _text = _text.replace("  ", " ") # 去除多余空格
        lang_for_clean = language_code.replace("all_", "") # 例如从 'all_zh' 获取 'zh' 用于 clean_text
        
        phones, word2ph, norm_text = clean_text_inf(_text, lang_for_clean)
        bert = get_bert_inf(phones, word2ph, norm_text, lang_for_clean) # lang_for_clean 用于BERT决策
    
    elif language_code in {"zh", "ja", "auto"}: # 分段处理这些语言模式
        textlist, langlist = [], []
        if language_code == "auto": # 自动检测语言
            for tmp in LangSegmenter.getTexts(text): 
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        else: # "zh" (中英混合) 或 "ja" (日英混合)
            for tmp in LangSegmenter.getTexts(text):
                if tmp["lang"] == "en": langlist.append("en") # 英文部分按英文处理
                else: langlist.append(language_code) # 其他部分强制按用户选择的语言处理
                textlist.append(tmp["text"])
        
        phones_list, bert_list, norm_text_list = [], [], []
        for i in range(len(textlist)): # 逐段处理
            lang_segment, text_segment = langlist[i], textlist[i]
            _phones, _word2ph, _norm_text = clean_text_inf(text_segment, lang_segment)
            _bert = get_bert_inf(_phones, _word2ph, _norm_text, lang_segment)
            
            phones_list.extend(_phones) # 使用 extend 合并音素ID列表
            norm_text_list.append(_norm_text)
            bert_list.append(_bert)
        
        if not bert_list: # 处理未生成BERT特征的情况
             # 如果 phones_list 非空，计算总音素数
            total_phones = len(phones_list) if phones_list else 0
            bert = torch.zeros((1024, total_phones), dtype=dtype_torch).to(device)
        else:
            bert = torch.cat(bert_list, dim=1) # 拼接各段的BERT特征
        
        phones, norm_text = phones_list, "".join(norm_text_list) # phones 已是扁平列表
    else: 
        raise ValueError(f"get_phones_and_bert不支持的语言代码: {language_code}")

    # 如果文本太短，在前面添加标点符号 (模仿WebUI行为)
    if not text_is_final and len(phones) < 6:
        leading_punc = "。" if language_code in ["zh", "all_zh", "ja", "all_ja", "auto"] else "."
        return get_phones_and_bert(leading_punc + text, language_code, text_is_final=True)
        
    return phones, bert.to(dtype_torch), norm_text


# --- 音频处理函数 ---
# 获取参考音频的频谱 (SoVITS v1/v2 版本，直接使用hps中的梅尔参数)
def get_spepc_v1v2(audio_path): 
    global hps, device, dtype_torch, mel_fn_dynamic
    # librosa.load 的采样率应为 hps.data.sampling_rate
    audio, sr = librosa.load(audio_path, sr=int(hps.data.sampling_rate)) 
    audio_norm = torch.FloatTensor(audio).unsqueeze(0) # (B, T_audio)
    # mel_fn_dynamic 需要 (音频张量, 采样率) 作为输入
    spec = mel_fn_dynamic(audio_norm, int(hps.data.sampling_rate)).to(device).to(dtype_torch)
    return spec

# SoVITS v3/v4 梅尔频谱归一化参数
spec_min_v3v4, spec_max_v3v4 = -12, 2
def norm_spec_v3v4(x): return (x - spec_min_v3v4) / (spec_max_v3v4 - spec_min_v3v4) * 2 - 1
def denorm_spec_v3v4(x): return (x + 1) / 2 * (spec_max_v3v4 - spec_min_v3v4) + spec_min_v3v4

# --- 文本切分函数 (来自 inference_webui.py) ---
# 按标点符号切分文本为句子列表
def split_by_punctuation(todo_text): 
    todo_text = todo_text.replace("……", "。").replace("——", "，") # 替换特殊标点
    if not todo_text or todo_text[-1] not in splits: # 处理空字符串或末尾无标点的情况
        todo_text += "。"
    
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    sentences = []
    while i_split_head < len_text:
        if todo_text[i_split_head] in splits: # 找到标点
            sentences.append(todo_text[i_split_tail : i_split_head + 1]) # 切分句子 (包含标点)
            i_split_tail = i_split_head + 1 # 更新下一句的起始位置
        i_split_head += 1
    
    if i_split_tail < len_text: # 理论上如果末尾有标点则不应发生
        sentences.append(todo_text[i_split_tail:])
        
    return [s for s in sentences if s.strip()] # 移除列表中的空字符串

# 按每4句切分文本 (cut1 逻辑)
def cut1_split_every_4_sentences(inp): 
    inp = inp.strip("\n") # 去除首尾空白换行
    if not inp: return "" # 处理空输入
    sentences = split_by_punctuation(inp) # 按标点切分为句子列表
    
    text_chunks = []
    if not sentences: return inp # 如果未找到句子 (例如只有空格)，返回原始输入

    for i in range(0, len(sentences), 4): # 每4句一组
        chunk = "".join(sentences[i:i+4])
        text_chunks.append(chunk)
    
    # 过滤掉完全由标点组成的空块
    text_chunks = [item for item in text_chunks if item.strip() and not set(item.strip()).issubset(punctuation)]
    return "\n".join(text_chunks) # 返回用换行符分隔的文本块字符串


# --- 核心推理函数 ---
# 对单个文本块执行TTS (重构以匹配WebUI逻辑)
def perform_tts_on_chunk(ref_wav_path, prompt_semantic_global, phones_prompt_global, bert_prompt_global,
                         text_chunk_to_synthesize, text_language_code,
                         top_k, temperature, speed_factor, sample_steps_cfg=None):
    global device, is_half, dtype_torch, hps, gpt_config, hz, max_sec
    global ssl_model, vq_model, t2s_model, symbols_version, sovits_model_version
    global bigvgan_model, hifigan_model, mel_fn_dynamic

    # 1. 处理当前文本块：获取音素和BERT特征
    print(f"  正在处理文本块: '{text_chunk_to_synthesize}'")
    phones_target, bert_target, norm_text_target = get_phones_and_bert(text_chunk_to_synthesize, text_language_code)
    
    # # --- 添加以下打印 ---
    # print(f"[DEBUG] perform_tts_on_chunk: Target text chunk: '{text_chunk_to_synthesize}'")
    # print(f"[DEBUG] perform_tts_on_chunk: Target phones_target length: {len(phones_target)}")
    # print(f"[DEBUG] perform_tts_on_chunk: Target bert_target shape: {bert_target.shape}")
    # # --- 打印结束 --
    
    bert_target = bert_target.to(device) # 确保目标BERT特征在正确设备上

    # 2. 准备GPT输入：拼接参考信息 (如果存在) 和目标信息
    #   is_ref_free: 是否为无参考文本模式 (即 prompt_semantic_global 为 None)
    is_ref_free = prompt_semantic_global is None

    if not is_ref_free: # 有参考文本模式
        all_phoneme_ids = torch.LongTensor(phones_prompt_global + phones_target).to(device).unsqueeze(0)
        bert_full = torch.cat([bert_prompt_global, bert_target], dim=1)
        # prompt_semantic_global 已经是 (T_codes_prompt), infer_panel需要 (B, T_codes_prompt)
        current_prompt_for_infer_panel = prompt_semantic_global.unsqueeze(0) 
    
        # # --- 添加以下打印 ---
        # print(f"[DEBUG] perform_tts_on_chunk: all_phoneme_ids (prompt+target) shape: {all_phoneme_ids.shape}")
        # print(f"[DEBUG] perform_tts_on_chunk: bert_full (prompt+target) shape: {bert_full.shape}")
        # print(f"[DEBUG] perform_tts_on_chunk: current_prompt_for_infer_panel (ref codes for GPT) shape: {current_prompt_for_infer_panel.shape}")
        # # --- 打印结束 ---
    
    else: # 无参考文本模式 (ref_free)
        # SoVITS v3/v4 在WebUI中不支持完全的ref_free (即无参考音频)
        # 此处脚本逻辑：如果prompt_text为空，则为ref_free，但v3/v4仍需ref_wav_path获取音色
        if sovits_model_version in {"v3", "v4"} and not ref_wav_path:
             print("错误: SoVITS v3/v4 无参考文本模式仍需要参考音频路径。")
             return None # 返回None表示此块失败
        all_phoneme_ids = torch.LongTensor(phones_target).to(device).unsqueeze(0)
        bert_full = bert_target # 只有目标的BERT特征
        current_prompt_for_infer_panel = None # 无语义参考

        # print(f"[DEBUG] ref_free perform_tts_on_chunk: all_phoneme_ids (target only) shape: {all_phoneme_ids.shape}")
        # print(f"[DEBUG] ref_free perform_tts_on_chunk: bert_full (target only) shape: {bert_full.shape}")
        # print(f"[DEBUG] ref_free perform_tts_on_chunk: current_prompt_for_infer_panel is None (ref_free)")

    bert_full = bert_full.unsqueeze(0) # 形状变为 (B, 1024, T_bert_total)
    all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device) # 音素序列总长度

    # 3. GPT推理：文本转语义 (Text-to-Semantic)
    with torch.no_grad():
        pred_semantic, idx = t2s_model.model.infer_panel(
            all_phoneme_ids,                # 输入的音素ID序列
            all_phoneme_len,                # 音素序列长度
            current_prompt_for_infer_panel, # 参考语义 (可能为None)
            bert_full,                      # 对应的BERT特征
            top_k=top_k, top_p=1.0, temperature=temperature, # 推理参数 (top_p=1.0 与WebUI默认一致)
            early_stop_num=hz * max_sec,    # 提前停止参数
        )
    # 提取新生成的目标语义部分
    pred_semantic_target = pred_semantic[:, -idx:].unsqueeze(1) # 形状 (B, 1, T_s_target)

    # # --- 添加以下打印 ---
    # print(f"[DEBUG] perform_tts_on_chunk: pred_semantic (raw from GPT) shape: {pred_semantic.shape}")
    # print(f"[DEBUG] perform_tts_on_chunk: idx (length of target semantic from GPT): {idx.item() if isinstance(idx, torch.Tensor) else idx}")
    # print(f"[DEBUG] perform_tts_on_chunk: pred_semantic_target (for SoVITS) shape: {pred_semantic_target.shape}") # (B, 1, T_s_target)
    # # --- 打印结束 ---

    # 4. SoVITS推理：语义转音频 (Semantic-to-Waveform)
    audio_output_chunk = None
    if sovits_model_version not in {"v3", "v4"}: # SoVITS v1/v2
        if not ref_wav_path: # v1/v2必须有参考音频
            print("错误: SoVITS v1/v2 推理需要参考音频。"); return None
        ref_spec_v1v2 = get_spepc_v1v2(ref_wav_path) # 获取参考音频的梅尔频谱
        with torch.no_grad():
            # 使用SoVITS模型解码生成音频
            audio_output_chunk_tensor = vq_model.decode(
                pred_semantic_target, # 目标语义
                torch.LongTensor(phones_target).to(device).unsqueeze(0), # 目标音素
                ref_spec_v1v2,        # 参考频谱 (单个)
                speed=speed_factor    # 语速控制
            )[0, 0] # 取第一个batch的第一个样本 (T_audio)
            audio_output_chunk = audio_output_chunk_tensor.data.cpu().float().numpy()

    else: # SoVITS v3/v4
        if not ref_wav_path: # v3/v4也必须有参考音频 (即使prompt_text为空，也用ref_wav获取音色)
            print("错误: SoVITS v3/v4 推理需要参考音频。"); return None

        # --- 准备参考音频及其特征 (v3/v4专用逻辑) ---
        # a. 参考音频的线性频谱：用于 decode_encp 的 'refer' 参数
        ref_audio_raw, ref_sr_orig = librosa.load(ref_wav_path, sr=None) # 加载原始采样率的参考音频
        ref_audio_torch_orig = torch.from_numpy(ref_audio_raw).to(device).to(dtype_torch)
        if ref_audio_torch_orig.ndim == 1: ref_audio_torch_orig = ref_audio_torch_orig.unsqueeze(0) # (B, T_audio)
        if ref_audio_torch_orig.shape[0] == 2: ref_audio_torch_orig = ref_audio_torch_orig.mean(0, keepdim=True) # 转为单声道

        # 将参考音频重采样至模型的目标采样率 (hps.data.sampling_rate)，用于计算线性频谱
        # 此采样率应与声码器的目标采样率一致
        model_target_sr = int(hps.data.sampling_rate)
        if ref_sr_orig != model_target_sr:
            ref_audio_resampled_for_linear = resample(ref_audio_torch_orig, ref_sr_orig, model_target_sr)
        else:
            ref_audio_resampled_for_linear = ref_audio_torch_orig

        # # --- 添加以下打印 ---
        # print(f"[DEBUG] perform_tts_on_chunk (v4): ref_wav_path: {ref_wav_path}")
        # print(f"[DEBUG] perform_tts_on_chunk (v4): ref_sr_orig (from librosa.load sr=None): {ref_sr_orig}")
        # print(f"[DEBUG] perform_tts_on_chunk (v4): model_target_sr (hps.data.sampling_rate for v4): {model_target_sr}")
        # # --- 打印结束 ---

        # 计算线性频谱 (spectrogram_torch 来自 GPT_SoVITS.module.mel_processing)
        linear_spec_refer = spectrogram_torch( 
            ref_audio_resampled_for_linear,
            hps.data.filter_length, model_target_sr, hps.data.hop_length, hps.data.win_length,
            center=False # 与WebUI一致
        ).to(device).to(dtype_torch) # 形状 (B, 频点数_线性, 帧数_参考)
        
        # # --- 添加以下打印 ---
        # print(f"[DEBUG] perform_tts_on_chunk (v4): linear_spec_refer shape: {linear_spec_refer.shape}") # (B, F_lin, T_frames_ref_linear)
        # # --- 打印结束 ---
        
        # b. 参考音频的梅尔频谱：用于 CFM (Flow Matching Model) 的 'mels' 参数
        # WebUI 中，v3的梅尔使用24kHz音频，v4使用32kHz音频进行计算和归一化
        # 这应与 hps.data.sampling_rate (即 model_target_sr) 和 mel_fn_dynamic 的配置一致
        if ref_sr_orig != model_target_sr:
             ref_audio_resampled_for_mel = resample(ref_audio_torch_orig, ref_sr_orig, model_target_sr)
        else:
             ref_audio_resampled_for_mel = ref_audio_torch_orig

        ref_mel_for_cfm_raw = mel_fn_dynamic(ref_audio_resampled_for_mel, model_target_sr) # (B, 频点数_梅尔, 帧数_参考)
        ref_mel_for_cfm_normalized = norm_spec_v3v4(ref_mel_for_cfm_raw) # v3/v4特有的梅尔频谱归一化
        
        # print(f"[调试 CFM 输入] ref_mel_for_cfm_raw 形状: {ref_mel_for_cfm_raw.shape}") # (B, n_mels, T_frames_ref)
        # print(f"[调试 CFM 输入] ref_mel_for_cfm_normalized 形状: {ref_mel_for_cfm_normalized.shape}") # (B, n_mels, T_frames_ref)
        # print(f"[调试 CFM 输入] hps.data.n_mel_channels (或等效值): {getattr(hps.data, 'n_mel_channels', '未找到')}")
        # if hasattr(vq_model, 'cfm') and hasattr(vq_model.cfm, 'n_feats'):
        #     print(f"[调试 CFM 输入] vq_model.cfm.n_feats: {vq_model.cfm.n_feats}")
            
        # --- SoVITS v3/v4 decode_encp 和 CFM 推理 ---
        # 准备参考音素和目标音素的张量
        phones_prompt_tensor = torch.LongTensor(phones_prompt_global if not is_ref_free else []).to(device).unsqueeze(0)
        phones_target_tensor = torch.LongTensor(phones_target).to(device).unsqueeze(0)

        with torch.no_grad():
            # 步骤1: 使用 decode_encp 获取说话人嵌入(ge)和CFM的参考特征(fea_ref_from_encp)
            #         decode_encp的第一个参数(语义)需要是 (B, 1, T_codes_prompt) 或 None
            prompt_for_decode_encp_ref = None
            if not is_ref_free and prompt_semantic_global is not None:
                 prompt_for_decode_encp_ref = prompt_semantic_global.unsqueeze(0).unsqueeze(0) # (B,1,T_codes)
            
            fea_ref_from_encp, ge = vq_model.decode_encp(
                prompt_for_decode_encp_ref, # 参考语义 (可能为None)
                phones_prompt_tensor,       # 参考音素 (如果ref_free则为空)
                linear_spec_refer,          # 参考音频的线性频谱
                speed=speed_factor
            ) # fea_ref_from_encp 形状: (B, 特征通道数, 编码后参考帧数)
            
            # # --- 添加以下打印 ---
            # if not is_ref_free and prompt_semantic_global is not None:
            #     print(f"[DEBUG] perform_tts_on_chunk (v4): prompt_for_decode_encp_ref (ref codes for SoVITS) shape: {prompt_for_decode_encp_ref.shape}")
            # else:
            #     print(f"[DEBUG] perform_tts_on_chunk (v4): prompt_for_decode_encp_ref is None")
            # print(f"[DEBUG] perform_tts_on_chunk (v4): phones_prompt_tensor (ref phones for SoVITS) shape: {phones_prompt_tensor.shape}")
            # print(f"[DEBUG] perform_tts_on_chunk (v4): fea_ref_from_encp shape: {fea_ref_from_encp.shape}") # (B, C_feat, T_frames_ref_after_encp)
            # print(f"[DEBUG] perform_tts_on_chunk (v4): ge (speaker embedding) shape: {ge.shape}")
            # # --- 打印结束 ---                        

            # 步骤2: 使用 decode_encp 获取CFM的目标特征(fea_todo_from_encp)
            fea_todo_from_encp, _ = vq_model.decode_encp(
                pred_semantic_target,       # 目标语义 (B, 1, T_s_target)
                phones_target_tensor,       # 目标音素
                linear_spec_refer,          # 此处也使用参考线性频谱 (与WebUI一致)
                ge=ge,                      # 使用上一步得到的说话人嵌入
                speed=speed_factor
            ) # fea_todo_from_encp 形状: (B, 特征通道数, 编码后目标帧数)
            
            # # --- 添加以下打印 ---
            # print(f"[DEBUG] perform_tts_on_chunk (v4): pred_semantic_target (target codes for SoVITS) shape: {pred_semantic_target.shape}")
            # print(f"[DEBUG] perform_tts_on_chunk (v4): phones_target_tensor (target phones for SoVITS) shape: {phones_target_tensor.shape}")
            # print(f"[DEBUG] perform_tts_on_chunk (v4): fea_todo_from_encp shape: {fea_todo_from_encp.shape}") # (B, C_feat, T_frames_target_after_encp)
            # # --- 打印结束 ---


            # --- CFM 循环推理逻辑 (匹配WebUI) ---
            # 对齐 fea_ref_from_encp 和 ref_mel_for_cfm_normalized 的帧数
            # 通常线性谱和梅尔谱使用相同的hop_length，帧数应一致。若不一致，WebUI取二者最小帧数。
            min_common_frames = min(fea_ref_from_encp.shape[2], ref_mel_for_cfm_normalized.shape[2])
            fea_ref_aligned_for_cfm = fea_ref_from_encp[:, :, :min_common_frames]
            mel_cond_aligned_for_cfm = ref_mel_for_cfm_normalized[:, :, :min_common_frames]
            
            # print(f"[调试 CFM 条件对齐后] fea_ref_aligned_for_cfm 形状: {fea_ref_aligned_for_cfm.shape}")
            # print(f"[调试 CFM 条件对齐后] mel_cond_aligned_for_cfm 形状: {mel_cond_aligned_for_cfm.shape}")
            
            # CFM迭代参数 (来自WebUI)
            Tref = 468 if sovits_model_version == "v3" else 500  # 参考帧数截断长度
            Tchunk = 934 if sovits_model_version == "v3" else 1000 # 每块处理的目标特征长度上限

            # CFM迭代的初始条件，从对齐后的参考特征/梅尔谱中按Tref截取尾部
            if min_common_frames > Tref:
                mel_cond_for_cfm_iter = mel_cond_aligned_for_cfm[:, :, -Tref:]
                fea_ref_for_cfm_iter = fea_ref_aligned_for_cfm[:, :, -Tref:]
            else: # 如果参考长度不足Tref，则全部使用
                mel_cond_for_cfm_iter = mel_cond_aligned_for_cfm
                fea_ref_for_cfm_iter = fea_ref_aligned_for_cfm
 
            # print(f"[调试 CFM 初始迭代条件] mel_cond_for_cfm_iter 形状: {mel_cond_for_cfm_iter.shape}")
            # print(f"[调试 CFM 初始迭代条件] fea_ref_for_cfm_iter 形状: {fea_ref_for_cfm_iter.shape}")  
        
            # current_T_cond_frames: 当前CFM迭代中，条件部分的帧长 (来自上一次输出的梅尔或初始参考梅尔)
            current_T_cond_frames = fea_ref_for_cfm_iter.shape[2] 
            # chunk_len_for_cfm: 每次从 fea_todo_from_encp 中取出的用于生成新梅尔谱的特征块长度
            chunk_len_for_cfm = Tchunk - current_T_cond_frames
            
            # ... (CFM 循环之前) ...
            # # --- 添加以下打印 ---
            # print(f"[DEBUG] perform_tts_on_chunk (v4 CFM): min_common_frames (for ref align): {min_common_frames}")
            # print(f"[DEBUG] perform_tts_on_chunk (v4 CFM): Tref (CFM ref truncate): {Tref}, Tchunk (CFM target chunk): {Tchunk}")
            # print(f"[DEBUG] perform_tts_on_chunk (v4 CFM): initial mel_cond_for_cfm_iter shape: {mel_cond_for_cfm_iter.shape}")
            # print(f"[DEBUG] perform_tts_on_chunk (v4 CFM): initial fea_ref_for_cfm_iter shape: {fea_ref_for_cfm_iter.shape}")
            # print(f"[DEBUG] perform_tts_on_chunk (v4 CFM): initial current_T_cond_frames: {current_T_cond_frames}")
            # print(f"[DEBUG] perform_tts_on_chunk (v4 CFM): initial chunk_len_for_cfm: {chunk_len_for_cfm}")
            # # --- 打印结束 ---

            cfm_outputs_mel = [] # 存储CFM生成的各段梅尔频谱
            idx_cfm = 0 # fea_todo_from_encp 的处理指针
            while True:
                # 从 fea_todo_from_encp 中取出当前块
                # print(f"\n[调试 CFM 循环 - 开始迭代 {idx_cfm // (chunk_len_for_cfm if chunk_len_for_cfm > 0 else 1) +1}]")
                fea_todo_chunk = fea_todo_from_encp[:, :, idx_cfm : idx_cfm + chunk_len_for_cfm]
                # print(f"[调试 CFM 循环] fea_todo_chunk 形状: {fea_todo_chunk.shape}")
                if fea_todo_chunk.shape[2] == 0: break # 如果没有更多目标特征，结束循环
                idx_cfm += chunk_len_for_cfm # 移动指针到下一个块

                # 准备CFM模块的输入特征：拼接条件特征和当前目标块特征
                # combined_fea_for_cfm 形状: (B, 特征通道数, 条件帧数 + 当前块帧数)
                combined_fea_for_cfm = torch.cat([fea_ref_for_cfm_iter, fea_todo_chunk], dim=2)
                # print(f"[调试 CFM 循环] combined_fea_for_cfm 形状: {combined_fea_for_cfm.shape}") # (B, C_feat, T_combined)
                
                # CFM模块输入需要转置: (B, 总帧数, 特征通道数)
                cfm_input_fea_transposed = combined_fea_for_cfm.transpose(1, 2)
                cfm_input_fea_len = torch.LongTensor([cfm_input_fea_transposed.size(1)]).to(device)
                
                # print(f"[调试 CFM 循环] cfm_input_fea_transposed 形状: {cfm_input_fea_transposed.shape}") # (B, T_combined, C_feat)
                # print(f"[调试 CFM 循环] mel_cond_for_cfm_iter (送入CFM) 形状: {mel_cond_for_cfm_iter.shape}") # 关键点3：这是送入CFM的梅尔条件
                
                # CFM 推理
                # sample_steps_cfg: CFM采样步数，若未提供则使用默认值
                #   WebUI默认 v3:32, v4:8 (通过命令行参数或此处逻辑设定)
                current_sample_steps = sample_steps_cfg or (32 if sovits_model_version == "v3" else 8)
                cfm_res_chunk_mel = vq_model.cfm.inference(
                    cfm_input_fea_transposed,   # 输入特征
                    cfm_input_fea_len,          # 特征长度
                    mel_cond_for_cfm_iter,      # 梅尔条件 (B, 梅尔通道数, 条件帧数)
                    current_sample_steps,       # 采样步数
                    inference_cfg_rate=0        # 推理配置率 (与WebUI一致)
                ) # cfm_res_chunk_mel 形状: (B, 梅尔通道数, 总帧数)


                # 提取新生成部分的梅尔频谱
                # 前 current_T_cond_frames 帧对应的是条件部分
                newly_generated_mel_chunk = cfm_res_chunk_mel[:, :, current_T_cond_frames:]
                cfm_outputs_mel.append(newly_generated_mel_chunk)
                
                # --- 添加以下打印 ---
                print(f"[DEBUG] perform_tts_on_chunk (v4 CFM loop {idx_cfm // (chunk_len_for_cfm if chunk_len_for_cfm > 0 else 1)}): combined_fea_for_cfm shape: {combined_fea_for_cfm.shape}")
                print(f"[DEBUG] perform_tts_on_chunk (v4 CFM loop ...): cfm_res_chunk_mel shape: {cfm_res_chunk_mel.shape}")
                print(f"[DEBUG] perform_tts_on_chunk (v4 CFM loop ...): newly_generated_mel_chunk shape: {newly_generated_mel_chunk.shape}")
                # --- 打印结束 --- 
                 
                
                # 更新下一次迭代的条件
                #   取当前输出梅尔的尾部作为下一次的梅尔条件
                mel_cond_for_cfm_iter = cfm_res_chunk_mel[:, :, -current_T_cond_frames:]
                #   取当前处理的 fea_todo_chunk 的尾部作为下一次的特征条件
                #   这需要与 mel_cond_for_cfm_iter 的长度 (current_T_cond_frames) 保持一致
                #   WebUI逻辑: fea_ref = fea_todo_chunk[:, :, -T_min:] (T_min即此处的current_T_cond_frames)
                #   如果 fea_todo_chunk 比 current_T_cond_frames 短，则取其全部
                if fea_todo_chunk.shape[2] < current_T_cond_frames:
                    fea_ref_for_cfm_iter = fea_todo_chunk 
                    # 如果长度不匹配，CFM可能出错。WebUI似乎对此有鲁棒处理 (取最小长度或填充)。
                    # 为安全起见，确保 fea_ref_for_cfm_iter 和 mel_cond_for_cfm_iter 的时间维度一致。
                    min_len_for_next_cond = min(mel_cond_for_cfm_iter.shape[2], fea_todo_chunk.shape[2])
                    mel_cond_for_cfm_iter = mel_cond_for_cfm_iter[:,:,-min_len_for_next_cond:]
                    fea_ref_for_cfm_iter = fea_todo_chunk[:,:,-min_len_for_next_cond:]
                    current_T_cond_frames = min_len_for_next_cond # 更新下次迭代的条件帧数
                    if current_T_cond_frames == 0: break # 如果无法再提供条件，则停止
                else: # fea_todo_chunk 足够长
                    fea_ref_for_cfm_iter = fea_todo_chunk[:, :, -current_T_cond_frames:]
                    
                # print(f"[调试 CFM 循环 - 更新后条件] mel_cond_for_cfm_iter 形状: {mel_cond_for_cfm_iter.shape}") # 关键点4
                # print(f"[调试 CFM 循环 - 更新后条件] fea_ref_for_cfm_iter 形状: {fea_ref_for_cfm_iter.shape}")   # 关键点5       
                
                chunk_len_for_cfm = Tchunk - current_T_cond_frames # 更新下次迭代的目标块长度
                if chunk_len_for_cfm <=0 : break # 如果没有更多空间生成，则停止

            if not cfm_outputs_mel: # 如果CFM没有产生任何输出 (例如输入过短)
                print("警告: CFM没有产生任何梅尔频谱输出，可能是输入过短。返回空音频。")
                audio_output_chunk = np.array([], dtype=np.float32)
            else:
                # 拼接所有CFM生成的梅尔频谱段
                final_mel_from_cfm = torch.cat(cfm_outputs_mel, dim=2)
                # --- 添加以下打印 ---
                print(f"[DEBUG] perform_tts_on_chunk (v4): final_mel_from_cfm (before denorm) shape: {final_mel_from_cfm.shape}") # (B, n_mels, T_total_mel_frames_generated)
                # 对最终梅尔频谱进行反归一化
                final_mel_denormed = denorm_spec_v3v4(final_mel_from_cfm)

                # 使用外部声码器 (BigVGAN for v3, HiFiGAN for v4) 将梅尔频谱转换为波形
                vocoder_to_use = bigvgan_model if sovits_model_version == "v3" else hifigan_model
                if vocoder_to_use is None: # 检查声码器是否已加载
                    print(f"错误: SoVITS {sovits_model_version} 声码器未加载。"); return None
                
                with torch.inference_mode(): # 使用 torch.inference_mode 优化推理 (来自WebUI)
                    wav_gen = vocoder_to_use(final_mel_denormed) # 声码器推理
                    audio_output_chunk_tensor = wav_gen[0, 0] # (T_audio)
                audio_output_chunk = audio_output_chunk_tensor.data.cpu().float().numpy()

    # 后处理：确保音频数据存在且值在 [-1, 1] 范围内
    if audio_output_chunk is not None and audio_output_chunk.size > 0 :

#         # --- 添加以下打印 ---
#         # audio_output_chunk 是 numpy 数组
#         print(f"[DEBUG] perform_tts_on_chunk (v4): Final audio_output_chunk (from vocoder) length: {audio_output_chunk.shape[0]}")
#         # 假设 HiFiGAN 输出是 32kHz
#         estimated_duration_wav = audio_output_chunk.shape[0] / 32000.0
#         print(f"[DEBUG] perform_tts_on_chunk (v4): Estimated duration from Wav samples (assuming 32kHz): {estimated_duration_wav:.2f}s")
#         # --- 打印结束 ---
        
        max_val = np.abs(audio_output_chunk).max()
        if max_val > 1.0: audio_output_chunk /= max_val # 简单幅度归一化
    elif audio_output_chunk is None: # 如果推理失败
        audio_output_chunk = np.array([], dtype=np.float32) # 确保返回空数组而不是None

    return audio_output_chunk


# 将长文本切块、逐块TTS并拼接结果
def get_tts_wav_stitched(ref_wav_path, prompt_text, prompt_language_code,
                         text_to_synthesize_full, text_language_code,
                         top_k, temperature, speed_factor, 
                         sample_steps_cfg=None, pause_duration_s=0.3):
    global device, is_half, hps, ssl_model, vq_model,bert_model,tokenizer # 添加了bert/tokenizer以供get_phones_and_bert使用

    t_start_total = ttime() # 记录总开始时间
    all_audio_segments = [] # 存储所有音频片段

    # 0. 预处理参考信息 (仅执行一次)
    prompt_semantic_global = None # 全局参考语义
    phones_prompt_global = []     # 全局参考音素
    bert_prompt_global = None     # 全局参考BERT特征

    # 判断是否为无参考文本模式
    #   对于v3/v4, 即使prompt_text为空，只要ref_wav_path存在，就不是完全的zero-shot，而是用ref_wav获取音色
    is_ref_free_mode = not (prompt_text and ref_wav_path)
    if sovits_model_version in {"v3", "v4"}: 
        if not ref_wav_path: # v3/v4必须有参考音频
            print("错误: SoVITS v3/v4 推理必须提供参考音频路径。"); return None, None
        # 对v3/v4而言，ref_free意味着无参考文本，但参考音频仍用于音色提取
        is_ref_free_mode = not prompt_text 

    if not is_ref_free_mode: # 有参考信息模式
        print(f"全局参考文本: '{prompt_text}', 语种代码: {prompt_language_code}")
        print(f"全局参考音频路径: {ref_wav_path}")
        
        # 使用SSL模型和VQ模型从参考音频提取语义编码 (prompt_semantic_global)
        wav16k_ref, _ = librosa.load(ref_wav_path, sr=16000) # SSL模型期望16kHz采样率
        # WebUI中的参考音频时长检查: 3-10秒 (wav16k.shape[0]在48000到160000之间)
        if not (48000 <= wav16k_ref.shape[0] <= 160000*1.5): # 稍微放宽上限
            print(f"警告: 参考音频时长 ({wav16k_ref.shape[0]/16000:.2f}秒) 不在建议的3-10秒范围。")
        
        wav16k_ref_torch = torch.from_numpy(wav16k_ref).to(device)
        if is_half: wav16k_ref_torch = wav16k_ref_torch.half()
        if wav16k_ref_torch.ndim == 1: wav16k_ref_torch = wav16k_ref_torch.unsqueeze(0) # 确保 (B, T)
        
        with torch.no_grad():
            # SSL模型提取隐状态
            ssl_content = ssl_model.model(wav16k_ref_torch)["last_hidden_state"].transpose(1, 2) # (B, C, T_ssl)
            # VQ模型提取离散编码
            codes = vq_model.extract_latent(ssl_content) # (B, 1, T_codes)
            prompt_semantic_global = codes[0, 0] # (T_codes_prompt), 用于 t2s_model.infer_panel
            
            # # --- 添加以下打印 ---
            # print(f"[DEBUG] get_tts_wav_stitched: Reference audio original path: {ref_wav_path}")
            # # 如果能获取原始采样率会更好，但 librosa.load 时已经指定了 sr=16000
            # # wav16k_ref_torch.shape[1] 是16kHz下的样本数
            # print(f"[DEBUG] get_tts_wav_stitched: Reference audio (16kHz) samples: {wav16k_ref_torch.shape[1]}, Duration (16kHz): {wav16k_ref_torch.shape[1]/16000.0:.2f}s")
            # print(f"[DEBUG] get_tts_wav_stitched: prompt_semantic_global (ref codes) shape: {prompt_semantic_global.shape}") # 应该类似 (T_codes_prompt,)
            # # --- 打印结束 ---
            
        # 从参考文本获取音素和BERT特征
        phones_prompt_global, bert_prompt_global, _ = get_phones_and_bert(prompt_text, prompt_language_code)
        bert_prompt_global = bert_prompt_global.to(device) # 确保在正确设备上
    else: # 无参考信息模式
        print("进行无参考文本推理 (GPT部分为zero-shot TTS, SoVITS v3/v4仍通过ref_wav_path进行音色迁移)。")


    # 1. 切分目标文本
    text_chunks_str = cut1_split_every_4_sentences(text_to_synthesize_full) # 使用cut1逻辑切分
    # 按换行符分割，并去除可能产生的空块
    text_chunks_list = [chunk for chunk in text_chunks_str.split('\n') if chunk.strip()] 
    
    if not text_chunks_list: # 如果切分后没有有效文本块
        print("错误: 切分后没有有效的文本块可供合成。")
        return None, None # 返回None表示失败
    
    print(f"切分后的文本块 ({len(text_chunks_list)}块):")

    # 2. 为每个文本块执行TTS
    num_chunks = len(text_chunks_list)
    for i, chunk in enumerate(text_chunks_list):
        # 如果块末尾没有标点，则添加一个 (get_phones_and_bert内部或外部期望有标点)
        if chunk and chunk[-1] not in splits:
             chunk += "。" if text_language_code in ["zh", "all_zh", "ja", "all_ja", "auto"] else "."

        print(f"\n--- 开始处理块 {i+1}/{num_chunks} ---")
        t_start_chunk = ttime() # 记录块处理开始时间
        
        # 调用核心TTS函数处理当前块
        audio_chunk_data = perform_tts_on_chunk(
            ref_wav_path, prompt_semantic_global, phones_prompt_global, bert_prompt_global,
            chunk, text_language_code,
            top_k, temperature, speed_factor, sample_steps_cfg
        )
        
        t_end_chunk = ttime() # 记录块处理结束时间
        print(f"--- 块 {i+1}/{num_chunks} 处理完毕, 耗时: {t_end_chunk - t_start_chunk:.3f}秒 ---")

        if audio_chunk_data is not None and audio_chunk_data.size > 0: # 如果成功生成音频
            all_audio_segments.append(audio_chunk_data)
            if i < num_chunks - 1: # 如果不是最后一个块，则在块之间添加静音
                silence_samples = int(hps.data.sampling_rate * pause_duration_s) # 计算静音样本数
                silence_array = np.zeros(silence_samples, dtype=np.float32) # 创建静音数组
                all_audio_segments.append(silence_array)
        else: # 如果块未能生成音频或生成了空音频
            print(f"警告: 块 {i+1} 未能生成音频或生成了空音频，已跳过。")

    if not all_audio_segments: # 如果所有块均未能生成音频
        print("错误: 所有文本块均未能生成音频。")
        return None, None # 返回None表示失败

    # 3. 拼接所有音频段
    final_audio_data = np.concatenate(all_audio_segments) # 拼接音频
    t_end_total = ttime() # 记录总结束时间
    print(f"\n总推理耗时 (包括切分和拼接): {t_end_total - t_start_total:.3f}秒")
    
    final_sampling_rate_to_return = int(hps.data.sampling_rate) # 默认值
    if sovits_model_version == "v4" and hifigan_model is not None:
        # 根据分析，当前HiFiGAN配置输出48kHz内容
        final_sampling_rate_to_return = 48000
        print(f"[INFO] get_tts_wav_stitched: For SoVITS v4 with HiFiGAN, final output sampling rate is set to {final_sampling_rate_to_return}Hz.")
    elif sovits_model_version == "v3" and bigvgan_model is not None:
        # 假设v3模型的hps.data.sampling_rate是24000，并且BigVGAN输出与之匹配
        # 如果hps.data.sampling_rate不是24000，这里可能也需要硬编码或检查
        if int(hps.data.sampling_rate) == 24000:
            final_sampling_rate_to_return = 24000
            print(f"[INFO] get_tts_wav_stitched: For SoVITS v3 with BigVGAN, final output sampling rate is {final_sampling_rate_to_return}Hz (from hps).")
        else:
            # 如果v3的hps不是24k，这是一个警告或需要修正的地方
            print(f"[WARNING] get_tts_wav_stitched: SoVITS v3 with BigVGAN, but hps.data.sampling_rate is {hps.data.sampling_rate}. Expected 24000Hz. Using 24000Hz for output.")
            final_sampling_rate_to_return = 24000 # 强制为24k
    # 对于v1/v2, final_sampling_rate_to_return 会保持为 hps.data.sampling_rate (通常是32000Hz)

    return final_sampling_rate_to_return, final_audio_data


# 主函数：解析参数并执行推理流程
def main():
    parser = argparse.ArgumentParser(description="GPT-SoVITS 命令行推理工具")
    # 模型路径参数
    parser.add_argument("--gpt_model_path", type=str, required=True, help="GPT模型的.ckpt文件路径")
    parser.add_argument("--sovits_model_path", type=str, required=True, help="SoVITS模型的.pth文件路径")
    parser.add_argument("--bert_model_dir", type=str, required=True, help="BERT预训练模型目录路径")
    parser.add_argument("--ssl_model_dir", type=str, required=True, help="SSL (Hubert/ContentVec)预训练模型目录路径")
    # 参考信息参数
    parser.add_argument("--ref_wav_path", type=str, required=True, help="参考音频的.wav文件路径")
    parser.add_argument("--prompt_text", type=str, default="", help="参考音频对应的文本 (可选, 若为空则为无参考文本模式)")
    parser.add_argument("--prompt_language", type=str, default="中文", help="参考文本的语种 (例如: 中文, 英文, 日文, 中英混合, 多语种混合)")
    # 目标文本参数
    parser.add_argument("--text_path", type=str, required=True, help="包含要合成的文本的文件路径 (UTF-8编码)")
    parser.add_argument("--text_language", type=str, default="中文", help="目标文本的语种 (同prompt_language的选项)")
    # 输出参数
    parser.add_argument("--output_wav_path", type=str, required=True, help="合成后音频的保存路径 (.wav)")
    # 设备和精度参数
    parser.add_argument("--gpu_id", type=str, default="0", help="使用的GPU ID (例如 '0', '0,1', 'cpu')")
    parser.add_argument("--use_half_precision", action="store_true", help="是否使用半精度 (FP16) 进行推理")
    # 推理超参数
    parser.add_argument("--top_k", type=int, default=20, help="GPT推理的top_k参数")
    parser.add_argument("--temperature", type=float, default=0.7, help="GPT推理的temperature参数")
    parser.add_argument("--speed", type=float, default=1.0, help="SoVITS推理的语速参数")
    parser.add_argument("--sample_steps", type=int, default=None, help="CFM采样步数 (仅SoVITS v3/v4, 默认v3:32, v4:8)")
    # 其他参数
    parser.add_argument("--pause_between_chunks_s", type=float, default=0.2, help="切分文本块之间的静音时长 (秒)")
    parser.add_argument("--project_root_dir", type=str, default=PROJECT_ROOT, help="项目根目录 (用于查找相对路径的预训练模型等)")


    args = parser.parse_args()

    global device, is_half, dtype_torch # dtype_torch需要在确定is_half后设置
    
    # 设置项目根目录，用于查找相对路径的依赖 (如LoRA底模、外部声码器)
    project_root_for_models = os.path.abspath(args.project_root_dir)
    print(f"项目根目录设置为: {project_root_for_models}")
    # 将项目根目录和GPT_SoVITS子目录添加到sys.path，确保能找到模块
    if project_root_for_models not in sys.path: sys.path.insert(0, project_root_for_models)
    gpt_sovits_subdir_abs = os.path.join(project_root_for_models, "GPT_SoVITS")
    if gpt_sovits_subdir_abs not in sys.path: sys.path.insert(0, gpt_sovits_subdir_abs)


    # 设置推理设备和精度
    if args.gpu_id.lower() == "cpu" or not torch.cuda.is_available():
        device, is_half = "cpu", False # CPU模式，不使用半精度
        print("使用CPU进行推理。")
    else:
        # 如果指定多个GPU ID，默认使用第一个
        if "," in args.gpu_id: gpu_id_to_use = args.gpu_id.split(",")[0]
        else: gpu_id_to_use = args.gpu_id
        device = f"cuda:{gpu_id_to_use}"
        is_half = args.use_half_precision
        print(f"使用 GPU:{gpu_id_to_use} 并 {'启用半精度 (FP16)' if is_half else '使用全精度 (FP32)'}。")
    
    dtype_torch = torch.float16 if is_half else torch.float32 # 在确定is_half后设置dtype_torch

    # 检查输入文件/目录是否存在
    paths_to_check = [args.gpt_model_path, args.sovits_model_path, args.bert_model_dir, args.ssl_model_dir, args.ref_wav_path, args.text_path]
    abs_paths = [os.path.abspath(p) for p in paths_to_check] # 获取绝对路径
    for p_abs, p_orig in zip(abs_paths, paths_to_check):
        if not os.path.exists(p_abs): 
            print(f"错误: 输入路径 '{p_orig}' (解析为 '{p_abs}') 不存在。"); sys.exit(1)
    
    gpt_model_path_abs, sovits_model_path_abs, bert_model_dir_abs, ssl_model_dir_abs, ref_wav_path_abs, text_path_abs = abs_paths
    output_wav_path_abs = os.path.abspath(args.output_wav_path)
    os.makedirs(os.path.dirname(output_wav_path_abs), exist_ok=True) # 确保输出目录存在

    # 加载所有模型
    try:
        print("开始加载模型..."); t_load_start = ttime()
        load_tokenizer_bert_ssl(bert_model_dir_abs, ssl_model_dir_abs) # 加载BERT和SSL
        load_sovits_model(sovits_model_path_abs, project_root_for_models) # 加载SoVITS (此函数会设置全局的symbols_version, sovits_model_version)
        load_gpt_model(gpt_model_path_abs) # 加载GPT
        init_vocoder_model(project_root_for_models) # 初始化外部声码器 (依赖于sovits_model_version)
        print(f"所有模型加载完毕, 耗时: {ttime() - t_load_start:.3f}秒。")
    except Exception as e: print(f"模型加载过程中发生错误: {e}"); traceback.print_exc(); sys.exit(1)

    # 读取目标文本文件内容
    try:
        with open(text_path_abs, 'r', encoding='utf-8') as f: text_to_synthesize_content = f.read().strip()
        if not text_to_synthesize_content: 
            print(f"错误: 目标文本文件 '{text_path_abs}' 为空。"); sys.exit(1)
    except Exception as e: print(f"读取目标文本文件 '{text_path_abs}' 时发生错误: {e}"); sys.exit(1)

    # 根据符号版本选择语言映射表 (symbols_version在load_sovits_model中设置)
    # WebUI中的语言选项与内部代码的映射
    dict_language_map_v1 = {"中文": "all_zh", "英文": "en", "日文": "all_ja", "中英混合": "zh", "日英混合": "ja", "多语种混合": "auto"}
    dict_language_map_v2 = {**dict_language_map_v1, "粤语": "all_yue", "韩文": "all_ko", "粤英混合": "yue", "韩英混合": "ko", "多语种混合(粤语)": "auto_yue"}
    
    if symbols_version is None: # 理论上应由load_sovits_model设置
        print("错误: symbols_version 未在模型加载时设定。请检查SoVITS模型加载过程。"); sys.exit(1)
    current_lang_map = dict_language_map_v1 if symbols_version == "v1" else dict_language_map_v2
    
    # 将用户输入的语言名称转换为内部使用的语言代码
    try:
        # 如果用户直接输入的是代码 (如 "all_zh")，则直接使用；否则从映射表中查找
        prompt_lang_code = args.prompt_language if args.prompt_language in current_lang_map.values() else current_lang_map[args.prompt_language]
        text_lang_code = args.text_language if args.text_language in current_lang_map.values() else current_lang_map[args.text_language]
    except KeyError as e: 
        print(f"错误: 无效的语言名称 '{e.args[0]}', 可用选项: {list(current_lang_map.keys())} 或其对应的代码。"); sys.exit(1)

    print(f"参考文本语言: '{args.prompt_language}' -> 内部代码: '{prompt_lang_code}'")
    print(f"目标文本语言: '{args.text_language}' -> 内部代码: '{text_lang_code}'")
    
    # 如果用户未指定sample_steps，则根据SoVITS模型版本设置默认值 (仅v3/v4需要)
    sample_steps_to_use = args.sample_steps
    if sample_steps_to_use is None and sovits_model_version in {"v3", "v4"}:
        sample_steps_to_use = 32 if sovits_model_version == "v3" else 8 # WebUI默认值
        print(f"CFM采样步数未指定，根据SoVITS模型版本 ('{sovits_model_version}') 自动设为: {sample_steps_to_use}")
    
    # 执行TTS推理 (带切分和拼接)
    print("开始执行TTS推理 (带切分)...")
    final_sampling_rate, audio_data = get_tts_wav_stitched(
        ref_wav_path=ref_wav_path_abs, 
        prompt_text=args.prompt_text, 
        prompt_language_code=prompt_lang_code,
        text_to_synthesize_full=text_to_synthesize_content, 
        text_language_code=text_lang_code,
        top_k=args.top_k, 
        temperature=args.temperature, 
        speed_factor=args.speed,
        sample_steps_cfg=sample_steps_to_use, # CFM采样步数
        pause_duration_s=args.pause_between_chunks_s # 块间停顿
    )

    # 检查推理结果并保存音频
    if audio_data is None or final_sampling_rate is None or audio_data.size == 0: 
        print("推理失败或未生成有效音频。"); sys.exit(1)
    try:
        sf.write(output_wav_path_abs, audio_data, final_sampling_rate) # 保存为WAV文件
        print(f"合成音频成功保存至: {output_wav_path_abs}")
    except Exception as e: print(f"保存音频文件时发生错误: {e}"); sys.exit(1)

if __name__ == "__main__":
    main()