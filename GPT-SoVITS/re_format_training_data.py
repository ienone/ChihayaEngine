# re_format_training_data.py

import os
import sys
import subprocess
import argparse
import shutil

# --- 全局配置 (根据您的环境调整) ---
try:
    # 当脚本放在项目根目录时:
    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
except NameError:
    # 如果直接在解释器中运行或打包后，__file__ 可能不存在
    PROJECT_ROOT = os.getcwd()

PYTHON_EXEC = sys.executable # 使用当前Python解释器

# 工具脚本的相对路径 (相对于PROJECT_ROOT)
SCRIPT_1A_GET_TEXT = "GPT_SoVITS/prepare_datasets/1-get-text.py" # 文本分词与BERT特征提取
SCRIPT_1B_GET_HUBERT_WAV32K = "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py" # 语音自监督特征提取
SCRIPT_1C_GET_SEMANTIC = "GPT_SoVITS/prepare_datasets/3-get-semantic.py" # 语义Token提取
S2_CONFIG_PATH_DEFAULT = "GPT_SoVITS/configs/s2.json" # SoVITS模型配置文件路径

# --- 辅助函数 ---
def check_path_exists(path_to_check, path_type="文件或目录", is_fatal=True, check_empty_if_dir=False):
    """检查路径是否存在，如果不存在则打印错误并可选地退出。可选检查目录是否为空。"""
    if not os.path.exists(path_to_check):
        print(f"错误: {path_type} '{path_to_check}' 不存在。")
        if is_fatal:
            sys.exit(1)
        return False
    if check_empty_if_dir and os.path.isdir(path_to_check) and not os.listdir(path_to_check):
        print(f"错误: 目录 '{path_to_check}' 为空。")
        if is_fatal:
            sys.exit(1)
        return False
    return True

def run_formatting_step(script_relative_path, env_vars, step_name):
    """
    执行单个格式化步骤的脚本。
    :param script_relative_path: 脚本相对于PROJECT_ROOT的路径。
    :param env_vars: 需要为该脚本设置的环境变量字典。
    :param step_name: 步骤的名称，用于打印信息。
    :return: 成功则返回True，失败则返回False。
    """
    script_abs_path = os.path.join(PROJECT_ROOT, script_relative_path)
    check_path_exists(script_abs_path, f"{step_name}脚本")

    print(f"\n--- 步骤: {step_name} ---")
    
    current_env = os.environ.copy()
    current_env.update(env_vars)
    
    for key, value in current_env.items():
        current_env[key] = str(value)

    print(f"执行命令: {PYTHON_EXEC} {script_abs_path}")
    print(f"使用环境变量 (部分):")
    # 只打印我们在此脚本中明确设置或修改的环境变量
    relevant_keys_to_print = ["_CUDA_VISIBLE_DEVICES", "is_half", "exp_name", "opt_dir", "i_part", "all_parts", "version"]
    relevant_keys_to_print.extend(env_vars.keys()) # 也包括特定步骤的env_vars
    for k in sorted(list(set(relevant_keys_to_print))):
        if k in current_env:
            print(f"  {k}: {current_env[k]}")
    
    process = subprocess.Popen([PYTHON_EXEC, script_abs_path], env=current_env, cwd=PROJECT_ROOT)
    process.wait() 

    if process.returncode != 0:
        print(f"错误: {step_name} 执行失败，返回码 {process.returncode}。请检查上面的输出获取详细错误信息。")
        return False
    
    print(f"{step_name} 执行成功。")
    return True

# --- 主逻辑 ---
def main():
    parser = argparse.ArgumentParser(description="GPT-SoVITS 训练集格式化命令行脚本 (单GPU)")
    parser.add_argument("--list_file", required=True, help="校对后的文本标注文件路径 (例如 input/my_exp.list)。")
    parser.add_argument("--wav_dir", required=True, help="训练集音频文件目录 (包含切分后的wav文件)。")
    parser.add_argument("--experiment_name", required=True, help="实验名称，用于创建输出子目录。")
    
    parser.add_argument("--output_base_dir", default="logs", help="所有格式化数据输出的根目录 (例如 logs)。")
    
    parser.add_argument("--bert_model_dir", default=os.path.join(PROJECT_ROOT, "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"), help="预训练中文BERT模型路径。")
    parser.add_argument("--ssl_model_dir", default=os.path.join(PROJECT_ROOT, "GPT_SoVITS/pretrained_models/chinese-hubert-base"), help="预训练SSL模型 (CnHubert) 路径。")
    parser.add_argument("--s2g_model_path", required=True, help="预训练SoVITS-G模型路径 (用于语义Token提取)。通常是官方对应版本的s2G底模。")
    parser.add_argument("--s2_config_path", default=os.path.join(PROJECT_ROOT, S2_CONFIG_PATH_DEFAULT), help="SoVITS-G模型对应的s2.json配置文件路径。")
    
    parser.add_argument("--gpu_id", default="0", help="用于处理的GPU卡号 (单个卡号，例如 '0' 或 '1')。")
    parser.add_argument("--use_half_precision", action="store_true", help="是否启用半精度处理 (如果GPU支持)。")
    parser.add_argument("--gpt_sovits_version", default="v2", choices=["v1", "v2", "v4"], help="GPT-SoVITS的版本，影响某些内部处理逻辑。")

    args = parser.parse_args()

    # --- 路径和环境准备 ---
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    
    check_path_exists(args.list_file, "标注列表文件")
    check_path_exists(args.wav_dir, "音频文件目录", check_empty_if_dir=True)
    check_path_exists(args.bert_model_dir, "BERT模型目录")
    check_path_exists(args.ssl_model_dir, "SSL模型目录")
    check_path_exists(args.s2g_model_path, "SoVITS-G模型")
    check_path_exists(args.s2_config_path, "s2.json配置文件")

    opt_dir = os.path.join(args.output_base_dir, args.experiment_name)
    os.makedirs(opt_dir, exist_ok=True)
    print(f"所有格式化后的数据将输出到: {opt_dir}")

    # 确保GPU ID是单个数字
    if not args.gpu_id.isdigit():
        print(f"错误: GPU ID '{args.gpu_id}' 不是一个有效的数字。请提供单个GPU卡号 (例如 0, 1)。")
        sys.exit(1)

    os.environ["version"] = args.gpt_sovits_version
    common_env_vars = {
        "_CUDA_VISIBLE_DEVICES": args.gpu_id, # 设置为单个GPU
        "is_half": str(args.use_half_precision),
        "exp_name": args.experiment_name,
        "opt_dir": opt_dir,
        # 对于单GPU处理，i_part总是0，all_parts总是1
        "i_part": "0",
        "all_parts": "1",
    }
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    os.environ['PYTHONPATH'] = PROJECT_ROOT + os.pathsep + current_pythonpath

    # --- 执行步骤 ---

    # 1Aa: 文本分词与BERT特征提取
    # 子脚本预计会输出到 opt_dir/2-name2text.txt (因为 i_part=0, all_parts=1)
    env_1a = common_env_vars.copy()
    env_1a.update({
        "inp_text": args.list_file,
        "inp_wav_dir": args.wav_dir,
        "bert_pretrained_dir": args.bert_model_dir,
    })
    if not run_formatting_step(SCRIPT_1A_GET_TEXT, env_1a, "1Aa-文本分词与BERT特征提取"):
        sys.exit(1)
    # 检查最终输出文件是否存在
    final_text_path = os.path.join(opt_dir, "2-name2text.txt")
    check_path_exists(final_text_path, "1Aa输出文件 (2-name2text.txt)")


    # 1Ab: 语音自监督特征提取 (Hubert/ContentVec)
    # 子脚本的输出是特征文件，通常在 opt_dir/5-wav32k 等，不需要聚合主文件
    env_1b = common_env_vars.copy()
    env_1b.update({
        "inp_text": args.list_file, 
        "inp_wav_dir": args.wav_dir, 
        "cnhubert_base_dir": args.ssl_model_dir,
    })
    if not run_formatting_step(SCRIPT_1B_GET_HUBERT_WAV32K, env_1b, "1Ab-语音自监督特征提取"):
        sys.exit(1)
    # 可以添加检查特定特征文件或目录是否生成的逻辑，如果需要

    # 1Ac: 语义Token提取
    # 子脚本预计会输出到 opt_dir/6-name2semantic.tsv
    env_1c = common_env_vars.copy()
    env_1c.update({
        "inp_text": args.list_file, 
        "pretrained_s2G": args.s2g_model_path,
        "s2config_path": args.s2_config_path,
    })
    if not run_formatting_step(SCRIPT_1C_GET_SEMANTIC, env_1c, "1Ac-语义Token提取"):
        sys.exit(1)
    # 检查最终输出文件是否存在
    final_semantic_path = os.path.join(opt_dir, "6-name2semantic.tsv")
    check_path_exists(final_semantic_path, "1Ac输出文件 (6-name2semantic.tsv)")


    print(f"\n--- 训练集格式化完成 (单GPU模式) ---")
    print(f"主要输出文件:")
    print(f"  文本与路径: {final_text_path}")
    print(f"  语义编码: {final_semantic_path}")
    print(f"  SSL特征等位于子目录，例如: {os.path.join(opt_dir, '5-wav32k')}")
    print(f"请检查 {opt_dir} 目录下的输出。")

if __name__ == "__main__":
    main()