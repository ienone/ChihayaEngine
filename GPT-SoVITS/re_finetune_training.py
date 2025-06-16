# re_finetune_training.py

import os
import sys
import subprocess
import argparse
import json
import yaml # PyYAML
import shutil

# --- 全局配置 (根据您的环境调整) ---
try:
    # 当脚本放在项目根目录时:
    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
except NameError:
    # 如果直接在解释器中运行或打包后，__file__ 可能不存在
    PROJECT_ROOT = os.getcwd()

PYTHON_EXEC = sys.executable # 使用当前Python解释器
TEMP_DIR = os.path.join(PROJECT_ROOT, "TEMP_TRAIN") # 独立的临时目录，避免与webui冲突

# 训练脚本的相对路径
SOVITS_TRAIN_SCRIPT_V1V2 = "GPT_SoVITS/s2_train.py"
SOVITS_TRAIN_SCRIPT_V3V4 = "GPT_SoVITS/s2_train_v3_lora.py" # v3和v4使用此脚本进行SoVITS部分LoRA微调
GPT_TRAIN_SCRIPT = "GPT_SoVITS/s1_train.py"

# 配置文件模板的相对路径
SOVITS_CONFIG_TEMPLATE = "GPT_SoVITS/configs/s2.json"
GPT_CONFIG_TEMPLATE_V1 = "GPT_SoVITS/configs/s1longer.yaml"
GPT_CONFIG_TEMPLATE_V2_V4 = "GPT_SoVITS/configs/s1longer-v2.yaml" # v2 和 v4 使用此模板

# 默认的实验数据根目录 (例如 logs/experiment_name 存放格式化数据)
DEFAULT_EXP_DATA_ROOT = "logs"
# 默认的模型权重输出根目录
DEFAULT_SOVITS_WEIGHT_OUTPUT_ROOT_BASE = "SoVITS_weights"
DEFAULT_GPT_WEIGHT_OUTPUT_ROOT_BASE = "GPT_weights"


# --- 辅助函数 ---
def check_path_exists(path_to_check, path_type="文件或目录", is_fatal=True):
    """检查路径是否存在，如果不存在则打印错误并可选地退出。"""
    if not os.path.exists(path_to_check):
        print(f"错误: {path_type} '{path_to_check}' 不存在。")
        if is_fatal:
            sys.exit(1)
        return False
    return True

def prepare_training_environment(exp_data_root, exp_name, step_name):
    """准备训练所需的环境和目录。"""
    print(f"\n--- {step_name}: 准备环境 ---")
    exp_dir = os.path.join(exp_data_root, exp_name)
    check_path_exists(exp_dir, f"实验数据目录 '{exp_dir}' (应包含格式化后的训练数据)")
    
    # 确保临时目录存在
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # 检查必要的训练数据文件是否存在 (示例，具体依赖于训练脚本)
    if "SoVITS" in step_name:
        check_path_exists(os.path.join(exp_dir, "2-name2text.txt"), "SoVITS训练所需的2-name2text.txt")
        check_path_exists(os.path.join(exp_dir, "6-name2semantic.tsv"), "SoVITS训练所需的6-name2semantic.tsv")
        # SoVITS还需要SSL特征等，这里不一一列举，假设格式化步骤已完成
    elif "GPT" in step_name:
        check_path_exists(os.path.join(exp_dir, "2-name2text.txt"), "GPT训练所需的2-name2text.txt")
        check_path_exists(os.path.join(exp_dir, "6-name2semantic.tsv"), "GPT训练所需的6-name2semantic.tsv")

    return exp_dir

def run_training_process(train_script_path, config_file_path, env_vars, step_name):
    """执行单个训练脚本进程。"""
    print(f"--- {step_name}: 开始训练 ---")
    
    current_env = os.environ.copy()
    current_env.update(env_vars)
    for key, value in current_env.items():
        current_env[key] = str(value) # 确保所有环境变量都是字符串

    command = [PYTHON_EXEC, train_script_path, "--config", config_file_path]
    
    print(f"执行命令: {' '.join(command)}")
    print(f"使用环境变量 (部分):")
    relevant_keys_to_print = ["_CUDA_VISIBLE_DEVICES", "is_half", "version", "hz"]
    relevant_keys_to_print.extend(env_vars.keys())
    for k in sorted(list(set(relevant_keys_to_print))):
        if k in current_env:
            print(f"  {k}: {current_env[k]}")

    process = subprocess.Popen(command, env=current_env, cwd=PROJECT_ROOT)
    process.wait()

    if process.returncode != 0:
        print(f"错误: {step_name} 训练失败，返回码 {process.returncode}。请检查日志获取详细信息。")
        return False
    
    print(f"{step_name} 训练完成。")
    return True

# --- SoVITS 训练逻辑 ---
def train_sovits(args):
    step_name = f"SoVITS ({args.gpt_sovits_version}) 训练"
    exp_dir = prepare_training_environment(args.exp_data_root, args.experiment_name, step_name)

    # 确定SoVITS版本对应的权重输出目录
    sovits_output_dir = f"{DEFAULT_SOVITS_WEIGHT_OUTPUT_ROOT_BASE}_{args.gpt_sovits_version}"
    os.makedirs(os.path.join(PROJECT_ROOT, sovits_output_dir), exist_ok=True)

    # 加载并修改s2.json配置文件
    s2_config_template_path = os.path.join(PROJECT_ROOT, SOVITS_CONFIG_TEMPLATE)
    check_path_exists(s2_config_template_path, "s2.json配置文件模板")
    with open(s2_config_template_path, 'r', encoding='utf-8') as f:
        s2_config = json.load(f)

    # 根据参数修改配置
    s2_config["train"]["batch_size"] = args.sovits_batch_size
    s2_config["train"]["epochs"] = args.sovits_total_epoch
    s2_config["train"]["text_low_lr_rate"] = args.sovits_text_low_lr_rate # v1/v2 relevant
    s2_config["train"]["pretrained_s2G"] = args.sovits_pretrained_s2g
    s2_config["train"]["pretrained_s2D"] = args.sovits_pretrained_s2d if args.sovits_pretrained_s2d else "" # v1/v2 relevant
    s2_config["train"]["if_save_latest"] = args.sovits_save_latest
    s2_config["train"]["if_save_every_weights"] = args.sovits_save_every_weights
    s2_config["train"]["save_every_epoch"] = args.sovits_save_every_epoch
    s2_config["train"]["gpu_numbers"] = args.gpu_id # 单GPU，所以是单个ID字符串
    s2_config["train"]["grad_ckpt"] = args.sovits_grad_ckpt if args.gpt_sovits_version in ["v3", "v4"] else False
    s2_config["train"]["lora_rank"] = int(args.sovits_lora_rank) if args.gpt_sovits_version in ["v3", "v4"] else 0
    
    s2_config["model"]["version"] = args.gpt_sovits_version # 关键: 设定模型内部版本
    s2_config["data"]["exp_dir"] = exp_dir # 实验数据目录
    s2_config["s2_ckpt_dir"] = exp_dir   # 另一个指向实验数据目录的键
    s2_config["save_weight_dir"] = sovits_output_dir # 权重保存目录
    s2_config["name"] = args.experiment_name # 实验名，可能用于日志或模型命名
    s2_config["version"] = args.gpt_sovits_version # 再次确认版本

    if not args.use_half_precision:
        s2_config["train"]["fp16_run"] = False
        # batch_size 在webui中会减半，但这里让用户直接指定最终的batch_size
        # 如果确实需要基于is_half调整，可以加逻辑：
        # if not args.use_half_precision: args.sovits_batch_size = max(1, args.sovits_batch_size // 2)

    # 将修改后的配置写入临时文件
    tmp_s2_config_path = os.path.join(TEMP_DIR, f"tmp_s2_config_{args.experiment_name}.json")
    with open(tmp_s2_config_path, 'w', encoding='utf-8') as f:
        json.dump(s2_config, f, indent=2)
    print(f"SoVITS训练配置文件已生成: {tmp_s2_config_path}")

    # 确定训练脚本
    if args.gpt_sovits_version in ["v1", "v2"]:
        train_script = SOVITS_TRAIN_SCRIPT_V1V2
    elif args.gpt_sovits_version in ["v3", "v4"]:
        train_script = SOVITS_TRAIN_SCRIPT_V3V4
    else:
        print(f"错误: 不支持的SoVITS版本 '{args.gpt_sovits_version}' 用于训练。")
        return

    train_script_abs_path = os.path.join(PROJECT_ROOT, train_script)

    # 设置环境变量
    env_vars = {
        "_CUDA_VISIBLE_DEVICES": args.gpu_id,
        "is_half": str(args.use_half_precision),
        # "version" 环境变量在webui中用于全局，这里s2_config内部已设置
    }
    
    run_training_process(train_script_abs_path, tmp_s2_config_path, env_vars, step_name)

# --- GPT 训练逻辑 ---
def train_gpt(args):
    step_name = f"GPT ({args.gpt_sovits_version}) 训练"
    exp_dir = prepare_training_environment(args.exp_data_root, args.experiment_name, step_name)

    # 确定GPT版本对应的权重输出目录
    gpt_output_dir = f"{DEFAULT_GPT_WEIGHT_OUTPUT_ROOT_BASE}_{args.gpt_sovits_version}"
    os.makedirs(os.path.join(PROJECT_ROOT, gpt_output_dir), exist_ok=True)

    # 选择GPT配置文件模板
    if args.gpt_sovits_version == "v1":
        gpt_config_template_path = os.path.join(PROJECT_ROOT, GPT_CONFIG_TEMPLATE_V1)
    elif args.gpt_sovits_version in ["v2", "v4"]: # v3 GPT使用v2的配置
        gpt_config_template_path = os.path.join(PROJECT_ROOT, GPT_CONFIG_TEMPLATE_V2_V4)
    else: # v3, 也使用v2的模板
        print(f"提示: GPT版本 '{args.gpt_sovits_version}' 将使用v2/v4的GPT配置文件模板。")
        gpt_config_template_path = os.path.join(PROJECT_ROOT, GPT_CONFIG_TEMPLATE_V2_V4)
        
    check_path_exists(gpt_config_template_path, "GPT配置文件模板")
    with open(gpt_config_template_path, 'r', encoding='utf-8') as f:
        gpt_config = yaml.safe_load(f)

    # 修改配置
    gpt_config["train"]["batch_size"] = args.gpt_batch_size
    gpt_config["train"]["epochs"] = args.gpt_total_epoch
    gpt_config["pretrained_s1"] = args.gpt_pretrained_s1
    gpt_config["train"]["save_every_n_epoch"] = args.gpt_save_every_epoch
    gpt_config["train"]["if_save_every_weights"] = args.gpt_save_every_weights
    gpt_config["train"]["if_save_latest"] = args.gpt_save_latest
    gpt_config["train"]["if_dpo"] = args.gpt_if_dpo
    gpt_config["train"]["half_weights_save_dir"] = gpt_output_dir # 权重保存目录
    gpt_config["train"]["exp_name"] = args.experiment_name
    
    gpt_config["train_semantic_path"] = os.path.join(exp_dir, "6-name2semantic.tsv")
    gpt_config["train_phoneme_path"] = os.path.join(exp_dir, "2-name2text.txt")
    gpt_config["output_dir"] = os.path.join(exp_dir, f"logs_s1_{args.gpt_sovits_version}") # 日志输出目录
    os.makedirs(gpt_config["output_dir"], exist_ok=True)


    if not args.use_half_precision:
        gpt_config["train"]["precision"] = "32"
        # batch_size调整逻辑同SoVITS
        # if not args.use_half_precision: args.gpt_batch_size = max(1, args.gpt_batch_size // 2)
    else:
        # s1longer.yaml 和 s1longer-v2.yaml 默认 precision: "bf16-mixed" 或 "16-mixed"
        # 如果用户指定 use_half_precision，确保配置文件中也是半精度
        if args.gpt_sovits_version == "v1": # s1longer.yaml
             gpt_config["train"]["precision"] = "bf16-mixed" # 或者 "16-mixed" 取决于你的preference
        else: # s1longer-v2.yaml
             gpt_config["train"]["precision"] = "16-mixed"

    # 将修改后的配置写入临时文件
    tmp_gpt_config_path = os.path.join(TEMP_DIR, f"tmp_gpt_config_{args.experiment_name}.yaml")
    with open(tmp_gpt_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(gpt_config, f, default_flow_style=False, sort_keys=False)
    print(f"GPT训练配置文件已生成: {tmp_gpt_config_path}")

    train_script_abs_path = os.path.join(PROJECT_ROOT, GPT_TRAIN_SCRIPT)

    # 设置环境变量
    env_vars = {
        "_CUDA_VISIBLE_DEVICES": args.gpu_id,
        "is_half": str(args.use_half_precision),
        "hz": "25hz" # GPT训练通常需要这个
        # "version" 环境变量对于s1_train.py可能不直接使用，配置内部为主
    }

    run_training_process(train_script_abs_path, tmp_gpt_config_path, env_vars, step_name)


# --- 主函数和命令行参数解析 ---
def main():
    parser = argparse.ArgumentParser(description="GPT-SoVITS 微调训练命令行脚本 (单GPU)")
    parser.add_argument("--experiment_name", required=True, help="实验名称，与数据格式化步骤中使用的名称一致。")
    parser.add_argument("--exp_data_root", default=DEFAULT_EXP_DATA_ROOT, help="格式化后的训练数据所在的根目录。")
    parser.add_argument("--gpu_id", default="0", help="用于训练的GPU卡号 (例如 '0')。")
    parser.add_argument("--use_half_precision", action="store_true", help="是否启用半精度训练。")
    parser.add_argument("--gpt_sovits_version", default="v2", choices=["v1", "v2", "v3", "v4"], help="GPT-SoVITS 的全局版本，会影响部分默认配置和脚本选择。")

    subparsers = parser.add_subparsers(title="训练阶段", dest="stage", required=True)

    # SoVITS 训练参数
    parser_sovits = subparsers.add_parser("sovits", help="执行SoVITS模型微调训练。")
    parser_sovits.add_argument("--sovits_batch_size", type=int, default=6, help="SoVITS训练的批处理大小。")
    parser_sovits.add_argument("--sovits_total_epoch", type=int, default=8, help="SoVITS训练的总轮数。")
    parser_sovits.add_argument("--sovits_save_every_epoch", type=int, default=4, help="SoVITS每隔多少轮保存一次权重。")
    parser_sovits.add_argument("--sovits_save_latest", action="store_true", default=True, help="SoVITS是否仅保存最新的几个权重。")
    parser_sovits.add_argument("--sovits_save_every_weights", action="store_true", default=True, help="SoVITS是否在每个保存点都保存完整的模型权重到weights目录。")
    parser_sovits.add_argument("--sovits_pretrained_s2g", required=True, help="预训练SoVITS Generator (s2G)模型路径。")
    parser_sovits.add_argument("--sovits_pretrained_s2d", default="", help="预训练SoVITS Discriminator (s2D)模型路径 (主要用于v1/v2)。")
    parser_sovits.add_argument("--sovits_text_low_lr_rate", type=float, default=0.4, help="SoVITS文本模块学习率权重 (仅v1/v2)。")
    parser_sovits.add_argument("--sovits_lora_rank", type=str, default="32", choices=["16", "32", "64", "128"], help="SoVITS LoRA秩 (仅v3/v4)。")
    parser_sovits.add_argument("--sovits_grad_ckpt", action="store_true", help="SoVITS是否开启梯度检查点 (仅v3/v4)。")
    parser_sovits.set_defaults(func=train_sovits)

    # GPT 训练参数
    parser_gpt = subparsers.add_parser("gpt", help="执行GPT模型微调训练。")
    parser_gpt.add_argument("--gpt_batch_size", type=int, default=6, help="GPT训练的批处理大小。")
    parser_gpt.add_argument("--gpt_total_epoch", type=int, default=15, help="GPT训练的总轮数。")
    parser_gpt.add_argument("--gpt_save_every_epoch", type=int, default=5, help="GPT每隔多少轮保存一次权重。")
    parser_gpt.add_argument("--gpt_save_latest", action="store_true", default=True, help="GPT是否仅保存最新的几个权重。")
    parser_gpt.add_argument("--gpt_save_every_weights", action="store_true", default=True, help="GPT是否在每个保存点都保存完整的模型权重到weights目录。")
    parser_gpt.add_argument("--gpt_pretrained_s1", required=True, help="预训练GPT (s1)模型路径。")
    parser_gpt.add_argument("--gpt_if_dpo", action="store_true", help="GPT是否开启DPO训练。")
    parser_gpt.set_defaults(func=train_gpt)

    args = parser.parse_args()

    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    
    # 全局环境变量，可能会被子脚本读取
    os.environ["version"] = args.gpt_sovits_version 
    
    # 调用选定阶段的函数
    args.func(args)

if __name__ == "__main__":
    # 确保临时目录存在且PYTHONPATH设置正确
    os.makedirs(TEMP_DIR, exist_ok=True)
    existing_pythonpath = os.environ.get('PYTHONPATH', '')
    os.environ['PYTHONPATH'] = PROJECT_ROOT + os.pathsep + existing_pythonpath
    
    main()