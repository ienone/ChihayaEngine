# re_preprocess_data.py

import os
import sys
import subprocess
import argparse
import shutil

# --- 全局配置---
try:
    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__)) # 在根目录
except NameError:
    PROJECT_ROOT = os.getcwd()
    
PYTHON_EXEC = sys.executable

# 工具脚本的相对路径 (相对于PROJECT_ROOT)
SLICER_SCRIPT = "tools/slice_audio.py"  # 语音切分脚本
DENOISER_SCRIPT = "tools/cmd-denoise.py"  # 降噪脚本
ASR_DAMO_SCRIPT = "tools/asr/funasr_asr.py" # 达摩ASR脚本

# --- 辅助函数 ---
def check_path_exists(path_to_check, path_type="文件或目录", is_fatal=True):
    """检查路径是否存在，如果不存在则打印错误并可选地退出。"""
    if not os.path.exists(path_to_check):
        print(f"错误: {path_type} '{path_to_check}' 不存在。")
        if is_fatal:
            sys.exit(1)
        return False
    return True

def run_command(command_list, step_name="命令"):
    """执行给定的命令列表并等待其完成。"""
    print(f"正在执行 {step_name}: {' '.join(command_list)}")
    try:
        process = subprocess.Popen(command_list, stdout=sys.stdout, stderr=sys.stderr)
        process.wait()
        if process.returncode != 0:
            print(f"错误: {step_name} 执行失败，返回码 {process.returncode}。")
            return False
        print(f"{step_name} 执行成功。")
        return True
    except Exception as e:
        print(f"错误: 执行 {step_name} 时发生异常: {e}")
        return False

# --- 功能模块函数 ---

def run_source_separation_external(input_dir, output_dir, separation_command_template):
    """
    执行外部源分离命令 (例如 Demucs)。
    separation_command_template: 一个包含 {input} 和 {output} 占位符的命令模板。
    """
    print(f"\n--- 步骤: 执行外部源分离 (例如 Demucs) ---")
    check_path_exists(input_dir, "源分离输入目录")
    os.makedirs(output_dir, exist_ok=True)

    supported_extensions = ('.wav', '.mp3', '.flac', '.m4a')
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(supported_extensions):
            input_file = os.path.join(input_dir, filename)
            cmd_str = separation_command_template.format(input=input_file, output=output_dir)
            print(f"为文件 '{filename}' 执行源分离命令: {cmd_str}")
            try:
                subprocess.run(cmd_str, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
            except subprocess.CalledProcessError as e:
                print(f"错误: 文件 '{filename}' 的源分离失败: {e}")
                return False
            except FileNotFoundError:
                print(f"错误: 无法找到源分离命令。请确保它已安装并在PATH中，或者模板中的路径正确。命令: {cmd_str.split()[0]}")
                return False
    print("外部源分离步骤完成（假设所有文件处理完毕）。")
    return True


def run_slicer_tool(input_path, output_root, threshold="-34", min_length="4000",
                    min_interval="300", hop_size="10", max_sil_kept="500",
                    max_vol_norm="0.9", alpha_mix="0.25", num_processes="1"):
    """执行语音切分工具。"""
    print(f"\n--- 步骤: 语音切分 ---")
    check_path_exists(input_path, "切分输入路径")
    os.makedirs(output_root, exist_ok=True)
    script_path = os.path.join(PROJECT_ROOT, SLICER_SCRIPT)
    check_path_exists(script_path, "切分工具脚本")

    command = [
        PYTHON_EXEC, script_path,
        input_path, output_root,
        str(threshold), str(min_length), str(min_interval),
        str(hop_size), str(max_sil_kept), str(max_vol_norm),
        str(alpha_mix),
        "0",
        str(num_processes)
    ]
    return run_command(command, step_name="语音切分")

def run_denoiser_tool(input_dir, output_dir, precision="float32"):
    """执行语音降噪工具。"""
    print(f"\n--- 步骤: 语音降噪 ---")
    check_path_exists(input_dir, "降噪输入目录")
    os.makedirs(output_dir, exist_ok=True)
    script_path = os.path.join(PROJECT_ROOT, DENOISER_SCRIPT)
    check_path_exists(script_path, "降噪工具脚本")

    command = [
        PYTHON_EXEC, script_path,
        "-i", input_dir,
        "-o", output_dir,
        "-p", precision
    ]
    return run_command(command, step_name="语音降噪")

def run_asr_tool_damo(input_dir, output_dir, model_size="large", language="zh", precision="float32"):
    """执行达摩ASR进行语音识别。"""
    print(f"\n--- 步骤: 语音识别 (达摩ASR) ---")
    check_path_exists(input_dir, "ASR输入目录")
    os.makedirs(output_dir, exist_ok=True)
    script_path = os.path.join(PROJECT_ROOT, ASR_DAMO_SCRIPT)
    check_path_exists(script_path, "达摩ASR脚本")

    command = [
        PYTHON_EXEC, script_path,
        "-i", input_dir,
        "-o", output_dir,
        "-s", model_size,
        "-l", language,
        "-p", precision
    ]
    input_dir_basename = os.path.basename(os.path.normpath(input_dir))
    expected_list_file = os.path.join(output_dir, f"{input_dir_basename}.list")

    if run_command(command, step_name="语音识别(达摩ASR)"):
        if os.path.exists(expected_list_file):
            print(f"ASR成功，标注文件已生成: {expected_list_file}")
            return expected_list_file
        else:
            print(f"错误: ASR执行完毕，但未找到预期的标注文件: {expected_list_file}")
            return None
    return None

# --- 主逻辑 ---
def main():
    parser = argparse.ArgumentParser(description="GPT-SoVITS 数据集前置处理命令行脚本 (无UI)")
    parser.add_argument("--input_audio_dir", required=True, help="原始音频文件所在的根目录。")
    parser.add_argument("--experiment_name", required=True, help="实验名称，用于创建输出子目录。")
    parser.add_argument("--output_base_dir", default="output_preprocess", help="所有预处理输出的根目录。")

    parser.add_argument("--run_separation", action="store_true", help="执行外部源分离步骤。")
    parser.add_argument("--separation_cmd_template",
                        default="echo '源分离命令未配置: demucs --two-stems=vocals -o \"{output}\" \"{input}\"'",
                        help="源分离命令模板，用 {input} 和 {output} 作为占位符。")

    parser.add_argument("--slice_threshold", default="-34", help="切分阈值 (dBFS)。")
    parser.add_argument("--slice_min_length", default="4000", help="切片最小长度 (ms)。")
    parser.add_argument("--slice_min_interval", default="300", help="切片最小间隔 (ms)。")
    parser.add_argument("--slice_hop_size", default="10", help="音量曲线计算的跳跃大小 (ms)。")
    parser.add_argument("--slice_max_sil_kept", default="500", help="切片后保留的最大静音长度 (ms)。")
    parser.add_argument("--slice_num_processes", default="1", help="切分工具使用的并行处理数量。")

    parser.add_argument("--run_denoise", action="store_true", help="执行降噪步骤。")
    parser.add_argument("--denoise_precision", default="float32", choices=["float32", "float16"], help="降噪精度。")

    args = parser.parse_args()

    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    base_output_dir = os.path.join(args.output_base_dir, args.experiment_name)
    separated_audio_dir = os.path.join(base_output_dir, "1_separated_vocals")
    sliced_audio_dir = os.path.join(base_output_dir, "2_sliced_audio")
    denoised_audio_dir = os.path.join(base_output_dir, "3_denoised_audio")
    asr_output_dir = os.path.join(base_output_dir, "4_asr_output")

    current_audio_input_dir = args.input_audio_dir

    if args.run_separation:
        if not run_source_separation_external(current_audio_input_dir, separated_audio_dir, args.separation_cmd_template):
            print("源分离失败，流程中止。")
            return
        current_audio_input_dir = separated_audio_dir
        print(f"源分离完成，后续处理将使用目录: {current_audio_input_dir}")
    else:
        print("跳过源分离步骤。")

    if not run_slicer_tool(current_audio_input_dir, sliced_audio_dir,
                           args.slice_threshold, args.slice_min_length, args.slice_min_interval,
                           args.slice_hop_size, args.slice_max_sil_kept,
                           num_processes=args.slice_num_processes):
        print("语音切分失败，流程中止。")
        return
    current_audio_input_dir = sliced_audio_dir
    print(f"语音切分完成，后续处理将使用目录: {current_audio_input_dir}")

    if args.run_denoise:
        if not run_denoiser_tool(current_audio_input_dir, denoised_audio_dir, args.denoise_precision):
            print("语音降噪失败，流程中止。")
            return
        current_audio_input_dir = denoised_audio_dir
        print(f"语音降噪完成，后续处理将使用目录: {current_audio_input_dir}")
    else:
        print("跳过语音降噪步骤。")

    generated_list_file = run_asr_tool_damo(current_audio_input_dir, asr_output_dir,
                                         model_size="large", language="zh", precision="float32")
    if not generated_list_file:
        print("语音识别失败，流程中止。")
        return
    
    print(f"\n语音识别完成。生成的标注列表文件位于: {generated_list_file}")
    print("请手动检查并校对该文件中的文本内容。")
    print("您可以使用 'tools/subfix_webui.py' 或其他文本编辑器进行校对。例如，要启动标注UI:")
    print(f"   {PYTHON_EXEC} {os.path.join(PROJECT_ROOT, 'tools/subfix_webui.py')} --load_list \"{generated_list_file}\" --webui_port YOUR_PORT_HERE")
    
    print("\n所有指定的前置数据集处理步骤已执行完毕。")
    #print(generated_list_file)
    
if __name__ == "__main__":
    existing_pythonpath = os.environ.get('PYTHONPATH', '')
    # 确保项目根目录在PYTHONPATH中，以便子脚本可以导入项目内的其他模块
    # 例如，tools/asr/funasr_asr.py 可能需要导入项目根目录下的其他工具或配置
    os.environ['PYTHONPATH'] = PROJECT_ROOT + os.pathsep + existing_pythonpath
    
    main()