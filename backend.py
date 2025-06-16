from flask import Flask, request, jsonify, Response, stream_with_context, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import subprocess
import threading
import uuid
import time
import json
from pathlib import Path
import shutil
import yaml # 用于处理 so-vits-svc 的 diffusion.yaml

# --- 配置 (与之前类似) ---
BASE_DIR = Path("/root/autodl-tmp")
CONDA_PATH = "/root/miniconda3/bin/conda"
FILES_DIR = BASE_DIR / "files"
INPUT_DIR = FILES_DIR / "input"
PROCESSED_DIR = FILES_DIR / "processed"
OUTPUT_DIR = FILES_DIR / "output"

DEMUCS_ENV = "demucs"
GPT_SOVITS_ENV = "GPTSoVits" # 注意：这里要和实际的Conda环境名称一致
SOVITS_SVC_ENV = "so-vits-svc"

GPT_SOVITS_PROJECT_DIR = BASE_DIR / "GPT-SoVITS"
SOVITS_SVC_PROJECT_DIR = BASE_DIR / "so-vits-svc"

FFMPEG_PATH = "ffmpeg"
FFPROBE_PATH = "ffprobe"



def setup_directories():
    # ... (与之前版本相同，确保所有目录存在)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    (INPUT_DIR / "songs_for_separation").mkdir(parents=True, exist_ok=True)
    (INPUT_DIR / "gpt_sovits_training_voice").mkdir(parents=True, exist_ok=True)
    (INPUT_DIR / "gpt_sovits_reference").mkdir(parents=True, exist_ok=True)
    (INPUT_DIR / "gpt_sovits_lyrics").mkdir(parents=True, exist_ok=True)
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "separated_audio").mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "gpt_sovits_output").mkdir(parents=True, exist_ok=True) # GPT推理的最终输出
    (PROCESSED_DIR / "gpt_sovits_preprocess_out").mkdir(parents=True, exist_ok=True) # GPT预处理输出
    (PROCESSED_DIR / "gpt_sovits_formatted_data").mkdir(parents=True, exist_ok=True) # GPT格式化数据输出
    (PROCESSED_DIR / "sliced_for_svc_ffmpeg").mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "temp_ffmpeg_segments").mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "svc_output").mkdir(parents=True, exist_ok=True)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "final_songs").mkdir(parents=True, exist_ok=True)

    (SOVITS_SVC_PROJECT_DIR / "dataset_raw").mkdir(parents=True, exist_ok=True)
    (SOVITS_SVC_PROJECT_DIR / "logs" / "44k").mkdir(parents=True, exist_ok=True)
    (SOVITS_SVC_PROJECT_DIR / "configs").mkdir(parents=True, exist_ok=True)
    (SOVITS_SVC_PROJECT_DIR / "results").mkdir(parents=True, exist_ok=True)
    (SOVITS_SVC_PROJECT_DIR / "raw").mkdir(parents=True, exist_ok=True)
    
    # GPT-SoVITS 项目内可能需要的数据目录 (根据 train.txt)
    (GPT_SOVITS_PROJECT_DIR / "raw_audio_data").mkdir(parents=True, exist_ok=True)
    (GPT_SOVITS_PROJECT_DIR / "output_preprocess_cli").mkdir(parents=True, exist_ok=True) # 对应 train.txt 的 PREPROCESS_OUTPUT_BASE_DIR
    (GPT_SOVITS_PROJECT_DIR / "logs_formatted").mkdir(parents=True, exist_ok=True)      # 对应 train.txt 的 FORMAT_OUTPUT_BASE_DIR
    (GPT_SOVITS_PROJECT_DIR / "output_audio").mkdir(parents=True, exist_ok=True)        # GPT推理输出目录
    (GPT_SOVITS_PROJECT_DIR / "GPT_weights_v4").mkdir(parents=True, exist_ok=True) # 假设v4, 根据版本变化
    (GPT_SOVITS_PROJECT_DIR / "SoVITS_weights_v4").mkdir(parents=True, exist_ok=True)

setup_directories()

tasks = {}
log_listeners = {}

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 300 * 1024 * 1024

# --- 辅助函数 (run_command, log_event, sse_stream 与之前类似) ---
def run_command(command_parts, cwd=None, env_name=None, task_id=None, custom_env=None, shell=False):
    """执行命令并流式传输其输出。"""
    full_command_str_for_log = []
    actual_command_to_run = []

    process_env = os.environ.copy()
    if custom_env:
        process_env.update(custom_env)

    if env_name:
        # 使用 CONDA_PATH
        actual_command_to_run.extend([CONDA_PATH, "run", "--no-capture-output", "-n", env_name])
        full_command_str_for_log.extend(["conda", "run", "-n", env_name]) # 日志中仍然显示 conda，便于阅读
    
    actual_command_to_run.extend(command_parts)
    full_command_str_for_log.extend(command_parts)

    cmd_display_str = ' '.join(str(p) for p in full_command_str_for_log)
    log_event(task_id, f"正在执行: {cmd_display_str} 于 {cwd or os.getcwd()}")
    
    cmd_for_popen = ' '.join(actual_command_to_run) if shell else actual_command_to_run

    process = subprocess.Popen(
        cmd_for_popen,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        cwd=str(cwd) if cwd else None,
        env=process_env,
        shell=shell,
        encoding='utf-8', # 尝试显式指定编码
        errors='replace'  # 替换无法解码的字符
    )

    if task_id and task_id in tasks:
        tasks[task_id]["process_popen"] = process 

    for line in iter(process.stdout.readline, ''):
        log_event(task_id, line.strip())
    
    process.stdout.close()
    return_code = process.wait()
    
    if return_code != 0:
        error_msg = f"命令执行失败，退出码 {return_code}: {cmd_display_str}"
        log_event(task_id, error_msg)
        if task_id and task_id in tasks: tasks[task_id]["status"] = "失败"
        raise RuntimeError(error_msg)
    else:
        log_event(task_id, f"命令执行成功: {cmd_display_str}")
    return True

def log_event(task_id, message):
    # 决定在控制台打印什么，以及放入SSE队列的是什么
    item_for_sse_queue = message  # 默认情况下，直接使用原始消息
    
    # 要一个统一的字符串表示形式用于控制台打印和持久化日志
    console_log_message_str = ""
    persistent_log_entry_str = ""

    if isinstance(message, dict) and message.get("type") in ["stage_update"]: # 如果有其他特殊事件类型，也加入这里
        # 对于控制台日志，清晰地显示这是一个SSE事件
        console_log_message_str = f"[SSE Event] {str(message)}" # 将字典转换为字符串以便打印
        # 对于任务的持久化日志，我们可能想存储其字符串表示或JSON字符串
        persistent_log_entry_str = json.dumps(message, ensure_ascii=False) # 例如，存储为JSON字符串
        # item_for_sse_queue 保持为原始字典 message
    else:
        # 这是一个普通的字符串日志消息
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        # 确保 message 是字符串以便进行拼接
        stringified_message = str(message) 
        console_log_message_str = f"[{timestamp} UTC] {stringified_message}"
        persistent_log_entry_str = console_log_message_str # 持久化日志存储带时间戳的字符串
        item_for_sse_queue = console_log_message_str # 普通日志，SSE队列也放入带时间戳的字符串

    print(f"任务 {task_id}: {console_log_message_str}") 
    
    if task_id and task_id in tasks:
        if "log" not in tasks[task_id]: tasks[task_id]["log"] = []
        # 将确定好的字符串版本存入任务日志
        tasks[task_id]["log"].append(persistent_log_entry_str) 
        
        if task_id in log_listeners:
            for queue in log_listeners[task_id]:
                try:
                    # 将原始项目（字典 或 带时间戳的字符串）放入SSE队列
                    queue.put(item_for_sse_queue, block=False) 
                except Exception:
                    pass

def sse_stream(task_id):
    import queue
    q = queue.Queue(maxsize=200)
    if task_id not in log_listeners: log_listeners[task_id] = []
    log_listeners[task_id].append(q)

    try:
        if task_id in tasks and "log" in tasks[task_id]:
            for log_entry in tasks[task_id]["log"]:
                yield f"data: {json.dumps(log_entry, ensure_ascii=False)}\n\n"
        
        while True:
            current_task_status = tasks.get(task_id, {}).get("status")
            current_stage_info = tasks.get(task_id, {}).get("current_stage_info", "未知阶段")
            
            if current_task_status in ["已完成", "失败"]:
                final_message = {"event": "EOS", "status": current_task_status, "stage": current_stage_info}
                if current_task_status == "已完成":
                    final_message["result_paths"] = tasks[task_id].get("result_paths", {})
                elif current_task_status == "失败":
                    final_message["error"] = tasks[task_id].get("error_message", "未知错误")
                yield f"data: {json.dumps(final_message, ensure_ascii=False)}\n\n"
                break
            try:
                message = q.get(timeout=1)
                # 如果消息是结构化的阶段更新，也发送
                if isinstance(message, dict) and message.get("type") == "stage_update":
                     yield f"data: {json.dumps(message, ensure_ascii=False)}\n\n"
                else: # 普通日志
                    yield f"data: {json.dumps(message, ensure_ascii=False)}\n\n"
            except queue.Empty:
                yield ": heartbeat\n\n"
            except Exception:
                break
    finally:
        if task_id in log_listeners and q in log_listeners[task_id]:
            log_listeners[task_id].remove(q)
            if not log_listeners[task_id]: del log_listeners[task_id]


# --- 任务流程管理 ---
PIPELINE_STAGES = [
    "initial_upload", # 内部阶段，表示初始文件已上传
    "demucs_separation",
    "gpt_sovits_preprocess",
    "gpt_sovits_format",
    "gpt_sovits_train_sovits",
    "gpt_sovits_train_gpt",
    "gpt_sovits_inference",
    "ffmpeg_slice_process",
    "so_vits_svc_prepare_data", # 包含 resample, flist_config, hubert_f0
    "so_vits_svc_train",
    "so_vits_svc_train_diffusion", # 可选
    "so_vits_svc_inference",
    "final_mix"
]

def get_next_stage(current_stage_key):
    try:
        current_index = PIPELINE_STAGES.index(current_stage_key)
        if current_index + 1 < len(PIPELINE_STAGES):
            return PIPELINE_STAGES[current_index + 1]
    except ValueError:
        return None
    return None

def update_task_stage(task_id, new_stage_key, status="进行中"):
    if task_id in tasks:
        tasks[task_id]["current_stage_key"] = new_stage_key
        tasks[task_id]["current_stage_info"] = new_stage_key.replace("_", " ").title() # 简单的友好名称
        tasks[task_id]["status"] = status
        # 通过SSE发送阶段更新事件
        stage_update_message = {
            "type": "stage_update",
            "task_id": task_id,
            "stage_key": new_stage_key,
            "stage_info": tasks[task_id]["current_stage_info"],
            "status": status
        }
        # 特殊处理：如果所有文件已就绪，告知前端可以进入Demucs
        if new_stage_key == "initial_upload" and status == "文件就绪":
            stage_update_message["next_expected_action"] = "demucs_separation"

        log_event(task_id, stage_update_message) # 这会将字典也放入日志流

def run_pipeline_stage_thread(task_id):
    try:
        task_data = tasks[task_id]
        current_stage = task_data["current_stage_key"]
        job_params = task_data["job_params"] # 所有累积的参数

        log_event(task_id, f"线程启动：处理阶段 {current_stage}")

        if current_stage == "demucs_separation":
            original_song_path = Path(task_data["initial_files"]["original_song_filepath"])
            _stage_1_demucs(task_id, original_song_path, job_params)
        
        elif current_stage == "gpt_sovits_preprocess":
            # train.txt 预处理部分移植
            # 需要: gpt_training_voice_filepath, experiment_name
            # 输出: ASR .list 文件路径 (需要从re_preprocess_data.py的逻辑中确定)
            _stage_2a_gpt_sovits_preprocess(task_id, job_params)

        elif current_stage == "gpt_sovits_format":
            # train.txt 格式化部分移植
            # 需要: 上一步的 ASR .list 文件, 切片/降噪后的音频目录, 预训练模型路径等
            _stage_2b_gpt_sovits_format(task_id, job_params)

        elif current_stage == "gpt_sovits_train_sovits":
            _stage_2c_gpt_sovits_train_sovits(task_id, job_params)
        
        elif current_stage == "gpt_sovits_train_gpt":
            _stage_2d_gpt_sovits_train_gpt(task_id, job_params)

        elif current_stage == "gpt_sovits_inference":
            _stage_2e_gpt_sovits_inference(task_id, job_params)
            # 此阶段完成后，result_paths 中应有 gpt_generated_vocals

        elif current_stage == "ffmpeg_slice_process":
            gpt_vocals_path_str = task_data["result_paths"]["gpt_generated_vocals"]
            _stage_3_ffmpeg_slice_process(task_id, gpt_vocals_path_str, job_params)

        elif current_stage == "so_vits_svc_prepare_data":
            ffmpeg_slices_dir_str = task_data["result_paths"]["ffmpeg_processed_slices_dir"]
            _stage_4a_so_vits_svc_prepare_data(task_id, ffmpeg_slices_dir_str, job_params)

        elif current_stage == "so_vits_svc_train":
            _stage_4b_so_vits_svc_train(task_id, job_params)
            # 如果需要训练扩散模型
            if job_params.get("svc_use_diff_train", False):
                update_task_stage(task_id, "so_vits_svc_train_diffusion", status="待处理")
                return # 等待用户确认或自动进入下一扩散训练阶段
        
        elif current_stage == "so_vits_svc_train_diffusion":
            if not job_params.get("svc_use_diff_train", False): # 如果用户没选但流程到了这里，跳过
                 log_event(task_id, "跳过SVC扩散模型训练，因为用户未选择。")
            else:
                _stage_4c_so_vits_svc_train_diffusion(task_id, job_params)
        
        elif current_stage == "so_vits_svc_inference":
            demucs_vocals_path_str = task_data["result_paths"]["demucs_vocals"]
            _stage_4d_so_vits_svc_inference(task_id, demucs_vocals_path_str, job_params)

        elif current_stage == "final_mix":
            svc_vocals_path_str = task_data["result_paths"]["svc_converted_vocals"]
            demucs_no_vocals_path_str = task_data["result_paths"]["demucs_no_vocals"]
            _stage_5_final_mix(task_id, svc_vocals_path_str, demucs_no_vocals_path_str, job_params)
        
        else:
            log_event(task_id, f"错误：未知的处理阶段 {current_stage}")
            tasks[task_id]["status"] = "失败"
            tasks[task_id]["error_message"] = f"未知的处理阶段: {current_stage}"
            return

        # 如果当前阶段成功完成
        log_event(task_id, f"阶段 {current_stage} 成功完成。")
        
        next_stage = get_next_stage(current_stage)
        if next_stage:
            if next_stage == "so_vits_svc_train_diffusion" and not job_params.get("svc_use_diff_train", False):
                # 如果下一阶段是可选的扩散训练且用户未选择，则跳到再下一个阶段
                log_event(task_id, f"跳过可选阶段 {next_stage}")
                next_stage = get_next_stage(next_stage) # 获取扩散训练之后的阶段


            if next_stage: # 确保在跳过后仍有下一阶段
                update_task_stage(task_id, next_stage, status="待处理") # 等待前端触发或参数
                log_event(task_id, f"任务 {task_id} 进入下一阶段 {next_stage}，等待参数或触发。")
            else: # 所有阶段完成
                tasks[task_id]["status"] = "已完成"
                log_event(task_id, f"任务 {task_id} 所有阶段已成功完成！")
        else: # 所有阶段完成
            tasks[task_id]["status"] = "已完成"
            log_event(task_id, f"任务 {task_id} 所有阶段已成功完成！")

    except Exception as e:
        import traceback
        error_full_trace = traceback.format_exc()
        error_message = f"流程线程错误在阶段 {tasks.get(task_id, {}).get('current_stage_key', '未知')}: {str(e)}\nTrace:\n{error_full_trace}"
        log_event(task_id, error_message)
        if task_id in tasks:
            tasks[task_id]["status"] = "失败"
            tasks[task_id]["error_message"] = str(e)

# --- 各阶段具体实现 (移植 train.txt 和原有逻辑) ---

def _stage_1_demucs(task_id, original_song_path, job_params):
    log_event(task_id, f"开始对 {original_song_path.name} 进行 Demucs 处理")
    demucs_output_target_dir = PROCESSED_DIR / "separated_audio"
    demucs_model_cli_arg = job_params.get("demucs_model_name", "htdemucs")

    cmd = [
        "demucs",
        "-n", demucs_model_cli_arg,
        "-o", str(demucs_output_target_dir),
        str(original_song_path)
    ]
    if job_params.get("demucs_two_stems_vocals", True):
        cmd.extend(["--two-stems", "vocals"])

    run_command(cmd, env_name=DEMUCS_ENV, task_id=task_id, cwd=BASE_DIR)

    song_name_stem = original_song_path.stem
    vocals_path = demucs_output_target_dir / demucs_model_cli_arg / song_name_stem / "vocals.wav"
    no_vocals_path = demucs_output_target_dir / demucs_model_cli_arg / song_name_stem / "no_vocals.wav"

    if not vocals_path.exists() or not no_vocals_path.exists():
        raise FileNotFoundError(f"Demucs 输出未找到。期望人声路径: {vocals_path}, 伴奏路径: {no_vocals_path}")

    tasks[task_id].setdefault("result_paths", {})
    tasks[task_id]["result_paths"]["demucs_vocals"] = str(vocals_path)
    tasks[task_id]["result_paths"]["demucs_no_vocals"] = str(no_vocals_path)
    log_event(task_id, f"Demucs 处理完成。人声: {vocals_path}, 伴奏: {no_vocals_path}")


# --- GPT-SoVITS 阶段 (train.txt 移植) ---
def _stage_2a_gpt_sovits_preprocess(task_id, job_params):
    log_event(task_id, "GPT-SoVITS: 开始数据预处理 (re_preprocess_data.py)")
    exp_name = job_params["experiment_name"]
    
    # train.txt 中的 PROJECT_ROOT_DIR 在这里是 GPT_SOVITS_PROJECT_DIR
    # train.txt 中的 INPUT_AUDIO_DIR = "${PROJECT_ROOT_DIR}/raw_audio_data/${EXPERIMENT_NAME}"
    # 这个目录需要包含用户上传的 gpt_training_voice_filename
    gpt_training_voice_orig_path = Path(tasks[task_id]["initial_files"]["gpt_training_voice_filepath"])
    
    # 将训练语音复制到 GPT-SoVITS 项目内期望的输入位置
    # re_preprocess_data.py 的 --input_audio_dir 参数
    script_input_audio_dir = GPT_SOVITS_PROJECT_DIR / "raw_audio_data" / exp_name
    script_input_audio_dir.mkdir(parents=True, exist_ok=True)
    # 清理旧文件（如果存在）
    for f in script_input_audio_dir.glob("*"):
        if f.is_file(): f.unlink()
        elif f.is_dir(): shutil.rmtree(f)

    shutil.copy(str(gpt_training_voice_orig_path), str(script_input_audio_dir / gpt_training_voice_orig_path.name))
    log_event(task_id, f"已将GPT训练语音 {gpt_training_voice_orig_path.name} 复制到 {script_input_audio_dir}")

    # train.txt 中的 PREPROCESS_OUTPUT_BASE_DIR = "${PROJECT_ROOT_DIR}/output_preprocess_cli"
    script_output_base_dir = GPT_SOVITS_PROJECT_DIR / "output_preprocess_cli" # re_preprocess_data.py 的 --output_base_dir
    script_output_base_dir.mkdir(parents=True, exist_ok=True)

    preprocess_script_path = GPT_SOVITS_PROJECT_DIR / "tools" / "re_preprocess_data.py" # 假设在 tools/ 下
    if not preprocess_script_path.exists():
        preprocess_script_path = GPT_SOVITS_PROJECT_DIR / "re_preprocess_data.py" # 如果在根目录
        if not preprocess_script_path.exists():
            raise FileNotFoundError(f"未找到 re_preprocess_data.py 脚本于 {GPT_SOVITS_PROJECT_DIR} 或其 tools 子目录")

    cmd_preprocess = [
        "python", str(preprocess_script_path),
        "--input_audio_dir", str(script_input_audio_dir),
        "--experiment_name", exp_name,
        "--output_base_dir", str(script_output_base_dir),
        "--slice_num_processes", str(job_params.get("gpt_slice_num_processes", 4)),
    ]
    if job_params.get("gpt_run_denoise", False): # 假设前端传递此参数
        cmd_preprocess.append("--run_denoise")
    # --run_separation 暂时不处理，因为它依赖 demucs 环境，且我们已经在外部做了

    run_command(cmd_preprocess, cwd=GPT_SOVITS_PROJECT_DIR, env_name=GPT_SOVITS_ENV, task_id=task_id)

    # 确定 ASR .list 文件路径 (移植 train.txt 中的逻辑)
    # SLICED_AUDIO_DIR_FOR_ASR="${PREPROCESS_OUTPUT_BASE_DIR}/${EXPERIMENT_NAME}/2_sliced_audio"
    # DENOISED_AUDIO_DIR_FOR_ASR="${PREPROCESS_OUTPUT_BASE_DIR}/${EXPERIMENT_NAME}/3_denoised_audio"
    # ASR_OUTPUT_DIR_FROM_PREPROCESS="${PREPROCESS_OUTPUT_BASE_DIR}/${EXPERIMENT_NAME}/4_asr_output"
    
    sliced_audio_dir_for_asr = script_output_base_dir / exp_name / "2_sliced_audio"
    denoised_audio_dir_for_asr = script_output_base_dir / exp_name / "3_denoised_audio"
    asr_output_dir_from_preprocess = script_output_base_dir / exp_name / "4_asr_output"

    asr_actual_input_dir_basename = ""
    if job_params.get("gpt_run_denoise", False) and denoised_audio_dir_for_asr.exists():
        asr_actual_input_dir_basename = denoised_audio_dir_for_asr.name
        tasks[task_id]["result_paths"]["gpt_wav_dir_for_format"] = str(denoised_audio_dir_for_asr)
        log_event(task_id, f"ASR 输入目录确定为降噪后的目录: {denoised_audio_dir_for_asr.name}")
    elif sliced_audio_dir_for_asr.exists():
        asr_actual_input_dir_basename = sliced_audio_dir_for_asr.name
        tasks[task_id]["result_paths"]["gpt_wav_dir_for_format"] = str(sliced_audio_dir_for_asr)
        log_event(task_id, f"ASR 输入目录确定为切分后的目录: {sliced_audio_dir_for_asr.name}")
    else:
        raise FileNotFoundError(f"预处理后，切片或降噪音频目录 ({sliced_audio_dir_for_asr} 或 {denoised_audio_dir_for_asr}) 未找到。")


    asr_output_list_file = asr_output_dir_from_preprocess / f"{asr_actual_input_dir_basename}.list"
    if not asr_output_list_file.exists():
        raise FileNotFoundError(f"ASR .list 文件未在 {asr_output_list_file} 生成。")

    tasks[task_id].setdefault("result_paths", {})
    tasks[task_id]["result_paths"]["gpt_asr_list_file"] = str(asr_output_list_file)
    
    # train.txt 中有手动校对步骤，这里需要前端确认或自动跳过
    # 假设我们目前自动继续
    log_event(task_id, f"GPT-SoVITS 预处理完成。ASR列表文件: {asr_output_list_file}")
    # (如果需要前端确认，这里应该将任务状态设为“等待确认”，然后前端再触发下一阶段)


def _get_gpt_pretrained_model_path(version_str, model_type, project_root_dir):
    # 移植 train.txt 中根据版本选择预训练模型的逻辑
    # model_type: "s2g", "s2d", "s1"
    base_models_dir = Path(project_root_dir) / "GPT_SoVITS" / "pretrained_models"
    
    # 确保路径使用 job_params 中的 gpt_sovits_version
    # job_params['gpt_sovits_version']
    
    # 示例：
    if model_type == "s2g":
        if version_str == "v1": return str(base_models_dir / "s2G488k.pth")
        if version_str == "v2": return str(base_models_dir / "gsv-v2final-pretrained" / "s2G2333k.pth")
        if version_str in ["v3", "v4"]: return str(base_models_dir / "gsv-v4-pretrained" / "s2Gv4.pth")
    elif model_type == "s2d":
        if version_str == "v1": return str(base_models_dir / "s2D488k.pth")
        if version_str == "v2": return str(base_models_dir / "gsv-v2final-pretrained" / "s2D2333k.pth")
        return None # v3/v4 可能不需要 s2d
    elif model_type == "s1":
        if version_str == "v1": return str(base_models_dir / "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt")
        if version_str == "v2": return str(base_models_dir / "gsv-v2final-pretrained" / "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt")
        if version_str in ["v3", "v4"]: return str(base_models_dir / "s1v3.ckpt")
    
    raise ValueError(f"未知的 GPT-SoVITS 版本 '{version_str}' 或模型类型 '{model_type}' 的预训练模型路径。")


def _stage_2b_gpt_sovits_format(task_id, job_params):
    log_event(task_id, "GPT-SoVITS: 开始训练集格式化 (re_format_training_data.py)")
    exp_name = job_params["experiment_name"]
    gpt_version = job_params.get("gpt_sovits_version", "v4")

    asr_list_file_path = tasks[task_id]["result_paths"]["gpt_asr_list_file"]
    wav_dir_for_format = tasks[task_id]["result_paths"]["gpt_wav_dir_for_format"] # 来自上一步

    # train.sh 中的 FORMAT_OUTPUT_BASE_DIR = "${PROJECT_ROOT_DIR}/logs_formatted"
    script_format_output_base_dir = GPT_SOVITS_PROJECT_DIR / "logs_formatted"
    script_format_output_base_dir.mkdir(parents=True, exist_ok=True)

    format_script_path = GPT_SOVITS_PROJECT_DIR / "tools" / "re_format_training_data.py" # 假设在 tools/ 下
    if not format_script_path.exists():
        format_script_path = GPT_SOVITS_PROJECT_DIR / "re_format_training_data.py" # 如果在根目录
        if not format_script_path.exists():
             raise FileNotFoundError(f"未找到 re_format_training_data.py 脚本于 {GPT_SOVITS_PROJECT_DIR} 或其 tools 子目录")


    # 获取预训练模型路径
    default_bert_model_dir = str(GPT_SOVITS_PROJECT_DIR / "GPT_SoVITS" / "pretrained_models" / "chinese-roberta-wwm-ext-large")
    default_ssl_model_dir = str(GPT_SOVITS_PROJECT_DIR / "GPT_SoVITS" / "pretrained_models" / "chinese-hubert-base")

    # 如果 job_params 中存在该键且值不为空字符串，则使用它，否则使用默认值
    bert_model_dir_from_params = job_params.get("gpt_bert_model_dir")
    bert_model_dir = bert_model_dir_from_params if bert_model_dir_from_params else default_bert_model_dir

    ssl_model_dir_from_params = job_params.get("gpt_ssl_model_dir")
    ssl_model_dir = ssl_model_dir_from_params if ssl_model_dir_from_params else default_ssl_model_dir
    
    pretrained_s2g_path = job_params.get("gpt_s2g_path", _get_gpt_pretrained_model_path(gpt_version, "s2g", GPT_SOVITS_PROJECT_DIR))

    cmd_format = [
        "python", str(format_script_path),
        "--list_file", str(asr_list_file_path),
        "--wav_dir", str(wav_dir_for_format),
        "--experiment_name", exp_name,
        "--output_base_dir", str(script_format_output_base_dir),
        "--bert_model_dir", str(bert_model_dir),
        "--ssl_model_dir", str(ssl_model_dir),
        "--s2g_model_path", str(pretrained_s2g_path),
        "--gpu_id", job_params.get("gpt_gpu_id", "0"),
        "--gpt_sovits_version", gpt_version,
    ]
    if job_params.get("gpt_use_half_precision", True):
        cmd_format.append("--use_half_precision")

    run_command(cmd_format, cwd=GPT_SOVITS_PROJECT_DIR, env_name=GPT_SOVITS_ENV, task_id=task_id)

    # 这个路径是 re_finetune_training.py 的 --exp_data_root 参数指向的父目录的子目录
    tasks[task_id]["result_paths"]["gpt_formatted_data_root_for_finetune"] = str(script_format_output_base_dir) # 父目录
    log_event(task_id, f"GPT-SoVITS 格式化完成。格式化数据根目录: {script_format_output_base_dir}")

def _stage_2c_gpt_sovits_train_sovits(task_id, job_params):
    log_event(task_id, "GPT-SoVITS: 开始 SoVITS 模型微调")
    exp_name = job_params["experiment_name"]
    gpt_version = job_params.get("gpt_sovits_version", "v4")
    
    finetune_script_path = GPT_SOVITS_PROJECT_DIR / "tools" / "re_finetune_training.py"
    if not finetune_script_path.exists():
        finetune_script_path = GPT_SOVITS_PROJECT_DIR / "re_finetune_training.py"
        if not finetune_script_path.exists():
            raise FileNotFoundError(f"未找到 re_finetune_training.py 脚本")

    exp_data_root = tasks[task_id]["result_paths"]["gpt_formatted_data_root_for_finetune"] 
    pretrained_s2g_path = job_params.get("gpt_s2g_path", _get_gpt_pretrained_model_path(gpt_version, "s2g", GPT_SOVITS_PROJECT_DIR))

    # 基本命令参数
    sovits_train_args_base = [
        "python", str(finetune_script_path),
        "--experiment_name", exp_name,
        "--exp_data_root", str(exp_data_root),
        "--gpu_id", job_params.get("gpt_gpu_id", "0"), # 前端应该提供 gpt_gpu_id, 默认为 "0"
        "--gpt_sovits_version", gpt_version,
    ]

    # 根据条件添加 --use_half_precision
    # 假设前端 SoVITS 训练表单中有一个名为 "gpt_sovits_train_use_half_precision" 的复选框参数
    # 并且后端 api_submit_stage_params 会将其转换为布尔值 True/False
    if job_params.get("gpt_sovits_train_use_half_precision", True): # 默认为 True 如果键不存在
        sovits_train_args_base.append("--use_half_precision")

    # 添加子命令和子命令的参数
    sovits_train_args_specific = [
        "sovits", # 子命令
        "--sovits_batch_size", str(job_params.get("gpt_sovits_batch_size", 4)),
        "--sovits_total_epoch", str(job_params.get("gpt_sovits_epochs", 10)),
        "--sovits_save_every_epoch", str(job_params.get("gpt_sovits_save_every", 2)),
        "--sovits_pretrained_s2g", str(pretrained_s2g_path),
    ]

    # 合并基础参数和特定参数
    sovits_train_args = sovits_train_args_base + sovits_train_args_specific

    # 添加版本相关的其他 SoVITS 参数
    pretrained_s2d_path = _get_gpt_pretrained_model_path(gpt_version, "s2d", GPT_SOVITS_PROJECT_DIR)
    if pretrained_s2d_path and gpt_version in ["v1", "v2"]:
        sovits_train_args.extend(["--sovits_pretrained_s2d", str(pretrained_s2d_path)])
    
    if gpt_version in ["v3", "v4"]: # 对于 v3/v4 (通常是 v4)
        sovits_train_args.extend(["--sovits_lora_rank", str(job_params.get("gpt_sovits_lora_rank", "128"))])
        # 假设前端有一个 gpt_sovits_grad_ckpt_checkbox
        if job_params.get("gpt_sovits_grad_ckpt", False): # 默认为 False 如果键不存在
            sovits_train_args.append("--sovits_grad_ckpt")

    run_command(sovits_train_args, cwd=GPT_SOVITS_PROJECT_DIR, env_name=GPT_SOVITS_ENV, task_id=task_id)
    log_event(task_id, "GPT-SoVITS: SoVITS 模型微调完成。")

def _stage_2d_gpt_sovits_train_gpt(task_id, job_params):
    log_event(task_id, "GPT-SoVITS: 开始 GPT 模型微调")
    exp_name = job_params["experiment_name"]
    gpt_version = job_params.get("gpt_sovits_version", "v4")

    finetune_script_path = GPT_SOVITS_PROJECT_DIR / "tools" / "re_finetune_training.py"
    if not finetune_script_path.exists():
        finetune_script_path = GPT_SOVITS_PROJECT_DIR / "re_finetune_training.py"
        if not finetune_script_path.exists():
            raise FileNotFoundError(f"未找到 re_finetune_training.py 脚本")

    exp_data_root = tasks[task_id]["result_paths"]["gpt_formatted_data_root_for_finetune"]
    pretrained_s1_path = job_params.get("gpt_s1_path", _get_gpt_pretrained_model_path(gpt_version, "s1", GPT_SOVITS_PROJECT_DIR))

    # 基本命令参数
    gpt_train_args_base = [
        "python", str(finetune_script_path),
        "--experiment_name", exp_name,
        "--exp_data_root", str(exp_data_root),
        "--gpu_id", job_params.get("gpt_gpu_id", "0"), # 前端应该提供 gpt_gpu_id, 默认为 "0"
        "--gpt_sovits_version", gpt_version,
    ]

    if job_params.get("gpt_gpt_train_use_half_precision", True): # 默认为 True 如果键不存在
        gpt_train_args_base.append("--use_half_precision")

    # 添加子命令和子命令的参数
    gpt_train_args_specific = [
        "gpt", # 子命令
        "--gpt_batch_size", str(job_params.get("gpt_gpt_batch_size", 4)),
        "--gpt_total_epoch", str(job_params.get("gpt_gpt_epochs", 15)),
        "--gpt_save_every_epoch", str(job_params.get("gpt_gpt_save_every", 3)),
        "--gpt_pretrained_s1", str(pretrained_s1_path),
    ]
    
    # 合并基础参数和特定参数
    gpt_train_args = gpt_train_args_base + gpt_train_args_specific

    if job_params.get("gpt_if_dpo", False): # 默认为 False 如果键不存在
        gpt_train_args.append("--gpt_if_dpo")

    run_command(gpt_train_args, cwd=GPT_SOVITS_PROJECT_DIR, env_name=GPT_SOVITS_ENV, task_id=task_id)
    log_event(task_id, "GPT-SoVITS: GPT 模型微调完成。")


def _stage_2e_gpt_sovits_inference(task_id, job_params):
    log_event(task_id, "GPT-SoVITS: 开始推理")
    exp_name = job_params["experiment_name"]
    gpt_version = job_params.get("gpt_sovits_version", "v4")
    
    inference_script_path = GPT_SOVITS_PROJECT_DIR / "tools" / "re_inference.py"
    if not inference_script_path.exists():
        inference_script_path = GPT_SOVITS_PROJECT_DIR / "re_inference.py"
        if not inference_script_path.exists():
             raise FileNotFoundError(f"未找到 re_inference.py 脚本")

    # 动态查找训练好的模型路径 (移植 train.txt 逻辑)
    # GPT模型路径
    gpt_epochs = job_params.get("gpt_gpt_epochs", 15) # 与训练时一致
    trained_gpt_model_path = GPT_SOVITS_PROJECT_DIR / f"GPT_weights_{gpt_version}" / f"{exp_name}-e{gpt_epochs}.ckpt"
    if not trained_gpt_model_path.exists():
        # 尝试查找包含step的，有些脚本可能这样保存
        gpt_models_found = list((GPT_SOVITS_PROJECT_DIR / f"GPT_weights_{gpt_version}").glob(f"{exp_name}-e{gpt_epochs}*.ckpt"))
        if gpt_models_found:
            trained_gpt_model_path = max(gpt_models_found, key=lambda p: p.stat().st_mtime) # 取最新的
        else:
            raise FileNotFoundError(f"训练后的GPT模型未在 {trained_gpt_model_path} 或类似路径找到。")
    log_event(task_id, f"找到训练后的GPT模型: {trained_gpt_model_path}")


    # SoVITS模型路径
    sovits_epochs = job_params.get("gpt_sovits_epochs", 4) # 与训练时一致
    sovits_lora_rank = job_params.get("gpt_sovits_lora_rank", "128") # 与训练时一致
    
    # SoVITS_weights_v4/exp_name_e10_s*_l128.pth
    # s* 代表 save_step，需要找到最新的
    sovits_model_pattern_dir = GPT_SOVITS_PROJECT_DIR / f"SoVITS_weights_{gpt_version}"
    sovits_model_pattern = f"{exp_name}_e{sovits_epochs}_s*_l{sovits_lora_rank}.pth"
    if gpt_version not in ["v3", "v4"]: # v1/v2没有lora_rank
         sovits_model_pattern = f"{exp_name}_e{sovits_epochs}_s*.pth"

    found_sovits_models = sorted(
        list(sovits_model_pattern_dir.glob(sovits_model_pattern)),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if not found_sovits_models:
        raise FileNotFoundError(f"训练后的SoVITS模型未在 {sovits_model_pattern_dir} 找到匹配 {sovits_model_pattern} 的文件。")
    trained_sovits_model_path = found_sovits_models[0] # 最新的
    log_event(task_id, f"找到训练后的SoVITS模型: {trained_sovits_model_path}")


    # 推理所需的参考音频和文本文件路径 (这些应该在任务初始化时由用户上传并保存)
    ref_wav_for_inference = Path(tasks[task_id]["initial_files"]["gpt_reference_audio_filepath"])
    # prompt_text_for_inference = job_params["gpt_reference_text"] # 来自用户输入
    # text_for_inference_path = Path(tasks[task_id]["initial_files"]["gpt_lyrics_filepath"]) # 要合成的歌词

    # re_inference.py 的 --text_path 是要合成的文本内容
    # re_inference.py 的 --prompt_text 是参考音频对应的文本
    # re_inference.py 的 --ref_wav_path 是参考音频

    # BERT 和 SSL 模型目录
    # 修正 BERT 和 SSL 模型目录的获取逻辑
    default_bert_model_dir = str(GPT_SOVITS_PROJECT_DIR / "GPT_SoVITS" / "pretrained_models" / "chinese-roberta-wwm-ext-large")
    default_ssl_model_dir = str(GPT_SOVITS_PROJECT_DIR / "GPT_SoVITS" / "pretrained_models" / "chinese-hubert-base")

    bert_model_dir_from_params = job_params.get("gpt_bert_model_dir") 
    bert_model_dir = bert_model_dir_from_params if bert_model_dir_from_params else default_bert_model_dir

    ssl_model_dir_from_params = job_params.get("gpt_ssl_model_dir")  
    ssl_model_dir = ssl_model_dir_from_params if ssl_model_dir_from_params else default_ssl_model_dir
    # 推理输出路径
    # train.txt 中的 INFERENCE_OUTPUT_WAV_PATH = "${PROJECT_ROOT_DIR}/output_audio/${EXPERIMENT_NAME}_final_speech.wav"
    script_inference_output_wav_path = GPT_SOVITS_PROJECT_DIR / "output_audio" / f"{exp_name}_gpt_raw_speech.wav" # 避免与全局路径冲突

    infer_cmd = [
        "python", str(inference_script_path),
        "--gpt_model_path", str(trained_gpt_model_path),
        "--sovits_model_path", str(trained_sovits_model_path),
        "--bert_model_dir", str(bert_model_dir),
        "--ssl_model_dir", str(ssl_model_dir),
        "--ref_wav_path", str(ref_wav_for_inference),
        "--prompt_text", job_params["gpt_reference_text"], # 这是参考音频的文本
        "--text_path", Path(tasks[task_id]["initial_files"]["gpt_lyrics_filepath"]),  # 这是要合成的歌词文件
        "--output_wav_path", str(script_inference_output_wav_path),
        "--gpu_id", job_params.get("gpt_gpu_id", "0"),
        "--top_k", str(job_params.get("gpt_infer_top_k", 20)),
        "--temperature", str(job_params.get("gpt_infer_temp", 0.7)),
        "--speed", str(job_params.get("gpt_infer_speed", 1.0)),
        "--sample_steps", str(job_params.get("gpt_infer_sample_steps", 32)), # v4通常32，旧版可能不同
    ]
    if job_params.get("gpt_use_half_precision", True):
        infer_cmd.append("--use_half_precision")
    # 注意：re_inference.py 内部可能会根据 SoVITS 模型自动确定符号版本，
    # --gpt_sovits_version_for_symbols 参数可能不需要显式传递，除非有特殊需求。

    run_command(infer_cmd, cwd=GPT_SOVITS_PROJECT_DIR, env_name=GPT_SOVITS_ENV, task_id=task_id)

    if not script_inference_output_wav_path.exists():
        raise FileNotFoundError(f"GPT-SoVITS 推理输出音频未在 {script_inference_output_wav_path} 生成。")

    # 将推理结果移动到统一的 PROCESSED_DIR
    final_gpt_output_path = PROCESSED_DIR / "gpt_sovits_output" / f"{exp_name}_gpt_generated_vocals.wav"
    final_gpt_output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(script_inference_output_wav_path), str(final_gpt_output_path))

    tasks[task_id].setdefault("result_paths", {})
    tasks[task_id]["result_paths"]["gpt_generated_vocals"] = str(final_gpt_output_path)
    log_event(task_id, f"GPT-SoVITS 推理完成。输出: {final_gpt_output_path}")


def _stage_3_ffmpeg_slice_process(task_id, gpt_vocals_path_str, job_params):
    
    tasks[task_id]["stage"] = "FFmpeg 切片与处理"
    gpt_vocals_path = Path(gpt_vocals_path_str)
    exp_name = job_params["experiment_name"]
    log_event(task_id, f"开始对 {gpt_vocals_path.name} 进行 FFmpeg 切片与处理")

    ffmpeg_processed_slices_dir = PROCESSED_DIR / "sliced_for_svc_ffmpeg" / exp_name
    ffmpeg_processed_slices_dir.mkdir(parents=True, exist_ok=True)
    for f in ffmpeg_processed_slices_dir.glob("*.wav"): f.unlink()

    temp_segments_dir = PROCESSED_DIR / "temp_ffmpeg_segments" / exp_name
    temp_segments_dir.mkdir(parents=True, exist_ok=True)
    for f in temp_segments_dir.glob("*.wav"): f.unlink()

    log_event(task_id, "FFmpeg: 切割为初始10秒片段...")
    log_event(task_id, "FFmpeg: 切割为10秒片段 (丢弃最后不足10秒的片段)...")


    # 获取音频总时长
    cmd_get_duration = [
        FFPROBE_PATH, "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(gpt_vocals_path)
    ]
    try:
        duration_str = subprocess.check_output(cmd_get_duration, text=True, cwd=str(BASE_DIR)).strip()
        total_duration_seconds = float(duration_str)
        log_event(task_id, f"音频总时长: {total_duration_seconds:.2f} 秒")
    except Exception as e:
        log_event(task_id, f"获取音频时长失败: {e}. 无法精确丢弃最后片段，将进行标准切片。")
        total_duration_seconds = -1 # 标记为未知

    # 计算可以切出的完整10秒片段数量
    num_full_segments = 0
    if total_duration_seconds > 0:
        num_full_segments = int(total_duration_seconds // 10)
    
    if num_full_segments == 0 and total_duration_seconds > 0: # 音频不足10秒，但存在
        log_event(task_id, f"音频总时长 {total_duration_seconds:.2f} 秒，不足10秒。将处理整个文件作为单个片段。")
        # 如果音频不足10秒，将其视为单个片段。
        cmd_segment = [
            FFMPEG_PATH, "-hide_banner", "-loglevel", "error",
            "-i", str(gpt_vocals_path),
            "-ar", "44100", # 重采样
            "-ac", "1",    # 单声道
            "-c:a", "pcm_s16le", # 确保输出是 wav
            str(temp_segments_dir / "slice_0000.wav") # 只输出一个文件
        ]
    elif num_full_segments > 0:
        # 只处理能构成完整10秒片段的部分
        duration_to_process = num_full_segments * 10
        log_event(task_id, f"将处理音频的前 {duration_to_process} 秒，切分为 {num_full_segments} 个10秒片段。")
        cmd_segment = [
            FFMPEG_PATH, "-hide_banner", "-loglevel", "error",
            "-i", str(gpt_vocals_path),
            "-t", str(duration_to_process), # 限制处理时长
            "-f", "segment",
            "-segment_time", "10",
            "-ar", "44100", # 在分段时进行重采样和单声道转换
            "-ac", "1",
            "-c:a", "pcm_s16le", # 确保输出是 wav
            "-reset_timestamps", "1",
            str(temp_segments_dir / "slice_%04d.wav")
        ]
    else: # total_duration_seconds <= 0 或获取失败
        # 无法确定时长或音频为空，进行标准切片，后续逻辑处理
        log_event(task_id, "无法确定音频时长或音频为空，尝试标准10秒切片（可能产生短片段）。")
        cmd_segment = [
            FFMPEG_PATH, "-hide_banner", "-loglevel", "error",
            "-i", str(gpt_vocals_path),
            "-f", "segment",
            "-segment_time", "10",
            "-ar", "44100", # 在分段时进行重采样和单声道转换
            "-ac", "1",
            "-c:a", "pcm_s16le", # 确保输出是 wav
            "-reset_timestamps", "1",
            str(temp_segments_dir / "slice_%04d.wav")
        ]
        
    run_command(cmd_segment, task_id=task_id)


    log_event(task_id, "FFmpeg: 将切片复制到最终目录")
    processed_slice_paths = []
    initial_slices = sorted(list(temp_segments_dir.glob("slice_*.wav")))
    
    if not initial_slices:
        # 检查原始GPT输出是否存在且非空，如果存在则它本身就是唯一的“片段”
        # （这种情况对应于原始音频不足10秒且被作为单个文件处理的逻辑）
        if (temp_segments_dir / "slice_0000.wav").exists() and (temp_segments_dir / "slice_0000.wav").stat().st_size > 0:
            initial_slices = [temp_segments_dir / "slice_0000.wav"]
        elif gpt_vocals_path.exists() and gpt_vocals_path.stat().st_size > 0 and num_full_segments == 0 and total_duration_seconds > 0:
             # 这种情况可能发生在上面 cmd_segment 如果直接输出到 ffmpeg_processed_slices_dir
             # 但我们现在是先输出到 temp_segments_dir
             pass # initial_slices 已经是空的，下面会报错
        else:
            raise RuntimeError(f"FFmpeg 切片失败: 在 {temp_segments_dir} 未找到切片，且源文件 {gpt_vocals_path.name} 可能为空或切片逻辑未产生输出。")


    for i, segment_file in enumerate(initial_slices):
        output_slice_name = f"processed_slice_{i:04d}.wav"
        output_slice_path = ffmpeg_processed_slices_dir / output_slice_name
        
        # 由于在 segment 命令中已经处理了采样率和声道，这里可以直接复制
        # 或者如果想确保格式，可以再用 ffmpeg 转一次，但通常复制即可
        try:
            shutil.copy(str(segment_file), str(output_slice_path))
            processed_slice_paths.append(str(output_slice_path))
        except Exception as e:
            log_event(task_id, f"复制切片 {segment_file.name} 失败: {e}")


    if temp_segments_dir.exists() and temp_segments_dir != ffmpeg_processed_slices_dir:
        try:
            shutil.rmtree(temp_segments_dir)
            log_event(task_id, f"已清理临时片段目录: {temp_segments_dir}")
        except OSError as e:
            log_event(task_id, f"清理临时片段目录 {temp_segments_dir} 失败: {e}")

    if not processed_slice_paths:
        # 增加对 num_full_segments == 0 且 total_duration_seconds > 0 的情况的检查
        # 确保如果原始文件被作为单个片段处理，它被正确识别
        if num_full_segments == 0 and total_duration_seconds > 0 and (ffmpeg_processed_slices_dir / "processed_slice_0000.wav").exists():
            log_event(task_id, "单个短音频片段已处理。")
        else:
            raise RuntimeError("FFmpeg 处理失败: 没有片段被处理或复制到最终目录。")


    tasks[task_id].setdefault("result_paths", {})
    tasks[task_id]["result_paths"]["ffmpeg_processed_slices_dir"] = str(ffmpeg_processed_slices_dir)
    log_event(task_id, f"FFmpeg 切片与处理完成。输出目录: {ffmpeg_processed_slices_dir}")


# --- so-vits-svc 阶段 ---
def _stage_4a_so_vits_svc_prepare_data(task_id, ffmpeg_slices_dir_str, job_params):
    log_event(task_id, "so-vits-svc: 开始数据准备")
    
    ffmpeg_slices_dir = Path(ffmpeg_slices_dir_str)
    exp_name = job_params["experiment_name"]
    svc_speaker_name = job_params.get("svc_speaker_name", f"{exp_name}_svc_spk")
    log_event(task_id, f"SVC 数据准备，说话人 '{svc_speaker_name}'。切片来源: {ffmpeg_slices_dir.name}")

    # 1. 复制音频文件
    svc_dataset_raw_speaker_dir = SOVITS_SVC_PROJECT_DIR / "dataset_raw" / svc_speaker_name
    svc_dataset_raw_speaker_dir.mkdir(parents=True, exist_ok=True)
    # 清理旧文件
    for f in svc_dataset_raw_speaker_dir.glob("*.wav"): f.unlink()
    
    source_slices = list(ffmpeg_slices_dir.glob("*.wav"))
    if not source_slices:
        raise FileNotFoundError(f"在 {ffmpeg_slices_dir} 中未找到用于 SVC 训练的 FFmpeg 处理后切片。")
    for slice_f in source_slices:
        shutil.copy(str(slice_f), str(svc_dataset_raw_speaker_dir / slice_f.name))
    log_event(task_id, f"SVC: 已复制 {len(source_slices)} 个切片到 {svc_dataset_raw_speaker_dir}")

    cwd_svc = SOVITS_SVC_PROJECT_DIR
    
    # 2. 运行 resample.py
    cmd_resample = ["python", "resample.py", "--skip_loudnorm"]
    run_command(cmd_resample, cwd=cwd_svc, env_name=SOVITS_SVC_ENV, task_id=task_id)

    # 3. 运行 preprocess_flist_config.py
    svc_speech_encoder = job_params.get("svc_speech_encoder", "vec768l12")
    cmd_flist = ["python", "preprocess_flist_config.py", "--speech_encoder", svc_speech_encoder]
    # --vol_aug 参数会自动在 config.json 中设置 vol_embedding: true
    if job_params.get("svc_vol_aug", False):
        cmd_flist.append("--vol_aug")
    run_command(cmd_flist, cwd=cwd_svc, env_name=SOVITS_SVC_ENV, task_id=task_id)
    
    # 4. 运行 preprocess_hubert_f0.py
    svc_f0_predictor = job_params.get("svc_f0_predictor", "rmvpe")
    cmd_hubert_f0 = ["python", "preprocess_hubert_f0.py", "--f0_predictor", svc_f0_predictor]
    # --use_diff 参数会为扩散模型准备数据
    if job_params.get("svc_use_diff_train", False):
        cmd_hubert_f0.append("--use_diff")

    run_command(cmd_hubert_f0, cwd=cwd_svc, env_name=SOVITS_SVC_ENV, task_id=task_id)
    
    log_event(task_id, "so-vits-svc: 数据准备完成。")

def _stage_4b_so_vits_svc_train(task_id, job_params):
    log_event(task_id, "so-vits-svc: 开始主模型训练")
    cwd_svc = SOVITS_SVC_PROJECT_DIR    
    
    # --- 在训练前修改 config.json ---
    config_json_path = cwd_svc / "configs" / "config.json"
    if config_json_path.exists():
        with open(config_json_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        config_updated = False
        
        # 确保 "train" 键存在
        if "train" not in config_data:
            config_data["train"] = {}
        # 修改训练轮数 (epochs)
        if "svc_train_epochs" in job_params and job_params["svc_train_epochs"]:
            config_data["train"]["epochs"] = int(job_params["svc_train_epochs"])
            config_updated = True
        # 修改批大小 (batch_size)
        if "svc_batch_size" in job_params and job_params["svc_batch_size"]:
            config_data["train"]["batch_size"] = int(job_params["svc_batch_size"])
            config_updated = True            
        # 修改保留检查点数 (keep_ckpts)
        if "svc_keep_ckpts" in job_params and job_params["svc_keep_ckpts"] is not None:
             config_data["train"]["keep_ckpts"] = int(job_params["svc_keep_ckpts"])
             config_updated = True             
        # 修改全内存模式 (all_in_mem)
        if "svc_all_in_mem" in job_params: # Key is "svc_all_in_mem" after checkbox processing
            config_data["train"]["all_in_mem"] = job_params["svc_all_in_mem"]
            config_updated = True   
        if config_updated:
            with open(config_json_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            log_event(task_id, f"SVC主模型: 已使用前端自定义参数更新 {config_json_path.name}。")
    else:
        log_event(task_id, f"警告: 未找到 config.json 文件于 {config_json_path}。无法应用自定义训练参数。")

    # --- 运行训练命令 ---
    cmd_train = ["python", "train.py", "-c", "configs/config.json", "-m", "44k"]    
    run_command(cmd_train, cwd=cwd_svc, env_name=SOVITS_SVC_ENV, task_id=task_id)
    log_event(task_id, "so-vits-svc: 主模型训练完成。")

def _stage_4c_so_vits_svc_train_diffusion(task_id, job_params):
    log_event(task_id, "so-vits-svc: 开始扩散模型训练")
    cwd_svc = SOVITS_SVC_PROJECT_DIR

    # --- 在训练前修改 diffusion.yaml ---
    diffusion_yaml_path = cwd_svc / "configs" / "diffusion.yaml"
    if diffusion_yaml_path.exists():
        with open(diffusion_yaml_path, 'r', encoding='utf-8') as f:
            diff_config_data = yaml.safe_load(f)
        
        config_updated = False
        
        # 安全地访问和创建嵌套字典
        if "model" not in diff_config_data: diff_config_data["model"] = {}
        if "train" not in diff_config_data: diff_config_data["train"] = {}
        # 修改音频切片时长 (duration)
        if "svc_diff_train_duration" in job_params and job_params["svc_diff_train_duration"]:
             # diffusion.yaml 中 duration 路径是 model.duration
             # 根据您的 diffusion.txt 它是 data.duration
             if "data" not in diff_config_data: diff_config_data["data"] = {}
             diff_config_data["data"]["duration"] = int(job_params["svc_diff_train_duration"])
             config_updated = True
        # 修改批大小 (batch_size)
        if "svc_diff_batch_size" in job_params and job_params["svc_diff_batch_size"]:
             diff_config_data["train"]["batch_size"] = int(job_params["svc_diff_batch_size"])
             config_updated = True
        # 修改训练轮数 (epochs)
        if "svc_diff_train_epochs" in job_params and job_params["svc_diff_train_epochs"]:
                diff_config_data["train"]["epochs"] = int(job_params["svc_diff_train_epochs"])
                config_updated = True
        # 修改总步数 (timesteps)
        if "svc_diff_timesteps" in job_params and job_params["svc_diff_timesteps"]:
             # timesteps 路径是 model.timesteps
             diff_config_data["model"]["timesteps"] = int(job_params["svc_diff_timesteps"])
             config_updated = True      
        # 修改最大训练步数 (k_step_max)
        
        if "svc_diff_k_step_max" in job_params and job_params["svc_diff_k_step_max"] is not None:
             # k_step_max 路径是 model.k_step_max
             diff_config_data["model"]["k_step_max"] = int(job_params["svc_diff_k_step_max"])
             config_updated = True
        # 修改缓存数据 (cache_all_data)
        if "svc_diff_cache_data" in job_params: 
            # cache_all_data 路径是 train.cache_all_data
            diff_config_data["train"]["cache_all_data"] = bool(job_params["svc_diff_cache_data"])
            config_updated = True
        if config_updated:
            with open(diffusion_yaml_path, 'w', encoding='utf-8') as f:
                # 使用 sort_keys=False 保持原始文件顺序
                yaml.dump(diff_config_data, f, allow_unicode=True, sort_keys=False)
            log_event(task_id, f"SVC扩散模型: 已使用前端自定义参数更新 {diffusion_yaml_path.name}。")
    else:
        log_event(task_id, f"警告: 未找到 diffusion.yaml 文件于 {diffusion_yaml_path}。无法应用自定义参数。")

    # --- 运行训练命令 --- 
    cmd_train_diff = ["python", "train_diff.py", "-c", "configs/diffusion.yaml"]
    run_command(cmd_train_diff, cwd=cwd_svc, env_name=SOVITS_SVC_ENV, task_id=task_id)
    log_event(task_id, "so-vits-svc: 扩散模型训练完成。")
    
def _stage_4d_so_vits_svc_inference(task_id, demucs_vocals_path_str, job_params):
    log_event(task_id, "so-vits-svc: 开始推理")

    cwd_svc = SOVITS_SVC_PROJECT_DIR
    exp_name = job_params["experiment_name"] 
    svc_speaker_name = job_params.get("svc_speaker_name", f"{exp_name}_svc_spk")

    model_dir_svc = cwd_svc / "logs" / "44k"
    g_models = list(model_dir_svc.glob("G_*.pth"))
    if not g_models: raise FileNotFoundError("SVC: 训练后未找到生成器模型 (G_*.pth)。")
    latest_g_model = max(g_models, key=lambda p: p.stat().st_mtime)
    svc_model_path = latest_g_model
    svc_config_path = cwd_svc / "configs" / "config.json"

    demucs_vocals_path = Path(demucs_vocals_path_str)
    raw_infer_dir_svc = cwd_svc / "raw"
    raw_infer_dir_svc.mkdir(exist_ok=True)
    for f in raw_infer_dir_svc.glob("*.*"): f.unlink()
    shutil.copy(str(demucs_vocals_path), str(raw_infer_dir_svc / demucs_vocals_path.name))

    cmd_infer = [
        "python", "inference_main.py",
        "-m", str(svc_model_path),
        "-c", str(svc_config_path),
        "-n", demucs_vocals_path.name,
        "-t", str(job_params.get("svc_pitch_adjust", 0)),
        "-s", svc_speaker_name,
        "-f0p", job_params.get("svc_f0_predictor_infer", job_params.get("svc_f0_predictor", "rmvpe")),
    ]
    
    if job_params.get("svc_use_diff_infer", False) and job_params.get("svc_use_diff_train", False):
        diff_model_dir_svc = model_dir_svc / "diffusion"
        diff_models = list(diff_model_dir_svc.glob("model_*.pt"))
        if diff_models:
            latest_diff_model = max(diff_models, key=lambda p: p.stat().st_mtime)
            cmd_infer.extend([
                "--shallow_diffusion",
                "-dm", str(latest_diff_model),
                "-dc", str(cwd_svc / "configs" / "diffusion.yaml"),
                "-ks", str(job_params.get("svc_k_step_infer", 100)),
            ])
        else:
            log_event(task_id, "SVC: 请求使用浅扩散推理，但未找到扩散模型。")

    run_command(cmd_infer, cwd=cwd_svc, env_name=SOVITS_SVC_ENV, task_id=task_id)

    # --- 关键修改部分开始 ---
    svc_results_dir = cwd_svc / "results"
    input_stem = demucs_vocals_path.stem # 例如: 'vocals'
    

    # 这个模式可以成功匹配 'vocals.wav_0key_speaker_07_sovits_rmvpe.flac'
    glob_pattern = f"{input_stem}*{svc_speaker_name}*.*"
    log_event(task_id, f"SVC: 正在使用模式 '{glob_pattern}' 在目录 {svc_results_dir} 中搜索输出文件...")

    found_infer_outputs = sorted(
        list(svc_results_dir.glob(glob_pattern)),
        key=lambda p: p.stat().st_mtime,
        reverse=True  # 最新的文件排在最前面
    )

    if not found_infer_outputs:
        raise FileNotFoundError(f"SVC: 在 {svc_results_dir} 中未找到匹配模式 '{glob_pattern}' 的推理输出。请检查该目录下的实际文件名。")
    
    # 使用找到的最新文件
    svc_inferred_vocal_path = found_infer_outputs[0]
    log_event(task_id, f"SVC: 成功找到推理输出文件: {svc_inferred_vocal_path.name}")
    
    final_svc_output_path = PROCESSED_DIR / "svc_output" / f"{exp_name}_svc_converted_vocals.wav"
    final_svc_output_path.parent.mkdir(parents=True, exist_ok=True)
    # --- 关键修改部分结束 ---

    # 注意：这里我们将文件移动并重命名为统一的 .wav 格式，以便下游处理（如最终混音）
    # FFmpeg 可以自动处理 .flac 到 .wav 的转换
    if svc_inferred_vocal_path.suffix.lower() != ".wav":
        log_event(task_id, f"检测到输出为非wav格式 ({svc_inferred_vocal_path.suffix})，将使用ffmpeg转换为wav。")
        cmd_convert = [
            FFMPEG_PATH, "-hide_banner", "-loglevel", "error",
            "-i", str(svc_inferred_vocal_path),
            "-c:a", "pcm_s16le", # 标准 wav 编码
            str(final_svc_output_path)
        ]
        run_command(cmd_convert, task_id=task_id)
        # 清理原始文件
        svc_inferred_vocal_path.unlink()
    else:
        shutil.move(str(svc_inferred_vocal_path), str(final_svc_output_path))


    tasks[task_id].setdefault("result_paths", {})
    tasks[task_id]["result_paths"]["svc_converted_vocals"] = str(final_svc_output_path)
    log_event(task_id, "so-vits-svc: 推理完成。")

def _stage_5_final_mix(task_id, svc_vocals_path_str, demucs_no_vocals_path_str, job_params):
    # ... (与之前版本类似)
    log_event(task_id, "开始最终混音")
    svc_vocals = Path(svc_vocals_path_str)
    no_vocals_bg = Path(demucs_no_vocals_path_str)
    exp_name = job_params["experiment_name"]
    final_song_path = OUTPUT_DIR / "final_songs" / f"{exp_name}_final_cover.wav"
    
    vocals_vol = job_params.get("final_mix_vocals_vol", 1.0)
    bg_vol = job_params.get("final_mix_bg_vol", 1.0)

    cmd_mix = [
        FFMPEG_PATH, "-hide_banner", "-loglevel", "error",
        "-i", str(svc_vocals),
        "-i", str(no_vocals_bg),
        "-filter_complex",
        f"[0:a]volume={vocals_vol}[vocalstream];[1:a]volume={bg_vol}[bgstream];[vocalstream][bgstream]amix=inputs=2:duration=longest",
        "-c:a", "pcm_s16le",
        str(final_song_path)
    ]
    run_command(cmd_mix, task_id=task_id)

    tasks[task_id].setdefault("result_paths", {})
    tasks[task_id]["result_paths"]["final_song"] = str(final_song_path)
    log_event(task_id, f"最终歌曲已混音: {final_song_path}")

@app.route('/initiate_task', methods=['POST'])
def api_initiate_task():
    task_id = str(uuid.uuid4())
    
    try:
        if 'experiment_name' not in request.form or not request.form['experiment_name']:
            return jsonify({"error": "缺少实验名称"}), 400
        
        exp_name = secure_filename(request.form['experiment_name'])
        
        initial_files_info = {}
        required_files_map = {
            "original_song_file": (INPUT_DIR / "songs_for_separation", "original_song_filepath"),
            "gpt_training_voice_file": (INPUT_DIR / "gpt_sovits_training_voice", "gpt_training_voice_filepath"),
            "gpt_reference_audio_file": (INPUT_DIR / "gpt_sovits_reference", "gpt_reference_audio_filepath"),
            "gpt_lyrics_file": (INPUT_DIR / "gpt_sovits_lyrics", "gpt_lyrics_filepath"),
        }
        for form_field, (target_dir, path_key) in required_files_map.items():
            if form_field not in request.files:
                return jsonify({"error": f"缺少必需文件: {form_field}"}), 400
            file = request.files[form_field]
            if file.filename == '':
                return jsonify({"error": f"未为 {form_field} 选择文件"}), 400
            
            filename = secure_filename(file.filename)
            # 为每个任务创建独立的子目录存放上传文件，避免文件名冲突
            task_upload_subdir = target_dir / task_id 
            task_upload_subdir.mkdir(parents=True, exist_ok=True)
            save_path = task_upload_subdir / filename
            file.save(str(save_path))
            initial_files_info[path_key] = str(save_path) # 存储绝对路径

        tasks[task_id] = {
            "status": "文件就绪", 
            "log": [], 
            "result_paths": {}, 
            "job_params": {"experiment_name": exp_name}, # 初始参数
            "initial_files": initial_files_info,
            "current_stage_key": "initial_upload", # 内部标记阶段
            "current_stage_info": "初始文件已上传"
        }
        log_event(task_id, f"任务 {task_id} 已创建，实验名称: {exp_name}。初始文件已上传。")
        update_task_stage(task_id, "initial_upload", status="文件就绪") # 发送SSE更新
        
        # 告知前端下一个期望的表单
        return jsonify({
            "task_id": task_id, 
            "status": "文件就绪", 
            "stage_info": "初始文件已上传",
            "next_expected_stage_key": "demucs_separation" # 下一步是Demucs
        })

    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        log_event(task_id, f"创建任务时发生意外错误: {str(e)}\n{tb_str}")
        return jsonify({"error": f"服务器错误: {str(e)}"}), 500


@app.route('/submit_stage_params/<task_id>/<stage_key>', methods=['POST'])
def api_submit_stage_params(task_id, stage_key):
    print(f"--- SSH TUNNEL HIT: /submit_stage_params/{task_id}/{stage_key} with form: {request.form.to_dict()} ---")
    if task_id not in tasks:
        return jsonify({"error": "任务 ID 未找到"}), 404
    
    if stage_key not in PIPELINE_STAGES:
        return jsonify({"error": f"无效的阶段标识: {stage_key}"}), 400

    task_data = tasks[task_id]
    # 简单校验，实际应用中可能需要更复杂的逻辑判断当前是否期望此阶段的参数
    # if task_data.get("current_stage_key") != stage_key and task_data.get("status") != "待处理":
    #     return jsonify({"error": f"任务当前不期望提交 {stage_key} 阶段的参数。"}), 400

    try:
        # 合并新提交的参数到 job_params
        # 注意：这里简单地用新参数覆盖旧参数（如果同名）
        # 对于文件，文件名已经在 initial_files 中，这里主要是接收文本参数
        for key, value in request.form.items():
            # 安全处理，例如转换特定参数类型
            if key.endswith("_checkbox"): # 假设复选框参数以 _checkbox 结尾
                task_data["job_params"][key.replace("_checkbox","")] = True if value.lower() == 'on' else False
            elif key in ["gpt_slice_num_processes", "gpt_sovits_batch_size", "gpt_sovits_epochs", "svc_pitch_adjust"]: # 数值型
                try:
                    task_data["job_params"][key] = int(value) if value else job_params.get(key) # 保留默认值如果为空
                except ValueError:
                    task_data["job_params"][key] = float(value) if value else job_params.get(key)
            else:
                 task_data["job_params"][key] = value

        log_event(task_id, f"收到阶段 {stage_key} 的参数: {request.form.to_dict()}")
        update_task_stage(task_id, stage_key, status="排队中") # 标记为排队中，准备执行

        # 启动对应阶段的处理线程
        thread = threading.Thread(target=run_pipeline_stage_thread, args=(task_id,))
        thread.daemon = True
        thread.start()

        return jsonify({
            "task_id": task_id,
            "status": "排队中",
            "stage_info": task_data["current_stage_info"],
            "message": f"阶段 {stage_key} 的处理已加入队列。"
        })

    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        log_event(task_id, f"提交阶段 {stage_key} 参数时出错: {str(e)}\n{tb_str}")
        task_data["status"] = "失败"
        task_data["error_message"] = str(e)
        return jsonify({"error": f"服务器错误: {str(e)}"}), 500


@app.route('/stream_logs/<task_id>')
def api_stream_logs(task_id):
    # ... (与之前版本相同)
    if task_id not in tasks:
        return "任务 ID 未找到。", 404
    return Response(stream_with_context(sse_stream(task_id)), mimetype='text/event-stream')

@app.route('/task_status/<task_id>')
def api_task_status(task_id):
    # ... (与之前版本相同)
    if task_id not in tasks:
        return jsonify({"error": "任务 ID 未找到"}), 404
    
    task_info = tasks[task_id].copy()
    task_info.pop("process_popen", None) 
    task_info.pop("log", None)
    task_info.pop("initial_files", None) # 这些是内部路径，不直接给前端
    task_info.pop("job_params", None) # 参数可能很多，不适合状态查询

    return jsonify(task_info)


@app.route('/download_file')
def download_file_generic():
    # ... (与之前版本相同)
    file_path_str = request.args.get('path')
    if not file_path_str:
        return "缺少 'path' 参数", 400
    
    resolved_path = Path(file_path_str).resolve()
    if not (str(resolved_path).startswith(str(OUTPUT_DIR.resolve())) or \
            str(resolved_path).startswith(str(PROCESSED_DIR.resolve()))):
        return "禁止访问此路径。", 403

    if not resolved_path.is_file():
        return "文件未找到或不是一个文件。", 404
    
    try:
        return send_file(str(resolved_path), as_attachment=True, download_name=resolved_path.name)
    except Exception as e:
        app.logger.error(f"发送文件 {resolved_path} 时出错: {e}")
        return f"发送文件时出错: {e}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6006, debug=True, threaded=True)