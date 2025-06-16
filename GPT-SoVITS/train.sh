#!/bin/bash

# ==============================================================================
# GPT-SoVITS 全流程自动化 Bash 脚本 (示例)
# ==============================================================================
# 请在使用前仔细阅读脚本内的注释，并根据您的实际情况修改所有路径和参数！
# ==============================================================================

# --- 用户可配置变量 ---

# 项目根目录 (假设此bash脚本与Python脚本位于同一项目根目录下)
PROJECT_ROOT_DIR=$(pwd) # 或者硬编码: "/path/to/your/GPT-SoVITS"

# Python解释器路径 (如果不在PATH中或需要特定版本的Python)
PYTHON_EXECUTABLE="python" # 如果python在PATH中
#PYTHON_EXECUTABLE="/path/to/your/conda/envs/gptsovits/bin/python" # 例如conda环境

# 实验名称 (请为您的项目选择一个唯一的名称)
EXPERIMENT_NAME="exp01"

# 原始音频数据目录 (包含待处理的 .wav, .mp3 等文件)
INPUT_AUDIO_DIR="${PROJECT_ROOT_DIR}/raw_audio_data/${EXPERIMENT_NAME}" # 示例路径

# 预处理步骤的输出根目录
PREPROCESS_OUTPUT_BASE_DIR="${PROJECT_ROOT_DIR}/output_preprocess_cli"

# 格式化数据的输出根目录 (将包含 logs/<EXPERIMENT_NAME>)
FORMAT_OUTPUT_BASE_DIR="${PROJECT_ROOT_DIR}/logs_formatted" # 脚本内默认是 "logs"

# 训练后模型权重的输出根目录 (脚本内会根据版本附加 _v1, _v2 等)
# SOVITS_WEIGHTS_BASE_DIR="${PROJECT_ROOT_DIR}/SoVITS_weights" # 由finetune脚本内部处理
# GPT_WEIGHTS_BASE_DIR="${PROJECT_ROOT_DIR}/GPT_weights"     # 由finetune脚本内部处理

# 推理输出音频的保存路径
INFERENCE_OUTPUT_WAV_PATH="${PROJECT_ROOT_DIR}/output_audio/${EXPERIMENT_NAME}_final_speech.wav"
# 待生成的文本文件路径
TEXT_FOR_INFERENCE_PATH="${INPUT_AUDIO_DIR}/reference/reference_9.txt"
# 用于推理的参考音频/内容
REF_WAV_FOR_INFERENCE="${INPUT_AUDIO_DIR}/reference/reference.wav" 
PROMPT_TEXT_FOR_INFERENCE="在六一国际儿童节到来之际,习近平总书记给江苏省淮安市"

# GPU ID (用于所有需要GPU的步骤)
GPU_ID_TO_USE="0"

# 是否使用半精度 (true/false) - 对于action='store_true'的参数，存在即为True
USE_HALF_PRECISION_FLAG="--use_half_precision" # 如果要用半精度，取消注释此行
# USE_HALF_PRECISION_FLAG="" # 如果不用半精度

# GPT-SoVITS 版本 (影响符号集、部分配置和脚本选择)
# v1, v2, v4 (v3的SoVITS训练使用v4脚本，GPT用v2配置)
GPT_SOVITS_VERSION="v4"

# 预训练模型路径 (请根据您下载和存放的位置修改)
# BERT 模型目录
BERT_MODEL_DIR="${PROJECT_ROOT_DIR}/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
# SSL (Hubert/ContentVec) 模型目录
SSL_MODEL_DIR="${PROJECT_ROOT_DIR}/GPT_SoVITS/pretrained_models/chinese-hubert-base"
# SoVITS-G 模型 (用于语义token提取 和 SoVITS训练底模)
# 根据 GPT_SOVITS_VERSION 选择合适的底模
PRETRAINED_S2G_PATH=""
if [ "$GPT_SOVITS_VERSION" == "v1" ]; then
    PRETRAINED_S2G_PATH="${PROJECT_ROOT_DIR}/GPT_SoVITS/pretrained_models/s2G488k.pth"
elif [ "$GPT_SOVITS_VERSION" == "v2" ]; then
    PRETRAINED_S2G_PATH="${PROJECT_ROOT_DIR}/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
elif [ "$GPT_SOVITS_VERSION" == "v3" ] || [ "$GPT_SOVITS_VERSION" == "v4" ]; then # v3也用v4的s2G底模
    PRETRAINED_S2G_PATH="${PROJECT_ROOT_DIR}/GPT_SoVITS/pretrained_models/gsv-v4-pretrained/s2Gv4.pth"
else
    echo "错误: 无效的 GPT_SOVITS_VERSION: ${GPT_SOVITS_VERSION}"
    exit 1
fi
# SoVITS-D 模型 (主要用于v1/v2 SoVITS训练)
PRETRAINED_S2D_PATH=""
if [ "$GPT_SOVITS_VERSION" == "v1" ]; then
    PRETRAINED_S2D_PATH="${PROJECT_ROOT_DIR}/GPT_SoVITS/pretrained_models/s2D488k.pth"
elif [ "$GPT_SOVITS_VERSION" == "v2" ]; then
    PRETRAINED_S2D_PATH="${PROJECT_ROOT_DIR}/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth"
fi
# GPT (s1) 模型 (用于GPT训练底模)
PRETRAINED_S1_PATH=""
if [ "$GPT_SOVITS_VERSION" == "v1" ]; then
    PRETRAINED_S1_PATH="${PROJECT_ROOT_DIR}/GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
elif [ "$GPT_SOVITS_VERSION" == "v2" ]; then
    PRETRAINED_S1_PATH="${PROJECT_ROOT_DIR}/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
elif [ "$GPT_SOVITS_VERSION" == "v3" ] || [ "$GPT_SOVITS_VERSION" == "v4" ]; then # v3/v4 GPT使用s1v3.ckpt作为底模
    PRETRAINED_S1_PATH="${PROJECT_ROOT_DIR}/GPT_SoVITS/pretrained_models/s1v3.ckpt"
fi


# 训练参数 (示例)
SOVITS_BATCH_SIZE=4
SOVITS_EPOCHS=10
SOVITS_SAVE_EVERY=2
SOVITS_LORA_RANK="128" # 仅v3/v4

GPT_BATCH_SIZE=4
GPT_EPOCHS=15
GPT_SAVE_EVERY=3

# 推理参数 (示例)
INFER_TOP_K=20
INFER_TEMP=0.7
INFER_SPEED=1.0
INFER_SAMPLE_STEPS=32 # v4, v3可能用32

# --- 脚本执行 ---

# 函数：检查上一个命令是否成功
check_status() {
    if [ $? -ne 0 ]; then
        echo "错误: 上一步骤执行失败。脚本中止。"
        exit 1
    fi
}

echo "============================================================"
echo "GPT-SoVITS 全流程脚本启动"
echo "实验名称: ${EXPERIMENT_NAME}"
echo "项目根目录: ${PROJECT_ROOT_DIR}"
echo "Python解释器: ${PYTHON_EXECUTABLE}"
echo "GPU ID: ${GPU_ID_TO_USE}"
echo "GPT-SoVITS 版本: ${GPT_SOVITS_VERSION}"
echo "使用半精度: ${USE_HALF_PRECISION_FLAG:-"否"}"
echo "============================================================"
sleep 2

# 确保输出目录存在
mkdir -p "${PREPROCESS_OUTPUT_BASE_DIR}"
mkdir -p "${FORMAT_OUTPUT_BASE_DIR}"
mkdir -p "$(dirname "${INFERENCE_OUTPUT_WAV_PATH}")"
mkdir -p "$(dirname "${TEXT_FOR_INFERENCE_PATH}")" # 确保推理文本的目录存在

# 步骤 1: 数据预处理 (re_preprocess_data.py)
# 注意: re_preprocess_data.py 假设在 tools/ 目录下，如果移动到根目录，路径需相应修改
# 假设已移动到根目录
PREPROCESS_SCRIPT_PATH="${PROJECT_ROOT_DIR}/re_preprocess_data.py"
# 如果在 tools/ 下: PREPROCESS_SCRIPT_PATH="${PROJECT_ROOT_DIR}/tools/re_preprocess_data.py"

echo -e "\n>>> 阶段 1: 数据预处理 (re_preprocess_data.py) <<<"
${PYTHON_EXECUTABLE} "${PREPROCESS_SCRIPT_PATH}" \
    --input_audio_dir "${INPUT_AUDIO_DIR}" \
    --experiment_name "${EXPERIMENT_NAME}" \
    --output_base_dir "${PREPROCESS_OUTPUT_BASE_DIR}" \
    --slice_num_processes 4 \
    # --run_denoise # 如果需要降噪，取消此行注释
    # --run_separation --separation_cmd_template "demucs --two-stems=vocals -o \"{output}\" \"{input}\"" # 如果需要人声分离

check_status

# --- 确定ASR List文件的准确路径 ---
# 1. 确定ASR工具实际接收的输入目录是什么
ASR_ACTUAL_INPUT_DIR=""
SLICED_AUDIO_DIR_FOR_ASR="${PREPROCESS_OUTPUT_BASE_DIR}/${EXPERIMENT_NAME}/2_sliced_audio"
DENOISED_AUDIO_DIR_FOR_ASR="${PREPROCESS_OUTPUT_BASE_DIR}/${EXPERIMENT_NAME}/3_denoised_audio"

if [ -d "${DENOISED_AUDIO_DIR_FOR_ASR}" ]; then # 并且在 PREPROCESS_ARGS 中指定了 --run_denoise
    # 如果你确定调用 re_preprocess_data.py 时 --run_denoise 被激活了，可以用这个
    # 或者，如果 --run_denoise 是由 bash 脚本变量控制的，这里也用那个变量
    # 例如: if [ "$USER_CHOICE_RUN_DENOISE" = true ]; then
    ASR_ACTUAL_INPUT_DIR="${DENOISED_AUDIO_DIR_FOR_ASR}"
    echo "ASR输入目录被确定为降噪后的目录: ${ASR_ACTUAL_INPUT_DIR}"
else
    ASR_ACTUAL_INPUT_DIR="${SLICED_AUDIO_DIR_FOR_ASR}"
    echo "ASR输入目录被确定为切分后的目录: ${ASR_ACTUAL_INPUT_DIR}"
fi

# 2. 获取该目录的 basename
ASR_INPUT_DIR_BASENAME=$(basename "${ASR_ACTUAL_INPUT_DIR}")

# 3. 构建 .list 文件路径
# ASR的输出目录在 re_preprocess_data.py 中是 <output_base_dir>/<experiment_name>/4_asr_output/
ASR_OUTPUT_DIR_FROM_PREPROCESS="${PREPROCESS_OUTPUT_BASE_DIR}/${EXPERIMENT_NAME}/4_asr_output"
ASR_OUTPUT_LIST_FILE_CONSTRUCTED="${ASR_OUTPUT_DIR_FROM_PREPROCESS}/${ASR_INPUT_DIR_BASENAME}.list"


echo -e "\n------------------------------------------------------------"
echo "数据预处理完成。ASR生成的标注列表文件预计位于:"
echo "${ASR_OUTPUT_LIST_FILE_CONSTRUCTED}" # 使用新构造的路径
echo "请在继续之前，手动检查并校对这个 .list 文件中的文本内容。"
read -p "校对完成后，按 Enter键 继续执行训练集格式化..."
echo "------------------------------------------------------------"

# 步骤 2: 训练集格式化 (re_format_training_data.py)
FORMAT_SCRIPT_PATH="${PROJECT_ROOT_DIR}/re_format_training_data.py"
FORMATTED_DATA_DIR_FULL="${FORMAT_OUTPUT_BASE_DIR}/${EXPERIMENT_NAME}" # 格式化数据实际存放目录

# ^^^ 注意: wav_dir 应该是切分后、可能降噪后的音频目录
# 如果运行了降噪，则是 PREPROCESS_OUTPUT_BASE_DIR/${EXPERIMENT_NAME}/3_denoised_audio
echo -e "\n>>> 阶段 2: 训练集格式化 (re_format_training_data.py) <<<"
${PYTHON_EXECUTABLE} "${FORMAT_SCRIPT_PATH}" \
    --list_file "${ASR_OUTPUT_LIST_FILE_CONSTRUCTED}" \
    --wav_dir "${PREPROCESS_OUTPUT_BASE_DIR}/${EXPERIMENT_NAME}/2_sliced_audio" \
    --experiment_name "${EXPERIMENT_NAME}" \
    --output_base_dir "${FORMAT_OUTPUT_BASE_DIR}" \
    --bert_model_dir "${BERT_MODEL_DIR}" \
    --ssl_model_dir "${SSL_MODEL_DIR}" \
    --s2g_model_path "${PRETRAINED_S2G_PATH}" \
    --gpu_id "${GPU_ID_TO_USE}" \
    ${USE_HALF_PRECISION_FLAG} \
    --gpt_sovits_version "${GPT_SOVITS_VERSION}"
check_status


# 步骤 3: 模型微调训练 (re_finetune_training.py)
FINETUNE_SCRIPT_PATH="${PROJECT_ROOT_DIR}/re_finetune_training.py"

# 3a: SoVITS 训练
echo -e "\n>>> 阶段 3a: SoVITS模型微调 (re_finetune_training.py sovits) <<<"
SOVITS_TRAIN_ARGS=(
    "${FINETUNE_SCRIPT_PATH}"
    --experiment_name "${EXPERIMENT_NAME}"
    --exp_data_root "${FORMAT_OUTPUT_BASE_DIR}" # 指向格式化数据的父目录
    --gpu_id "${GPU_ID_TO_USE}"
    ${USE_HALF_PRECISION_FLAG}
    --gpt_sovits_version "${GPT_SOVITS_VERSION}"
    sovits # 子命令
    --sovits_batch_size ${SOVITS_BATCH_SIZE}
    --sovits_total_epoch ${SOVITS_EPOCHS}
    --sovits_save_every_epoch ${SOVITS_SAVE_EVERY}
    --sovits_pretrained_s2g "${PRETRAINED_S2G_PATH}"
)
if [ -n "$PRETRAINED_S2D_PATH" ] && { [ "$GPT_SOVITS_VERSION" == "v1" ] || [ "$GPT_SOVITS_VERSION" == "v2" ]; }; then
    SOVITS_TRAIN_ARGS+=(--sovits_pretrained_s2d "${PRETRAINED_S2D_PATH}")
fi
if { [ "$GPT_SOVITS_VERSION" == "v3" ] || [ "$GPT_SOVITS_VERSION" == "v4" ]; }; then
    SOVITS_TRAIN_ARGS+=(--sovits_lora_rank ${SOVITS_LORA_RANK})
    # SOVITS_TRAIN_ARGS+=(--sovits_grad_ckpt) # 如果需要开启梯度检查点
fi
${PYTHON_EXECUTABLE} "${SOVITS_TRAIN_ARGS[@]}"
check_status

# 3b: GPT 训练
echo -e "\n>>> 阶段 3b: GPT模型微调 (re_finetune_training.py gpt) <<<"
${PYTHON_EXECUTABLE} "${FINETUNE_SCRIPT_PATH}" \
    --experiment_name "${EXPERIMENT_NAME}" \
    --exp_data_root "${FORMAT_OUTPUT_BASE_DIR}" \
    --gpu_id "${GPU_ID_TO_USE}" \
    ${USE_HALF_PRECISION_FLAG} \
    --gpt_sovits_version "${GPT_SOVITS_VERSION}" \
    gpt \
    --gpt_batch_size ${GPT_BATCH_SIZE} \
    --gpt_total_epoch ${GPT_EPOCHS} \
    --gpt_save_every_epoch ${GPT_SAVE_EVERY} \
    --gpt_pretrained_s1 "${PRETRAINED_S1_PATH}"\
    --gpt_if_dpo # 如果需要DPO训练
check_status


# 获取训练后的模型路径
TRAINED_GPT_MODEL_PATH="${PROJECT_ROOT_DIR}/GPT_weights_${GPT_SOVITS_VERSION}/${EXPERIMENT_NAME}-e${GPT_EPOCHS}.ckpt" # 确保这个GPT路径也正确

# --- 动态查找 SoVITS 模型路径 ---
SOVITS_MODEL_PATTERN="${PROJECT_ROOT_DIR}/SoVITS_weights_${GPT_SOVITS_VERSION}/${EXPERIMENT_NAME}_e${SOVITS_EPOCHS}_s*_l${SOVITS_LORA_RANK}.pth"

echo "正在尝试查找 SoVITS 模型，匹配模式: ${SOVITS_MODEL_PATTERN}"

# 查找匹配的文件，并选择最新的一个
FOUND_SOVITS_MODEL_PATH=$(ls -t ${SOVITS_MODEL_PATTERN} 2>/dev/null | head -n 1)

# 检查是否找到了文件
if [ -z "${FOUND_SOVITS_MODEL_PATH}" ]; then # 如果变量为空，说明没找到文件
    echo "错误: 找不到匹配模式 '${SOVITS_MODEL_PATTERN}' 的 SoVITS 模型文件。"
    echo "请检查 SoVITS_weights_${GPT_SOVITS_VERSION} 目录以及文件名模式中的固定部分 (${EXPERIMENT_NAME}, e${SOVITS_EPOCHS}, l${SOVITS_LORA_RANK}) 是否正确。"
    exit 1
elif [ $(ls ${SOVITS_MODEL_PATTERN} 2>/dev/null | wc -l) -gt 1 ]; then # 如果找到多个文件
    echo "警告: 找到多个匹配模式 '${SOVITS_MODEL_PATTERN}' 的 SoVITS 模型文件。将使用最新的一个:"
    echo "${FOUND_SOVITS_MODEL_PATH}"
    echo "所有匹配的文件:"
    ls -1 ${SOVITS_MODEL_PATTERN} # 列出所有找到的文件，每行一个
fi

# 将【找到的具体文件路径】赋给 TRAINED_SOVITS_MODEL_PATH
TRAINED_SOVITS_MODEL_PATH="${FOUND_SOVITS_MODEL_PATH}"

echo "训练完成。假设最新的模型权重位于:"
echo "GPT: ${TRAINED_GPT_MODEL_PATH}"
echo "SoVITS (实际找到的): ${TRAINED_SOVITS_MODEL_PATH}" # 这里打印的是一个没有通配符的精确路径
echo "如果上述路径不准确，请修改下面的推理步骤参数。"
sleep 2

# 步骤 4: 推理 (re_inference.py)
INFERENCE_SCRIPT_PATH="${PROJECT_ROOT_DIR}/re_inference.py"

# 确保用于推理的文本文件存在且有内容
if [ ! -f "${TEXT_FOR_INFERENCE_PATH}" ]; then
    echo "错误: 用于推理的文本文件 '${TEXT_FOR_INFERENCE_PATH}' 不存在。"
    echo "请创建此文件并填入要合成的中文文本。"
    exit 1
fi
if [ ! -s "${TEXT_FOR_INFERENCE_PATH}" ]; then
    echo "错误: 用于推理的文本文件 '${TEXT_FOR_INFERENCE_PATH}' 为空。"
    exit 1
fi
# 确保参考音频存在
if [ ! -f "${REF_WAV_FOR_INFERENCE}" ]; then
    echo "错误: 用于推理的参考音频 '${REF_WAV_FOR_INFERENCE}' 不存在。"
    exit 1
fi


echo -e "\n>>> 阶段 4: 推理 (re_inference.py) <<<"
INFERENCE_CMD=(
    ${PYTHON_EXECUTABLE} "${INFERENCE_SCRIPT_PATH}" \
    --gpt_model_path "${TRAINED_GPT_MODEL_PATH}" \
    --sovits_model_path "${TRAINED_SOVITS_MODEL_PATH}" \
    --bert_model_dir "${BERT_MODEL_DIR}" \
    --ssl_model_dir "${SSL_MODEL_DIR}" \
    --ref_wav_path "${REF_WAV_FOR_INFERENCE}" \
    --prompt_text "${PROMPT_TEXT_FOR_INFERENCE}" \
    --text_path "${TEXT_FOR_INFERENCE_PATH}" \
    --output_wav_path "${INFERENCE_OUTPUT_WAV_PATH}" \
    --gpu_id "${GPU_ID_TO_USE}" \
    ${USE_HALF_PRECISION_FLAG} \
    --top_k ${INFER_TOP_K} \
    --temperature ${INFER_TEMP} \
    --speed ${INFER_SPEED} 
    --sample_steps ${INFER_SAMPLE_STEPS}
)
    #--gpt_sovits_version_for_symbols "v2" \
    #  # 如果需要指定CFM步数
    # ^^^ 注意：re_inference.py内部会优先使用从SoVITS模型加载的符号版本
    # 如果SoVITS是v1符号，这里传v2也没用，脚本内部会用v1

# --- 打印将要执行的命令 ---
echo "将执行以下推理命令:"
# 为了可读性和可复制性，将数组元素打印为一行，注意处理空格和引号
# 使用 printf '%q ' "${INFERENCE_CMD[@]}" 可以很好地处理带空格的参数，并进行shell转义
printf "    " # 开头缩进
printf "%q " "${INFERENCE_CMD[@]}"
echo # 换行
echo "你可以复制上面的命令 (除了开头的缩进) 并在终端中单独运行以进行调试。"
echo "------------------------------------------------------------"

# --- 询问用户是否继续执行 ---
read -p "按 Enter键 继续执行脚本中的推理步骤，或按 Ctrl+C 中止并手动调试..."

# --- 执行推理命令 ---
"${INFERENCE_CMD[@]}"
check_status

echo -e "\n============================================================"
echo "GPT-SoVITS 全流程脚本执行完毕!"
echo "最终合成的音频文件位于: ${INFERENCE_OUTPUT_WAV_PATH}"
echo "============================================================"

exit 0
