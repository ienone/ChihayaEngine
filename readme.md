# 🎤 千造AI音 - 少样本歌声转换克隆系统

[![AGPL-3.0](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0) ![Python](https://img.shields.io/badge/Python-3.8%2B-green)

## ✨ 核心功能

- **少样本训练**：突破传统限制，仅需10秒人声音频即可构建可用模型
- **端到端流程**：集成GPT-SoVITS声音克隆 + so-vits-svc歌声转换完整流程
- **可视化界面**：提供友好的Web操作界面，降低技术门槛
- **高效干声生成**：快速生成高质量训练样本

> **创新点**：解决了so-vits-svc传统训练需要数小时纯净人声样本的痛点，大幅降低翻唱创作门槛。

## 🚀 快速开始

### 环境准备

1. 克隆依赖项目：
   ```bash
   git clone https://github.com/facebookresearch/demucs demucs/
   git clone https://github.com/svc-develop-team/so-vits-svc sovits_svc/
   git clone https://github.com/RVC-Boss/GPT-SoVITS GPT_SoVITS/
   ```

2. 创建虚拟环境：
   ```bash
   conda create -n demucs
   conda create -n GPTSoVits 
   conda create -n so-vits-svc 
   conda create -n flask_app 
   ```

3. 下载预训练模型与解码器等（请参考各子项目文档）

### 启动系统

```bash
python backend.py
```
然后访问 `code.html` 开始使用

## ⚠️ 注意事项

### 常见问题解决

**模块导入错误**：

未定义无法导入 或 ModuleNotFoundError，尤其与 GPT_SoVITS 相关，运行修复脚本：

```bash
python rename.py
```
> 该脚本会：
> 1. 创建项目备份
> 2. 扫描所有Python文件
> 3. 修正模块导入路径（如将 `from module` → `from GPT_SoVITS.module`）

## 🎧 效果展示
<video src="https://github.com/ienone/ChihayaEngine/blob/main/files/demo.mp4" controls width="800"></video>

<audio src="https://github.com/ienone/ChihayaEngine/blob/main/files/%E7%A4%BA%E4%BE%8B-%E6%96%AD%E6%A1%A5%E6%AE%8B%E9%9B%AA.flac" controls></audio>


---

## 🖥️ 硬件配置与性能参考

### 测试平台
- **显卡型号**: NVIDIA RTX 3090 (24GB VRAM)
- **显存占用**:控制batch size最低 <8GB
- **运行状态**: 最好情况下GPU利用率70%左右波动

### 训练效率基准
| 模块               | 耗时范围      | 备注                                                         |
| ------------------ | ------------- | ------------------------------------------------------------ |
| **GPT-SoVITS微调** | 3-5分钟       |                                                              |
| **音频合成**       | 1小时+/2h音频 | 可通过批处理优化提升效率                                     |
| **so-vits-svc**    |               |                                                              |
| - 基模型微调       | ~30分钟       | (3000步),建议开启`all_in_mem`：加载所有数据集到内存中，显著提速 |
| - 扩散模型微调     | ~1小时        |                                                              |

💡 **优化方向**：GPT-SoVITS启用批处理逻辑或许可显著提升音频合成效率

---
