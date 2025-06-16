import os
import sys
import shutil
import re
import datetime

# --- 配置 ---
# 项目根目录 (包含 GPT_SoVITS 文件夹的目录)
PROJECT_ROOT_PARENT = "/root/autodl-tmp/GPT-SoVITS" 
# 要处理的顶层包的名称 (即我们希望作为导入前缀的包名)
TOP_LEVEL_PACKAGE_NAME = "GPT_SoVITS"
# 顶层包的实际路径
PACKAGE_ROOT_TO_SCAN = os.path.join(PROJECT_ROOT_PARENT, TOP_LEVEL_PACKAGE_NAME)


def backup_project(source_dir, backup_dir_base="project_backups"):
    """创建项目的备份。"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{os.path.basename(source_dir)}_backup_{timestamp}"
    backup_path = os.path.join(backup_dir_base, backup_name)

    if not os.path.exists(source_dir):
        print(f"错误: 源目录 '{source_dir}' 不存在，无法备份。")
        return None

    try:
        os.makedirs(backup_dir_base, exist_ok=True)
        shutil.copytree(source_dir, backup_path)
        print(f"项目已成功备份到: {backup_path}")
        return backup_path
    except Exception as e:
        print(f"错误: 备份项目失败: {e}")
        return None

def get_internal_modules(package_scan_path):
    """
    扫描指定包路径下的第一级子目录，将其视为内部模块。
    同时也会将包内直接的 .py 文件（去除.py后缀）视为模块。
    """
    internal_modules = set()
    if not os.path.isdir(package_scan_path):
        print(f"警告: 包扫描路径 '{package_scan_path}' 不是一个有效目录。")
        return list(internal_modules)

    for item in os.listdir(package_scan_path):
        item_path = os.path.join(package_scan_path, item)
        if os.path.isdir(item_path):
            # 检查是否是合法的包 (包含__init__.py) 或只是一个模块目录
            # 对于简单的子目录名作为模块名，我们直接添加
            if not item.startswith('.') and not item.startswith('_'): # 排除隐藏目录和私有目录
                 internal_modules.add(item)
        elif item.endswith(".py") and item != "__init__.py":
            if not item.startswith('.') and not item.startswith('_'):
                module_name = item[:-3] # 去除 .py 后缀
                internal_modules.add(module_name)
                
    print(f"检测到的内部顶层模块/包名: {sorted(list(internal_modules))}")
    return sorted(list(internal_modules))


def process_python_file(filepath, internal_modules_list, package_prefix_to_add):
    """处理单个Python文件，修正导入语句。"""
    print(f"正在处理文件: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"  错误: 读取文件 {filepath} 失败: {e}")
        return False

    new_lines = []
    file_modified_flag = False
    modifications_log = [] # 记录本文件的修改

    for line_num, line in enumerate(lines):
        original_line = line
        processed_line = line # 默认为原始行

        # 匹配: from module import ... 或 from module.submodule import ...
        # 排除: from .module import ..., from ..module import ..., from package_prefix.module import ...
        from_match = re.match(r"^\s*from\s+([a-zA-Z0-9_.]+)\s+import\s+(.+)", line)
        if from_match:
            module_path = from_match.group(1)
            import_targets = from_match.group(2)
            
            if not module_path.startswith(".") and not module_path.startswith(package_prefix_to_add):
                first_module_part = module_path.split('.')[0]
                if first_module_part in internal_modules_list:
                    new_module_path = package_prefix_to_add + module_path
                    processed_line = f"from {new_module_path} import {import_targets}\n"
                    # 确保实际发生了改变 (处理末尾换行符和空格可能导致的误判)
                    if original_line.strip() != processed_line.strip():
                        modifications_log.append(f"  行 {line_num+1}: '{original_line.strip()}' -> '{processed_line.strip()}' (from import)")
                        file_modified_flag = True
        
        # 匹配: import module 或 import module.submodule 或 import module as alias
        # 排除: import package_prefix.module ...
        # 注意: 相对导入 `import .module` 不存在
        import_match = re.match(r"^\s*import\s+([a-zA-Z0-9_.]+)(\s+as\s+[a-zA-Z0-9_]+)?(.*)", line)
        if import_match and not from_match: # 确保不是from import语句的一部分
            module_path_full = import_match.group(1) # 可能是 module 或 module.submodule
            alias_part = import_match.group(2) if import_match.group(2) else ""
            trailing_comment = import_match.group(3) if import_match.group(3) else "" # 保留行尾注释

            if not module_path_full.startswith(package_prefix_to_add):
                first_module_part = module_path_full.split('.')[0]
                if first_module_part in internal_modules_list:
                    new_module_path = package_prefix_to_add + module_path_full
                    processed_line = f"import {new_module_path}{alias_part}{trailing_comment}\n"
                    if original_line.strip() != processed_line.strip():
                        modifications_log.append(f"  行 {line_num+1}: '{original_line.strip()}' -> '{processed_line.strip()}' (import)")
                        file_modified_flag = True
        
        new_lines.append(processed_line)

    if file_modified_flag:
        print(f"  文件 '{os.path.basename(filepath)}' 中检测到修改:")
        for log_entry in modifications_log:
            print(log_entry)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            print(f"  文件 {filepath} 已被修改并保存。")
        except Exception as e:
            print(f"  错误: 写入文件 {filepath} 失败: {e}")
            return False
    else:
        print(f"  文件 '{os.path.basename(filepath)}' 无需修改。")
        
    return file_modified_flag


def main():
    print("--- 开始批量修复Python导入语句 ---")

    # 1. 备份项目 (强烈建议！)
    # backup_path = backup_project(PACKAGE_ROOT_TO_SCAN) # 备份的是顶层包目录
    backup_path = backup_project(PROJECT_ROOT_PARENT) # 备份整个项目根目录更安全
    if not backup_path:
        confirm_backup = input("警告: 自动备份失败或跳过。是否确定在没有备份的情况下继续? (yes/no): ")
        if confirm_backup.lower() != 'yes':
            print("操作已中止。")
            return
    else:
        print(f"项目已备份到 '{backup_path}'。如果出现问题，请从此路径恢复。")


    # 2. 获取内部模块列表
    # 我们要扫描的是 PACKAGE_ROOT_TO_SCAN (例如 /root/autodl-tmp/GPT-SoVITS/GPT_SoVITS/)
    # 因为导入时是 from module import ..., module 应该是这个目录下的子文件夹或.py文件
    internal_modules = get_internal_modules(PACKAGE_ROOT_TO_SCAN)
    if not internal_modules:
        print("错误: 未能检测到任何内部模块。请检查 PACKAGE_ROOT_TO_SCAN 配置是否正确。")
        return
        
    package_prefix = TOP_LEVEL_PACKAGE_NAME + "."

    # 3. 遍历并处理所有Python文件
    # 遍历的应该是包含这些内部模块的顶层包目录
    # 例如 /root/autodl-tmp/GPT-SoVITS/GPT_SoVITS/
    # 因为导入是在这些文件内部发生的
    total_files_processed = 0
    total_files_modified = 0

    for root, dirs, files in os.walk(PACKAGE_ROOT_TO_SCAN):
        # 排除常见的非代码目录
        dirs[:] = [d for d in dirs if d not in ['.git', 'venv', '__pycache__', '.ipynb_checkpoints', 'pretrained_models', 'TEMP', 'logs', 'output_preprocess_cli', 'SoVITS_weights', 'GPT_weights', 'output_audio', 'project_backups']]
        
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                if process_python_file(file_path, internal_modules, package_prefix):
                    total_files_modified += 1
                total_files_processed += 1
    
    print("\n--- 批量修复完成 ---")
    print(f"总共处理了 {total_files_processed} 个Python文件。")
    print(f"总共修改了 {total_files_modified} 个Python文件。")
    if total_files_modified > 0:
        print("请务必使用版本控制工具 (如 git diff) 仔细检查所有更改，")
        print("并进行充分测试，确保没有引入新的错误！")

if __name__ == "__main__":
    main()