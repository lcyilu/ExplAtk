#!/bin/bash
set -euo pipefail  # 严格错误处理模式[9,11](@ref)

# ===== 配置参数 =====
SOURCE_DIR="$1"          # 源文件目录（脚本第一个参数）
TXL_PROGRAM="$2"         # TXL程序路径（脚本第二个参数）
FAILED_LIST="failed_files_devign_novul.txt"  # 失败文件列表输出路径 ##需标记数据集名称
LOG_FILE="txl_batch.log"         # 综合日志文件
MAX_CONCURRENCY=8        # 最大并发数（根据CPU核心数调整）[4](@ref)
RETRY_COUNT=2            # 失败重试次数[4,11](@ref)

# ===== 输入验证 =====
if [[ $# -ne 2 ]]; then
  echo "用法: $0 <源文件目录> <TXL程序路径>"
  exit 1
fi

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "错误: 源文件目录 $SOURCE_DIR 不存在!" >&2
  exit 2
fi

if [[ ! -f "$TXL_PROGRAM" ]]; then
  echo "错误: TXL程序 $TXL_PROGRAM 不存在!" >&2
  exit 3
fi

# ===== 初始化环境 =====
echo "===== TXL批量处理开始 [$(date)] ====" | tee "$LOG_FILE"
rm -f "$FAILED_LIST"  # 清理旧记录
touch "$FAILED_LIST"
export FAILED_LIST SOURCE_DIR TXL_PROGRAM  # 导出为环境变量

# ===== 核心处理函数 =====
process_file() {
  local file="$1"
  local retry=0
  local output status
  
  while (( retry <= RETRY_COUNT )); do
    # 执行TXL转换并捕获输出[1](@ref)
    output=$(txl "$file" "$TXL_PROGRAM" 2>&1)
    status=$?
    
    # 检查语法错误[9,10](@ref)
    if [[ "$output" == *"Syntax error at or near"* ]]; then
      if (( retry == RETRY_COUNT )); then
        echo "[错误] 解析失败: $file" | tee -a "$LOG_FILE"
        echo "$file" >> "$FAILED_LIST"
        return 1
      else
        ((retry++))
        sleep 0.5
      fi
    else
      echo "[成功] 处理完成: $file" | tee -a "$LOG_FILE"
      return 0
    fi
  done
}

# ===== 主循环 =====
echo "扫描源文件目录: $SOURCE_DIR"
echo "使用TXL程序: $TXL_PROGRAM"
echo "并发数: $MAX_CONCURRENCY | 重试次数: $RETRY_COUNT"

# 使用find+xargs实现并发处理[4,7](@ref)
# export -f process_file  # 导出函数供子shell使用
# find "$SOURCE_DIR" -type f -print0 | \
  # xargs -0 -P "$MAX_CONCURRENCY" -I {} bash -c 'process_file "$@"' _ {}

export -f process_file  # 导出函数供子shell使用
cat {HOME_PATH}/VulDS/utils/CLONEGEN/CodeTransformationTest/RM/failed_files_last.txt | xargs -d '\n' -P "$MAX_CONCURRENCY" -I {} bash -c 'process_file "$@"' _ {}  

# ===== 结果统计 =====
total_files=$(find "$SOURCE_DIR" -type f | wc -l)
failed_count=$(wc -l < "$FAILED_LIST")

echo "===== 处理完成 [$(date)] ====" | tee -a "$LOG_FILE"
echo "总计文件: $total_files | 成功: $((total_files - failed_count)) | 失败: $failed_count"
echo "失败文件列表保存至: $FAILED_LIST"