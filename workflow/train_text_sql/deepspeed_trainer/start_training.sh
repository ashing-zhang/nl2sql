#!/bin/bash

# DeepSpeed QLoRA Training启动脚本
set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# 默认配置
DEFAULT_CONFIG="configs/train_config.yaml"
DEFAULT_DEEPSPEED_CONFIG="configs/deepspeed_config.json"

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# 检查依赖
check_dependencies() {
    log_step "检查依赖..."
    
    if ! command -v python &> /dev/null; then
        log_error "Python未安装"
        exit 1
    fi
    
    python -c "import torch, transformers, datasets, peft" || {
        log_error "缺少必要的Python包，请运行: pip install -r requirements.txt"
        exit 1
    }
    
    log_info "依赖检查完成"
}

# 检查GPU
check_gpu() {
    log_step "检查GPU环境..."
    
    if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
        log_info "检测到 $GPU_COUNT 个GPU"
    else
        log_warn "未检测到GPU，将使用CPU训练"
    fi
}

# 显示帮助信息
show_help() {
    echo "DeepSpeed QLoRA Training启动脚本"
    echo ""
    echo "使用方法:"
    echo "  $0 [模式] [参数]"
    echo ""
    echo "模式:"
    echo "  single     单GPU训练"
    echo "  multi      多GPU训练"
    echo "  custom     自定义参数训练"
    echo "  quick      快速测试"
    echo "  help       显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 single"
    echo "  $0 multi --num_gpus 2"
    echo "  $0 custom --model_name workflow/models/Tongyi-Finance-14B-Chat"
    echo "  $0 quick"
}

# 单GPU训练
single_gpu_training() {
    log_step "启动单GPU训练..."
    python scripts/train.py "$@"
}

# 多GPU训练
multi_gpu_training() {
    log_step "启动多GPU训练..."
    local num_gpus=${1:-$(python -c "import torch; print(torch.cuda.device_count())")}
    log_info "使用 $num_gpus 个GPU进行训练"
    deepspeed --num_gpus=$num_gpus scripts/train.py --deepspeed $DEFAULT_DEEPSPEED_CONFIG "${@:2}"
}

# 快速测试
quick_test() {
    log_step "启动快速测试..."
    python quick_start.py
}

# 主函数
main() {
    log_info "DeepSpeed QLoRA Training启动脚本"
    
    # 检查依赖
    check_dependencies
    
    # 检查GPU
    check_gpu
    
    # 解析模式
    MODE=${1:-single}
    
    case $MODE in
        single)
            single_gpu_training "${@:2}"
            ;;
        multi)
            multi_gpu_training "${@:2}"
            ;;
        custom)
            if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
                local gpu_count=$(python -c "import torch; print(torch.cuda.device_count())")
                if [[ $gpu_count -gt 1 ]]; then
                    multi_gpu_training "${@:2}"
                else
                    single_gpu_training "${@:2}"
                fi
            else
                single_gpu_training "${@:2}"
            fi
            ;;
        quick)
            quick_test
            ;;
        help)
            show_help
            ;;
        *)
            log_error "未知模式: $MODE"
            show_help
            exit 1
            ;;
    esac
    
    log_info "训练完成！"
}

# 执行主函数
main "$@" 