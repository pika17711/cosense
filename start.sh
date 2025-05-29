#!/bin/bash

# 启动脚本
# 支持启动、停止、重启和状态检查四个子系统

export PYTHONPATH="/src:$PYTHONPATH"

# 配置各模块的Python解释器路径和主程序
COLLABORATION_PYTHON="/home/nvidia/mydisk/miniconda3/envs/rospy/bin/python3"
COLLABORATION_MAIN="src/collaboration/main.py"

DETECTION_PYTHON="/home/nvidia/mydisk/miniconda3/envs/cosense_detection/bin/python3"
DETECTION_MAIN="src/detection/main.py"

PERCEPTION_PYTHON="/home/nvidia/mydisk/miniconda3/envs/rospy/bin/python3"
PERCEPTION_MAIN="src/perception/main.py"

PRESENTATION_PYTHON="/path/to/presentation/python"
PRESENTATION_MAIN="src/presentation/main.py"

# 配置日志文件路径
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# 配置PID文件路径
PID_DIR="pids"
mkdir -p "$PID_DIR"

COLLABORATION_PID="$PID_DIR/collaboration.pid"
DETECTION_PID="$PID_DIR/detection.pid"
PERCEPTION_PID="$PID_DIR/perception.pid"
PRESENTATION_PID="$PID_DIR/presentation.pid"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# 检查命令行参数
if [ $# -lt 1 ]; then
    echo "用法: $0 [start|stop|restart|status] [module_name]"
    echo "模块名可选值: collaboration, detection, perception, presentation, all"
    exit 1
fi

command="$1"
module="$2"

# 检查模块名是否有效
if [ "$module" != "collaboration" ] && [ "$module" != "detection" ] && [ "$module" != "perception" ] && [ "$module" != "presentation" ] && [ "$module" != "all" ]; then
    echo -e "${RED}错误: 无效的模块名 '$module'${NC}"
    exit 1
fi

# 启动单个服务
start_service() {
    local service_name="$1"
    local python_path="$2"
    local main_file="$3"
    local pid_file="$4"
    local log_file="$LOG_DIR/${service_name}.log"
    local run_foreground="$5"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null; then
            echo -e "${YELLOW}$service_name 已经在运行中，PID: $pid${NC}"
            return 1
        else
            echo -e "${YELLOW}发现过时的PID文件，正在清理...${NC}"
            rm -f "$pid_file"
        fi
    fi
    
    echo -e "${GREEN}正在启动 $service_name...${NC}"
    
    if [ "$run_foreground" = "true" ]; then
        # 前台运行（用于collaboration模块）
        "$python_path" "$main_file" &
        echo $! > "$pid_file"
        
        echo -e "${GREEN}$service_name 已在前台启动，PID: $pid${NC}"
        echo -e "${YELLOW}提示: 使用 Ctrl+C 停止服务，或在另一个终端使用 '$0 stop $service_name'${NC}"
        
        # 等待服务退出并清理PID文件
        wait $pid
        rm -f "$pid_file"
        echo -e "${GREEN}$service_name 已停止${NC}"
    else
        # 后台运行（用于其他模块）
        nohup "$python_path" "$main_file" > "$log_file" 2>&1 &
        echo $! > "$pid_file"
        
        # 验证服务是否成功启动
        sleep 1
        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file")
            if ps -p "$pid" > /dev/null; then
                echo -e "${GREEN}$service_name 已在后台启动，PID: $pid${NC}"
                echo -e "${GREEN}日志文件: $log_file${NC}"
                return 0
            else
                echo -e "${RED}$service_name 启动失败${NC}"
                rm -f "$pid_file"
                return 1
            fi
        else
            echo -e "${RED}$service_name 启动失败，未生成PID文件${NC}"
            return 1
        fi
    fi
}

# 停止单个服务
stop_service() {
    local service_name="$1"
    local pid_file="$2"
    
    if [ ! -f "$pid_file" ]; then
        echo -e "${YELLOW}$service_name 未运行${NC}"
        return 1
    fi
    
    local pid=$(cat "$pid_file")
    if ! ps -p "$pid" > /dev/null; then
        echo -e "${YELLOW}$service_name 未运行（过时的PID文件）${NC}"
        rm -f "$pid_file"
        return 1
    fi
    
    echo -e "${GREEN}正在停止 $service_name (PID: $pid)...${NC}"
    kill "$pid"
    
    # 等待服务停止
    local timeout=10
    while ps -p "$pid" > /dev/null && [ "$timeout" -gt 0 ]; do
        sleep 1
        timeout=$((timeout - 1))
    done
    
    if ps -p "$pid" > /dev/null; then
        echo -e "${RED}无法停止 $service_name，正在强制终止...${NC}"
        kill -9 "$pid"
        sleep 1
    fi
    
    if [ ! -f "$pid_file" ]; then
        echo -e "${YELLOW}PID文件 $pid_file 不存在${NC}"
    else
        rm -f "$pid_file"
    fi
    
    echo -e "${GREEN}$service_name 已停止${NC}"
    return 0
}

# 检查单个服务状态
check_status() {
    local service_name="$1"
    local pid_file="$2"
    
    if [ ! -f "$pid_file" ]; then
        echo -e "${RED}$service_name 未运行${NC}"
        return 1
    fi
    
    local pid=$(cat "$pid_file")
    if ps -p "$pid" > /dev/null; then
        echo -e "${GREEN}$service_name 正在运行中，PID: $pid${NC}"
        return 0
    else
        echo -e "${RED}$service_name 未运行（过时的PID文件）${NC}"
        rm -f "$pid_file"
        return 1
    fi
}

# 根据命令和模块执行相应操作
case "$command" in
    start)
        if [ "$module" = "all" ]; then
            echo -e "${GREEN}正在启动所有服务...${NC}"
            
            # 先启动后台服务
            start_service "perception" "$PERCEPTION_PYTHON" "$PERCEPTION_MAIN" "$PERCEPTION_PID" "false"
            start_service "detection" "$DETECTION_PYTHON" "$DETECTION_MAIN" "$DETECTION_PID" "false"
            # start_service "presentation" "$PRESENTATION_PYTHON" "$PRESENTATION_MAIN" "$PRESENTATION_PID" "false"
            
            # 最后启动前台服务
            echo -e "${GREEN}现在启动 collaboration 模块（前台运行）...${NC}"
            start_service "collaboration" "$COLLABORATION_PYTHON" "$COLLABORATION_MAIN" "$COLLABORATION_PID" "true"
        else
            case "$module" in
                collaboration)
                    start_service "collaboration" "$COLLABORATION_PYTHON" "$COLLABORATION_MAIN" "$COLLABORATION_PID" "true"
                    ;;
                detection|perception|presentation)
                    start_service "$module" "$(eval echo \$${module^^}_PYTHON)" "$(eval echo \$${module^^}_MAIN)" "$(eval echo \$${module^^}_PID)" "false"
                    ;;
            esac
        fi
        ;;
    
    stop)
        if [ "$module" = "all" ]; then
            echo -e "${GREEN}正在停止所有服务...${NC}"
            # 按相反顺序停止服务
            stop_service "presentation" "$PRESENTATION_PID"
            stop_service "perception" "$PERCEPTION_PID"
            stop_service "detection" "$DETECTION_PID"
            stop_service "collaboration" "$COLLABORATION_PID"
        else
            case "$module" in
                collaboration|detection|perception|presentation)
                    stop_service "$module" "$(eval echo \$${module^^}_PID)"
                    ;;
            esac
        fi
        ;;
    
    restart)
        if [ "$module" = "all" ]; then
            echo -e "${GREEN}正在重启所有服务...${NC}"
            $0 stop all
            sleep 1
            $0 start all
        else
            echo -e "${GREEN}正在重启 $module...${NC}"
            $0 stop "$module"
            sleep 1
            $0 start "$module"
        fi
        ;;
    
    status)
        if [ "$module" = "all" ]; then
            echo -e "${GREEN}检查所有服务状态...${NC}"
            check_status "collaboration" "$COLLABORATION_PID"
            check_status "detection" "$DETECTION_PID"
            check_status "perception" "$PERCEPTION_PID"
            check_status "presentation" "$PRESENTATION_PID"
        else
            check_status "$module" "$(eval echo \$${module^^}_PID)"
        fi
        ;;
    
    *)
        echo -e "${RED}错误: 未知命令 '$command'${NC}"
        exit 1
        ;;
esac

exit 0    
