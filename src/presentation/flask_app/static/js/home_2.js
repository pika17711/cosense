const longitudeDisplay = document.getElementById('longitude');
const latitudeDisplay = document.getElementById('latitude');
const headingDisplay = document.getElementById('heading');
const speedDisplay = document.getElementById('speed');

const carIdDisplay = document.getElementById('car-id');
const communicationRateDisplay = document.getElementById('communication-rate');
const logContainer = document.getElementById('logContainer')

const commandInput = document.getElementById('commandInput');


commandInput.addEventListener('keypress', function(event) {
    // 检查按下的键是否是回车键 (keyCode 13 或 key 'Enter')
    if (event.key === 'Enter') {
        // 阻止回车键的默认行为（例如提交表单，如果输入框在一个form中）
        event.preventDefault();

        // 调用要触发的函数
        sendCommand();
    }
});

// async function sendCommand(){
//     const inputValue = commandInput.value.trim();
//     if (inputValue === "") {
//         return;
//     }

//     try {
//         // 使用 fetch API 发送 POST 请求
//         const response = await fetch('/send_command', {
//             method: 'POST',
//             headers: {
//                 'Content-Type': 'application/json'
//             },
//             // 将数据转换为 JSON 字符串
//             body: JSON.stringify({ text_content: inputValue })
//         });

//         // 检查响应状态
//         if (!response.ok) {
//             // 如果响应状态码不是 2xx，抛出错误
//             const errorData = await response.json();
//             throw new Error(errorData.message || '网络响应不正确');
//         }

//         const data = await response.json(); // 解析 JSON 响应

//         if (data.status === 'success') {
//             showMessage(data.message, "success");
//             commandInput.value = ''; // 清空输入框
//         } else {
//             showMessage(data.message || '后端处理失败', "error");
//         }

//     } catch (error) {
//         console.error('发送文本到后端时出错:', error);
//         showMessage(`请求失败: ${error.message}`, "error");
//     }
// }

async function sendCommand(){
    const inputValue = commandInput.value.trim();
    if (inputValue === "") {
        return;
    }

//    console.info(inputValue)

    // 使用 fetch API 发送 POST 请求
    fetch('/send_command', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json', // 告诉服务器我们发送的是 JSON
        },
        body: JSON.stringify({ command: inputValue }), // 将数据转换为 JSON 字符串
    })
    .then(response => response.json()) // 解析 JSON 响应
    .then(data => {
        console.log('后端响应:', data);
        if (data.status === 'success') {
            console.error(`后端成功处理: ${data.message}`);
        } else {
            console.error(`后端错误: ${data.message}`);
        }
    })
    .catch(error => {
        console.error('发送请求失败:', error);
    });

    commandInput.value = ''; // 清空输入框
}


// 定义一个函数来获取并更新车辆状态
function updateCarState() {
    fetch('/get_car_state') // 向后端API发起请求
        .then(response => response.json()) // 将响应解析为JSON
        .then(data => {
            longitudeDisplay.textContent = data.longitude;
            latitudeDisplay.textContent = data.latitude;
            headingDisplay.textContent = data.heading;
            speedDisplay.textContent = data.speed;
            communicationRateDisplay.textContent = data.communication_rate
        })
        .catch(error => {
            console.error('获取车辆状态失败:', error);
            longitudeDisplay.textContent = 'N/A';
            latitudeDisplay.textContent = 'N/A';
            headingDisplay.textContent = 'N/A';
            speedDisplay.textContent = 'N/A';
            communicationRateDisplay.textContent = 'N/A'
        });
}

function updateCarId() {
    fetch('/get_car_id') // 向后端API发起请求
        .then(response => response.json()) // 将响应解析为JSON
        .then(data => {
            carIdDisplay.textContent = data.car_id;
        })
        .catch(error => {
            console.error('获取车辆ID失败:', error);
            carIdDisplay.textContent = 'N/A';
        });
}

//function updateCommunicationRate() {
//    fetch('/get_communication_rate') // 向后端API发起请求
//        .then(response => response.json()) // 将响应解析为JSON
//        .then(data => {
//            communicationRateDisplay.textContent = data.communication_rate;
//        })
//        .catch(error => {
//            console.error('获取通信码率失败:', error);
//            communicationRateDisplay.textContent = 'N/A';
//        });
//}

async function fetchNewLog() {
    fetch('/get_log') // 向后端API发起请求
        .then(response => response.json()) // 将响应解析为JSON
        .then(data => {
            if (data.log){
                console.log('Received log:', data.log);
                var logEntry = document.createElement('div');
                logEntry.className = 'log-entry';

                // 尝试根据日志级别添加样式
                if (data.log.includes(" - INFO - ")) {
                    logEntry.classList.add('INFO');
                } else if (data.log.includes(" - WARNING - ")) {
                    logEntry.classList.add('WARNING');
                } else if (data.log.includes(" - ERROR - ")) {
                    logEntry.classList.add('ERROR');
                } else if (data.log.includes(" - CRITICAL - ")) {
                    logEntry.classList.add('CRITICAL');
                } else if (data.log.includes(" - DEBUG - ")) {
                    logEntry.classList.add('DEBUG');
                }

                logEntry.textContent = data.log;
                logContainer.appendChild(logEntry);

                // 自动滚动到底部
                logContainer.scrollTop = logContainer.scrollHeight;
            }
            fetchNewLog()
        })
        .catch(error => {
            console.error('获取日志失败:', error);
            fetchNewLog()
        });
}

// 每秒更新一次
setInterval(updateCarState, 1000); // 1000 毫秒 = 1 秒
setInterval(updateCarId, 10 * 1000); // 10000 毫秒 = 10 秒
//setInterval(updateCommunicationRate, 1000); // 1000 毫秒 = 1 秒

// 第一次加载时，希望立即更新（覆盖初始值），调用一次
updateCarState();
updateCarId();
fetchNewLog();
//updateCommunicationRate();