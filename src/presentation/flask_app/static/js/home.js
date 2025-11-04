const longitudeDisplay = document.getElementById('longitude');
const latitudeDisplay = document.getElementById('latitude');
const headingDisplay = document.getElementById('heading');
const speedDisplay = document.getElementById('speed');

const carIdDisplay = document.getElementById('car-id');
const communicationRateDisplay = document.getElementById('communication-rate');

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

// 每秒更新一次
setInterval(updateCarState, 1000); // 1000 毫秒 = 1 秒
setInterval(updateCarId, 10 * 1000); // 10000 毫秒 = 10 秒
//setInterval(updateCommunicationRate, 1000); // 1000 毫秒 = 1 秒

// 第一次加载时，希望立即更新（覆盖初始值），调用一次
updateCarState();
updateCarId();
//updateCommunicationRate();