document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('query-form');
    const questionInput = document.getElementById('question');
    const sqlOutput = document.getElementById('sql-output');
    const statusDiv = document.getElementById('status');

    // 添加键盘事件监听
    questionInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault(); // 阻止默认的换行行为
            form.dispatchEvent(new Event('submit')); // 触发表单提交
        }
        // Shift+Enter 保持换行功能
    });

    // 检查 API 健康状态
    fetch('/health')
        .then(res => res.json())
        .then(data => {
            if (data.model_loaded) {
                statusDiv.textContent = 'API Status: Healthy (Model Loaded)';
                statusDiv.style.color = 'green';
            } else {
                statusDiv.textContent = 'API Status: Model Not Loaded';
                statusDiv.style.color = 'orange';
            }
        })
        .catch(() => {
            statusDiv.textContent = 'API Status: Unreachable';
            statusDiv.style.color = 'red';
        });

    form.addEventListener('submit', function (e) {
        e.preventDefault();
        sqlOutput.textContent = '';
        statusDiv.textContent = 'Generating...';
        statusDiv.style.color = 'blue';
        const question = questionInput.value.trim();
        if (!question) return;
        fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        })
            .then(res => res.json())
            .then(data => {
                if (data.status === 'success') {
                    sqlOutput.textContent = data.sql;
                    statusDiv.textContent = 'Success!';
                    statusDiv.style.color = 'green';
                } else {
                    sqlOutput.textContent = '';
                    statusDiv.textContent = 'Error: ' + (data.message || 'Unknown error');
                    statusDiv.style.color = 'red';
                }
            })
            .catch(err => {
                sqlOutput.textContent = '';
                statusDiv.textContent = 'Request failed.';
                statusDiv.style.color = 'red';
            });
    });
}); 