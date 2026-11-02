document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    const reportContainer = document.getElementById('report-container');
    const reportContent = document.getElementById('report-content');

    // Function to add a message to the chat box
    function addMessage(message, sender) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', `${sender}-message`);
        messageElement.innerText = message;
        chatBox.appendChild(messageElement);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    // Handle sending a chat message
    async function sendMessage() {
        const message = userInput.value.trim();
        if (message === '') return;

        addMessage(message, 'user');
        userInput.value = '';

        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message })
        });

        const data = await response.json();
        addMessage(data.response, 'bot');
    }

    // Handle analyzing the conversation
    async function analyzeConversation() {
        reportContainer.style.display = 'block';
        reportContent.innerHTML = '<p>正在分析您的性格，请稍候...</p>';

        const response = await fetch('/analyze', {
            method: 'POST'
        });

        const report = await response.json();

        reportContent.innerHTML = `
            <h3>大五人格模型 (Big Five)</h3>
            <ul>
                <li><strong>开放性 (Openness):</strong> ${report.openness}</li>
                <li><strong>尽责性 (Conscientiousness):</strong> ${report.conscientiousness}</li>
                <li><strong>外向性 (Extraversion):</strong> ${report.extraversion}</li>
                <li><strong>宜人性 (Agreeableness):</strong> ${report.agreeableness}</li>
                <li><strong>神经质 (Neuroticism):</strong> ${report.neuroticism}</li>
            </ul>
            <h3>总结</h3>
            <p>${report.summary}</p>
        `;
    }

    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    analyzeBtn.addEventListener('click', analyzeConversation);
});