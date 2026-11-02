import gradio as gr
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import requests
import json
from collections import Counter
import os

print(f"正在使用的 Gradio 版本: {gr.__version__}")

# --- 1. 配置 ---
API_KEY = os.environ.get("BAIDU_API_KEY", "")
BERT_MODEL_NAME = 'bert-base-chinese'
MAX_LEN = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tasks_to_analyze = {
    'emotion': {'labels': ['positive', 'negative'], 'model_path': 'bert_emotion.pt'},
    'Openness': {'labels': ['high', 'low'], 'model_path': 'bert_Openness.pt'},
    'Conscientiousness': {'labels': ['high', 'low'], 'model_path': 'bert_Conscientiousness.pt'},
    'Extraversion': {'labels': ['high', 'low'], 'model_path': 'bert_Extraversion.pt'},
    'Agreeableness': {'labels': ['high', 'low'], 'model_path': 'bert_Agreeableness.pt'},
    'Neuroticism': {'labels': ['high', 'low'], 'model_path': 'bert_Neuroticism.pt'}
}

BERT_TOKENIZER = BertTokenizer.from_pretrained(BERT_MODEL_NAME, local_files_only=True)
LOADED_MODELS = {}

def get_ernie_response(messages):
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro"
    payload = json.dumps({"messages": messages})
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }
    response = requests.post(url, headers=headers, data=payload)
    if response.status_code == 200:
        return response.json().get("result")
    else:
        return f"AI连接失败，状态码: {response.status_code}，内容: {response.text}"

def predict_with_bert(texts, task_name):
    task_info = tasks_to_analyze[task_name]
    model_path = task_info['model_path']
    labels = task_info['labels']
    if not os.path.exists(model_path):
        return [f"错误: 模型文件 {model_path} 未找到" for _ in texts]
    if model_path not in LOADED_MODELS:
        model = BertForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=len(labels), local_files_only=True)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        LOADED_MODELS[model_path] = model
    model = LOADED_MODELS[model_path]
    results = []
    for text in texts:
        encoding = BERT_TOKENIZER(text, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt')
        input_ids = encoding['input_ids'].to(DEVICE)
        attention_mask = encoding['attention_mask'].to(DEVICE)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = torch.argmax(outputs.logits, dim=1).item()
        results.append(labels[pred])
    return results

def respond(message, chat_history, user_inputs_state):
    user_message_formatted = {"role": "user", "content": message}
    messages_for_api = []
    for user_msg, ai_msg in chat_history:
        messages_for_api.append({"role": "user", "content": user_msg})
        messages_for_api.append({"role": "assistant", "content": ai_msg})
    messages_for_api.append(user_message_formatted)
    bot_message = get_ernie_response(messages_for_api)
    chat_history.append((message, bot_message))
    user_inputs_state.append(message)
    return "", chat_history, user_inputs_state

# 新增：定义每个特质的详细信息，用于生成更丰富的报告
trait_map = {
    'emotion':           {'name': '情绪状态', 'emoji': '😊', 'positive': '积极乐观', 'negative': '偶有消沉'},
    'Openness':          {'name': '开放性 (想象力)', 'emoji': '🎨', 'high': '高 - 富有创造力，好奇心强', 'low': '低 - 注重实际，偏爱传统'},
    'Conscientiousness': {'name': '尽责性 (条理性)', 'emoji': '✅', 'high': '高 - 做事有条理，认真负责', 'low': '低 - 灵活随性，不拘小节'},
    'Extraversion':      {'name': '外向性 (社交性)', 'emoji': '🗣️', 'high': '高 - 热情开朗，喜欢社交', 'low': '低 - 安静内向，享受独处'},
    'Agreeableness':     {'name': '宜人性 (合作性)', 'emoji': '🤝', 'high': '高 - 乐于助人，有同情心', 'low': '低 - 观点独立，有批判精神'},
    'Neuroticism':       {'name': '情绪稳定性', 'emoji': '⚖️', 'high': '高 - 容易紧张，情绪波动较大', 'low': '低 - 情绪稳定，沉着冷静'}
}

def analyze_history(user_inputs_state):
    """
    分析所有用户输入并生成美化版的报告
    """
    if not user_inputs_state:
        return "#### 报告生成区\n\n还没有对话历史，快来和我聊聊吧！"

    report = f"## 📝 您的对话分析报告\n\n基于 **{len(user_inputs_state)}** 句对话的初步分析：\n\n---\n"

    for task_name, info in tasks_to_analyze.items():
        predictions = predict_with_bert(user_inputs_state, task_name)
        # 如果没有有效的预测（比如模型文件丢失），则跳过
        if not predictions or any("错误:" in p for p in predictions):
            continue
            
        counts = Counter(predictions)
        final_label = counts.most_common(1)[0][0]
        
        details = trait_map.get(task_name, {})
        result_text = details.get(final_label, final_label)
        
        report += f"### {details.get('emoji', '➡️')} {details.get('name', task_name)}\n"
        report += f"**您的倾向是: {result_text}**\n\n---\n"

    report += "\n\n✨ *这只是一个基于对话的初步参考哦* ✨"
    return report

def clear_chat(chat_history, user_inputs_state):
    return [], [], []

# --- 3. 全新美化版 Gradio 界面 ---

with gr.Blocks(
    title="温馨对话分析", 
    theme=gr.themes.Soft(
        primary_hue="blue"
    ).set(
        body_background_fill="#EBF5FF",  # 直接通过主题设置淡蓝色背景
        block_background_fill="white",   # 将聊天/报告等组件背景设为白色
        block_border_width="1px",        # 为组件添加细边框
        block_border_color="#D4E6F1",    # 边框颜色
        button_primary_background_fill="#85C1E9", # 统一按钮颜色
        button_primary_background_fill_hover="#A9CCE3",
        button_secondary_background_fill="#D4E6F1",
        button_secondary_background_fill_hover="#EAF2F8",
    )
) as demo:
    gr.Markdown("""
    <div style='text-align:center; font-size:2em; color:#1B4F72; margin-bottom:0.5em;'>
        欢迎来到小伙伴聊天屋 🧸✨
    </div>
    """)
    gr.Markdown("""
    <div style='text-align:center; color:#21618C; font-size:1.1em; margin-bottom:1em;'>
        你可以和我聊任何事哦~ 我会一直陪着你。
    </div>
    """)
    with gr.Row():
        gr.Image(value="fox.png", show_label=False, interactive=False, height=300, elem_id="left-img") # 可爱小狐狸
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="聊天窗口", height=500, avatar_images=(None, "ai_avatar.png")) # AI头像
            msg = gr.Textbox(label="", placeholder="在这里输入...", show_label=False, container=False)
        with gr.Column(scale=1):
            report_output = gr.Markdown(label="分析报告", elem_id="report-output")
            with gr.Row():
                 analyze_btn = gr.Button("📊 生成分析报告", elem_id="analyze-btn")
                 clear_btn = gr.Button("🧹 清空对话", elem_id="clear-btn")
            gr.Image(value="bear.png", show_label=False, interactive=False, height=300, elem_id="right-img") # 可爱小熊
            
    gr.Markdown("""
    <div style='text-align:center; color:#21618C; font-size:0.9em; margin-top:2em;'>
        - Made with Love for Awesome Kids -
    </div>
    """)

    user_inputs_history = gr.State([])
    
    msg.submit(respond, [msg, chatbot, user_inputs_history], [msg, chatbot, user_inputs_history])
    analyze_btn.click(analyze_history, [user_inputs_history], [report_output])
    clear_btn.click(clear_chat, [chatbot, user_inputs_history], [chatbot, user_inputs_history])


if __name__ == "__main__":
    demo.launch() 