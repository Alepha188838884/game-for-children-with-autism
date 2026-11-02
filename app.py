from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import requests
import json

app = Flask(__name__)

# 模型和分词器的全局变量，初始为None
chat_tokenizer = None
chat_model = None

# 存储聊天记录
chat_history = []

def load_chat_model():
    """加载聊天模型，如果尚未加载。"""
    global chat_tokenizer, chat_model
    if chat_tokenizer is None or chat_model is None:
        print("Loading chat model for the first time...")
        chat_tokenizer = AutoTokenizer.from_pretrained('uer/gpt2-chinese-cluecorpussmall')
        chat_model = AutoModelForCausalLM.from_pretrained('uer/gpt2-chinese-cluecorpussmall')
        print("Chat model loaded.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    chat_history.append(f"你: {user_input}")

    OLLAMA_API_URL = "http://localhost:11434/api/chat"
    OLLAMA_MODEL_NAME = "chevalblanc/gpt-4o-mini"
    bot_response = ""

    try:
        messages = [{'role': 'user' if line.startswith('你:') else 'assistant', 'content': line.split(':', 1)[1].strip()} for line in chat_history]
        data = {'model': OLLAMA_MODEL_NAME, 'messages': messages, 'stream': False}
        response = requests.post(OLLAMA_API_URL, json=data, timeout=20)
        response.raise_for_status()
        bot_response = response.json()['message']['content'].strip()
    except requests.exceptions.RequestException as e:
        print(f"Ollama call failed: {e}. Falling back to local model.")
        load_chat_model() # 确保备用模型已加载
        history_text = "\n".join(chat_history)
        input_ids = chat_tokenizer.encode(history_text, return_tensors='pt')
        output_sequences = chat_model.generate(
            input_ids=input_ids,
            max_length=len(input_ids[0]) + 50,
            temperature=0.8, top_k=50, top_p=0.95,
            repetition_penalty=1.2, do_sample=True, num_return_sequences=1
        )
        generated_text = chat_tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        bot_response = generated_text.split("\n")[-1].replace('机器人:', '').strip()

    chat_history.append(f"机器人: {bot_response}")
    return jsonify({'response': bot_response})

@app.route('/analyze', methods=['POST'])
def analyze():
    full_chat = "\n".join(chat_history)

    if not full_chat or len(chat_history) < 2: # 增加判断，至少有1轮对话
        return jsonify({'error': '聊天记录过少，无法进行有效分析。请至少进行一轮以上对话。'}), 400

    OLLAMA_API_URL = "http://localhost:11434/api/chat"
    OLLAMA_MODEL_NAME = "chevalblanc/gpt-4o-mini"

    # 构建分析提示
    analysis_prompt = f"""
    你现在是一位专业的心理分析师。请根据以下对话记录，对用户的“大五人格特质” (Big Five personality traits) 进行分析。
    大五人格特质包括：
    1.  开放性 (Openness): 想象力、创造力、对新事物的接受程度。
    2.  尽责性 (Conscientiousness): 条理性、责任心、自律能力。
    3.  外向性 (Extraversion): 社交性、活力、乐观程度。
    4.  宜人性 (Agreeableness): 合作性、同情心、信任他人。
    5.  神经质 (Neuroticism): 情绪稳定性、焦虑、易怒程度。

    对话记录如下：
    ---
    {full_chat}
    ---

    请完成以下任务：
    1.  为用户的每项特质给出一个百分比得分（例如：'75%'）。
    2.  提供一个简洁的总体摘要（不超过100字），解释你的分析依据。
    3.  你的回答必须是纯粹的JSON格式，不包含任何额外的解释或标记。JSON对象应包含以下键：'openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism', 'summary'。

    示例输出格式：
    {{
        "openness": "60%",
        "conscientiousness": "75%",
        "extraversion": "55%",
        "agreeableness": "65%",
        "neuroticism": "40%",
        "summary": "用户在对话中表现出较强的责任心和条理性，但在社交和新体验方面略显保守。情绪整体稳定。"
    }}
    """

    try:
        messages = [{'role': 'user', 'content': analysis_prompt}]
        data = {'model': OLLAMA_MODEL_NAME, 'messages': messages, 'stream': False, 'format': 'json'}
        response = requests.post(OLLAMA_API_URL, json=data, timeout=60) # 增加超时时间
        response.raise_for_status()
        
        # Ollama在设置format='json'时，返回的content就是一个JSON字符串
        report_str = response.json()['message']['content']
        report = json.loads(report_str)
        return jsonify(report)

    except requests.exceptions.RequestException as e:
        print(f"Ollama call for analysis failed: {e}")
        return jsonify({'error': '调用分析服务失败，请检查Ollama服务是否正常运行。'}), 500
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Failed to parse analysis response: {e}")
        return jsonify({'error': '分析结果解析失败，模型可能返回了非预期的格式。'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)