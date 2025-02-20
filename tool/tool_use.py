from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel
import numpy as np
import json
import ipdb


class ChatBot:
    def __init__(self, model_path, lora_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )
        if lora_path:
            self.model = PeftModel.from_pretrained(self.model, lora_path).to(
                self.device
            )
        self.system_prompt = """
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. 
The assistant first thinks about the reasoning process in the mind and then provides the user 
with the answer. The reasoning process and answer are enclosed within <think> </think> and 
<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> 
<answer> answer here </answer>.

The Assistant can also use tools to solve problems. Here are the available tools:
{tools}

When using a tool, the Assistant should follow these rules:
1. If a tool is needed, the Assistant should return a strict JSON format:
{{
  "tool": "tool_name",
  "params": {{"param1": "value1", ...}}
}}
2. If no tool is needed, the Assistant should directly provide the answer.

Examples:
User：Can you perform linear regression on these data points? x = [1, 2, 3, 4], y = [2, 4, 6, 8.001]
Assistant：
<think> I need to use the linear_regression tool to compute linear regression. I should clear about the tool name and params. </think>
<answer> Yes, I can perform linear regression on the given data points. Here are the results:
<tool_use>
{{
  "tool": "linear_regression",
  "parameters": {{"x": [1, 2, 3, 4], "y": [2, 4, 6, 8.001]}}
}}
</tool_use>
</answer>
""".format(
            tools=json.dumps(tools, ensure_ascii=False)
        )

    def generate_response(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        output = ""
        with torch.inference_mode():
            generated_ids = self.model.generate(
                input_ids,
                max_new_tokens=2048,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=0.6,
            )
            output = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0][len(prompt) :]

        return output

    def tool_use(tool_call:dict):
        """
        使用工具
        tool_call: 一次工具调用的json
        """
        if tool_call["tool"] == "linear_regression":
            x = tool_call["params"]["x"]
            y = tool_call["params"]["y"]
            slope, intercept = linear_regression(x, y)
            return {"slope": slope, "intercept": intercept}
        else:
            raise ValueError(f"Unknown tool: {tool_call['tool']}")

    def reinit_chat_history(self, chat_history):
        chat_history = [self.system_prompt]
        return chat_history

    def chat(self):
        chat_history = []
        chat_history.append(self.system_prompt)
        while True:
            print("Type 'exit' or 'quit' to end the chat.")
            print("-" * 30)
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            if user_input == "":
                print("Please enter a question.")
                continue
            if user_input == "reset":
                chat_history = self.reinit_chat_history(chat_history)
                print("Chat history reset.")
                continue
            user_input = "User: " + user_input + "<think>"
            chat_history.append(user_input)
            prompt = "\n".join(chat_history)
            response = self.generate_response(prompt).split("Assistant:")[-1]

            print("-" * 30)
            print(response)
            # response_wo_think = response.split("</think>")[-1]
            response = "Assistant: " + response
            chat_history.append(response)
            # ipdb.set_trace()


def linear_regression(x, y):
    """
    linear regression tool
    :param x: list
    :param y: list
    :return: Regression coefficient (slope, intercept)
    """
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # 计算斜率 (slope) 和截距 (intercept)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean

    return slope, intercept


linear_regression_tool = {
    "name": "linear_regression",
    "description": "对给定的数据点进行线性回归，返回斜率和截距",
    "parameters": {
        "x": {"type": "array", "description": "自变量数组"},
        "y": {"type": "array", "description": "因变量数组"},
    },
}

tools = {"linear_regression": linear_regression_tool}



if __name__ == "__main__":
    model_path = "/model/DeepSeek-R1-Distill-Qwen-14B"
    chatbot = ChatBot(model_path, None)
    chatbot.chat()
