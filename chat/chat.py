from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel
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

    def reinit_chat_history(self, chat_history):
        system_prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. \
 The assistant first thinks about the reasoning process in the mind and then provides the user \
 with the answer. The reasoning process and answer are enclosed within <think> </think> and \
 <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> \
 <answer> answer here </answer>. "
        chat_history = [system_prompt]
        return chat_history

    def chat(self):
        chat_history = []
        system_prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. \
 The assistant first thinks about the reasoning process in the mind and then provides the user \
 with the answer. The reasoning process and answer are enclosed within <think> </think> and \
 <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> \
 <answer> answer here </answer>. "
        chat_history.append(system_prompt)
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
            # response_wo_think = response.split("</think>")[-1]
            response = "Assistant: " + response
            chat_history.append(response)
            print("-" * 30)
            print(response)


if __name__ == "__main__":
    # Example usage
    model_path = "/model/DeepSeek-R1-Distill-Qwen-14B"
    chatbot = ChatBot(model_path, None)
    chatbot.chat()
