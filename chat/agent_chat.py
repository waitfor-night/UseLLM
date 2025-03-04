import os
import openai
from autogen import ConversableAgent
import logging

#logging.basicConfig(level=logging.DEBUG)


llm_config = {
    "config_list": [
        {
            "api_type": "openai",
            "model": "moonshot-v1-32k",
            "api_key": os.environ["KIMI_API_KEY"],
            "base_url": "https://api.moonshot.cn/v1",
            "temperature": 0.6,
            "max_tokens": 2048,
        }
    ]
}

chat_agent = ConversableAgent(
    name="helpful_agent",
    llm_config=llm_config,
    system_message="You are a poetic AI assistant, response in chinese.",
)
beginning_message = input("Please input your beginning message: ")
chat_agent.run(beginning_message)
