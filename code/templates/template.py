from typing import List, Dict


class BaseTemplate():
    """
    Example:
    base_prompt = BaseTemplate("You will be given a chat history and a question. You have to generate a response to the question.\n Chat History: {chat_history}\n Question: {question}\n", ["chat_history", "question"])
    prompt = base_prompt.get_prompt({"chat_history": "A: Hello, how are you?\nB: I am fine, thank you.", "question": "What is the weather like today?"})
    print(prompt)   # Output: You will be given a chat history and a question. You have to generate a response to the question.\n Chat History: A: Hello, how are you?\nB: I am fine, thank you.\n Question: What is the weather like today?
    """
    def __init__(self, template: str, placeholders: List[str]):
        self.template = template
        self.placeholders = placeholders

    def get_prompt(self, placeholders_values: Dict[str, str]):
        if set(placeholders_values.keys()) != set(self.placeholders):
            raise ValueError(f"Expected placeholders: {self.placeholders}. Got: {list(placeholders_values.keys())}")
        return self.template.format(**placeholders_values)

def get_chat_history(
    messages: List[Dict[str, str]], eos: str = "</s>", bos: str = "<s>",
    instruct_start_token: str = "[INST]", instruct_end_token: str = "[/INST]"
):
    """
    Example:
    messages = [
        {"role": "user", "message": "Hello, how are you?"},
        {"role": "assistant", "message": "I am fine, thank you."},
        {"role": "user", "message": "What is the weather like today?"}
    ]
    chat_history = get_chat_history(messages)
    print(chat_history) # Output: <s>[INST] User: Hello, how are you? [/INST] Assistant: I am fine, thank you. </s> [INST] User: What is the weather like today? [/INST]
    """
    chat_history = f"{bos}"
    for message in messages:
        if message["role"] == "user":
            chat_history += f"{instruct_start_token} User: {message['message']} {instruct_end_token}\n"
        elif message["role"] == "assistant":
            chat_history += f"Assistant: {message['message']} {eos}\n"
    return chat_history
        
    