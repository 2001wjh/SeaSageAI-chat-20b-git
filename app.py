import gradio as gr
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

# download SeaSageAI_InternLM2 to the base_path directory using git tool
base_path = './SeaSageAI_InternLM2'
os.system(f'git clone https://code.openxlab.org.cn/wjh2001/SeaSageAI_InternLM2.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')

tokenizer = AutoTokenizer.from_pretrained(base_path,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_path,trust_remote_code=True, torch_dtype=torch.float16).cuda()

def chat(message,history):
    for response,history in model.stream_chat(tokenizer,message,history,max_length=2048,top_p=0.7,temperature=1):
        yield response

gr.ChatInterface(chat,
                 title="SeaSageAI_InternLM2-20B",
                description="""
InternLM is mainly developed by Shanghai AI Laboratory, finetuning by wjh.  
                 """,
                 ).queue(1).launch()
