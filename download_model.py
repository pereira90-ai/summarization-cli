from huggingface_hub import login  
import shutil  
import os

HUGGINGFACE_API_KEY = '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'

login(token=HUGGINGFACE_API_KEY)
from huggingface_hub import hf_hub_download  

def model_download(filename = 'model-00001-of-00002.safetensors'):
    repo_id = 'meta-llama/Llama-3.2-3B'  # Replace with the actual repo ID  
    dest_dir_path = 'pretrained/llama-3.2-3b'

    destination_file_path = os.path.join(dest_dir_path, filename)  

    # Download the file  
    file_path = hf_hub_download(repo_id=repo_id, filename=filename)  

    os.makedirs(dest_dir_path, exist_ok = True)
    shutil.move(file_path, destination_file_path)

if __name__ == '__main__':
    # model_download()
    model_download('model-00002-of-00002.safetensors')