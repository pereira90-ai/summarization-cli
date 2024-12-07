from transformers import pipeline
import torch
import PyPDF2
import sys


def read_pdf(file_path):  
    """Reads a PDF file and extracts the text from it."""  
    with open(file_path, "rb") as f:  
        reader = PyPDF2.PdfReader(f)  
        text = ""  
        for page in range(len(reader.pages)):  
            text += reader.pages[page].extract_text()  
    return text 

def main(inpu_pdf, output_txt, MAX_NEW_TOKENS=256):
    # Load the lightweight text model
    model_id = "pretrained/llama-3.2-3b"
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    text = read_pdf(inpu_pdf)
    messages = [
        {"role": "user", "content": text + 'Summarize the above sentence.'},
    ]

    outputs = pipe(
        messages,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    response = outputs[0]["generated_text"][-1]["content"]
    print(response)

    with open(output_txt, 'w') as file:
        file.write(response)


if __name__ == "__main__":  
    if len(sys.argv) < 3:  
        print("Usage: python app.py <input_pdf_file> <output_txt_file>")  
        sys.exit(1)  
    
    input_pdf = sys.argv[1]  
    output_txt = sys.argv[2]  
    if len(sys.argv) == 4 and type(sys.argv[3]) == int:
        main(input_pdf, output_txt, sys.argv[3])
    else:
        main(input_pdf, output_txt)