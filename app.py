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

def main(inpu_pdf, output_txt, MAX_NEW_TOKENS=512):
    # Load the lightweight text model
    model_id = "pretrained/llama-3.2-3b"
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if inpu_pdf.split('.')[-1] == 'pdf':
        text = read_pdf(inpu_pdf)
    elif inpu_pdf.split('.')[-1] == 'txt':
        with open(inpu_pdf, 'r') as file:
            text_lines = file.readlines()
            include = True
            teacher_texts = []
            for text_line in text_lines:
                if "teacher:" in text_line.lower():
                    include = True
                elif 'student' in text_line.lower():
                    include = False 
                # print(str(include) + text_line)
                if include:
                    teacher_texts.append(text_line)

            text = '\n'.join(teacher_texts)

            # with open('teacher.txt', 'w') as file:
            #     file.write(text)
            # text = text.replace('Teacher:', '')
    else: 
        print("Input file must be one of two types: txt or pdf")        
        return
    messages = [
        {"role": "user", "content": 'You are an experienced summarizor. Summarize the following sentences: "' + text + '"'}
    ]
    # summary = pipe(text, min_length=25, do_sample=True)  
    # print(summary[0]['summary_text'])
    outputs = pipe(
        messages,
        max_new_tokens=MAX_NEW_TOKENS
    )

    response = outputs[0]["generated_text"][-1]["content"]
    print(response)

    with open(output_txt, 'w', encoding='utf-8') as file:
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