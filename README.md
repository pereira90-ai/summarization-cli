# LLama 3.2 Summarizor

This is just the simple summarization tool for evaluating the LLama3 model's performance using cli commands.


## How to use

### Clone the repository from the github repository:
```git
git clone https://github.com/pereira90-ai/llama3-cli-summarizor

cd llaba3-cli-summarizor
```

### Environment Setup

- Install conda environment from `yaml` file: 
```python
conda env create -f environment.yaml -n summarizor
conda activate summarizor
```

- Install cuda enabled torch on the conda environment:
```python
pip uninstall torch -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Download LLama 3.2 model from huggingface
First of all make sure you are in the project directory: `llama3-cli-summarizor`
And then: 
```bash
python download_model.py
```
### Run the summarizor with command
You can use text or PDF file as an input
```python
python app.py <your_pdf.txt> <output_file.txt> <max_token_len>
```

Sample usage:
```python
python app.py meeting_summary.pdf out.txt 256
```
Here `your_pdf.txt` is the path of your meeting log file to summarize while `output_file.txt` is the path of the result.
And `max_token_len` is the optional parameter to change the length of the summary length.

## Enjoy!