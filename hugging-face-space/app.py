import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from optimum.bettertransformer import BetterTransformer
import gradio as gr
import pypdf
import re

tokenizer = AutoTokenizer.from_pretrained(
    "facebook/bart-large-cnn",
    use_fast=True
)

model_hf = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/bart-large-cnn",
    torch_dtype=torch.bfloat16
)

model = BetterTransformer.transform(model_hf, keep_original_model=True)

def extract_abstract(pdf_path):
  with open(pdf_path, "rb") as f:
    reader = pypdf.PdfReader(f)
    text = reader.pages[0].extract_text(orientations=(0)) + reader.pages[1].extract_text(orientations=(0))
    text = text.replace("\n", "")
    
    abstract_regex = re.compile(r"Abstract|ABSTRACT", re.IGNORECASE)
    abstract_match = re.search(abstract_regex, text)
    if not abstract_match:
        return ""
    
    abstract_start = abstract_match.start() + 8
    introduction_regex = re.compile(r"Introduction|ntroduction|INTRODUCTION|NTRODUCTION")
    abstract_end = re.search(introduction_regex, text[abstract_start:]).start() + abstract_start

    if abstract_start != -1 and abstract_end != -1:
        return text[abstract_start:abstract_end]
    else:
        return ""

def summarize_abstract(pdf_path):
    abstract_text = extract_abstract(pdf_path)
    
    inputs = tokenizer(
        abstract_text,
        max_length=130,
        return_tensors="pt"
    )
    
    prediction = model.generate(**inputs)
    prediction = tokenizer.batch_decode(prediction, skip_special_tokens=True)
    
    return prediction[0]

demo = gr.Interface(
    fn=summarize_abstract,
    inputs=[gr.File(label="PDF path")],
    outputs=[gr.Textbox(label="Abstract summary")],
    description="""
    # BART Large CNN Abstract Summarization
    [Code](https://github.com/darylalim/bart-large-cnn-abstract-summarization)
    """
)

demo.queue()

demo.launch()