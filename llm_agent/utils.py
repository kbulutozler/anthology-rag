import re, pdfminer.high_level
def text_from_pdf(path):
    return pdfminer.high_level.extract_text(path)
def clean(txt): return re.sub(r"\s+"," ",txt).strip() 