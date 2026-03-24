from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader,TextLoader,CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader

def load_all_documents(data_dir: str)->List[Any]:
    """ 
Load all support files from the data directory and convert to langchain Structure
Support : PDF, TXT, CSV, Excel, WORD, JSON

"""
    #Use Project root data Folder
    data_path =Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")
    documents =[]
    
    #PDF files
    
    pdf_files = list(data_path.glob('**/*.pdf'))
    print(f"[DEBUG] Found {len(pdf_files)} PDF Files : {[str(f) for f in pdf_files]}")
    
    for pdf_file in pdf_files: 
        print(f"[DEBUG] Loading PDF : {pdf_file}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} Pages of PDF docs from {pdf_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[Error] Failed to load PDF {pdf_file}: {e}")
    
    
    # TXT Files
    
    txt_files = list(data_path.glob('**/*.txt'))
    print(f"[DEBUG] Found {len(txt_files)} TXT Files : {[str(f) for f in txt_files]}")
    
    for txt_file in txt_files: 
        print(f"[DEBUG] Loading TXT  : {txt_file}")
        try:
            loader = TextLoader(str(txt_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} Pages of TXT docs from {txt_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[Error] Failed to load TXT {txt_file}: {e}")

    
    # CSV Files
    
    csv_files = list(data_path.glob('**/*.csv'))
    print(f"[DEBUG] Found {len(csv_files)} CSV Files : {[str(f) for f in csv_files]}")
    
    for csv_file in csv_files: 
        print(f"[DEBUG] Loading CSV : {csv_file}")
        try:
            loader = CSVLoader(str(csv_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} Pages of CSV docs from {csv_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[Error] Failed to load CSV {csv_file}: {e}")
    

    # DOCS Files
    
    doc_files = list(data_path.glob('**/*.docx'))
    print(f"[DEBUG] Found {len(doc_files)} DOC Files : {[str(f) for f in doc_files]}")
    
    for doc_file in doc_files: 
        print(f"[DEBUG] Loading DOC : {doc_file}")
        try:
            loader = Docx2txtLoader(str(doc_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} Pages of DOC docs from {doc_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[Error] Failed to load DOC {doc_file}: {e}")
            
            
               
    # Excel Files
    
    xlsx_files = list(data_path.glob('**/*.xlsx'))
    print(f"[DEBUG] Found {len(xlsx_files)} Excel Files : {[str(f) for f in xlsx_files]}")
    
    for xlsx_file in xlsx_files: 
        print(f"[DEBUG] Loading Excel : {xlsx_file}")
        try:
            loader = UnstructuredExcelLoader(str(xlsx_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} Pages of EXCEL docs from {xlsx_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[Error] Failed to load EXCEL {xlsx_file}: {e}")


    # JSON Files
    
    json_files = list(data_path.glob('**/*.json'))
    print(f"[DEBUG] Found {len(json_files)} JSON Files : {[str(f) for f in json_files ]}")
    
    for json_file in json_files: 
        print(f"[DEBUG] Loading JSON : {json_file}")
        try:
            loader = JSONLoader(str(json_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} Pages of JSON docs from {json_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[Error] Failed to load JSON {json_file}: {e}")
            
    return documents
    