try:
    import re
    import os
    import cv2
    import spacy
    import string
    import spacy.cli
    import numpy as np
    import pytesseract
    import pandas as pd
    import torch.nn as nn
    import en_core_web_trf
    import pybtex.database
    import spacy_transformers
    import layoutparser as lp
    #-----------------------------------#
    import traceback
    import multiprocessing
    from tqdm import tqdm
    import pdf2image
    import pdfplumber
    from PIL import Image
    from docx import Document
except Exception as e:
    # handle the exception here
    print(f"An error occurred while importing libraries: {e}")



# pdf splitter >> split the images
def pdf_Splitter(pdf_file, data_dir):

    try:
        pdf_images = pdf2image.convert_from_path(os.path.join(data_dir, pdf_file))
    except Exception as e:
      print("Error during PDF conversion:")
      print(traceback.format_exc())
      return
    
    try:
        pdf = pdfplumber.open(os.path.join(data_dir, pdf_file))
    except Exception as e:
      print("Error occurred while opening the PDF file:")
      print(traceback.format_exc())  
      return
    
    for page_id in tqdm(range(len(pdf.pages))):

        img_out='/imageOut/pdf_img'
        if not os.path.exists(img_out):
            os.makedirs(img_out)

        img_path= (img_out+'/'+pdf_file.replace('.pdf', '') + '_{}_ori.jpg'.format(str(page_id)))
        pdf_images[page_id].save(img_path)

    return img_out #give me  the images dirc



# add boxes in the words and add there types 
def preprocess(image_path):

    image = Image.open(image_path)
    image = image.convert("RGB")

    width, height = image.size
    w_scale = 1000 / width
    h_scale = 1000 / height
    ocr_df = pytesseract.image_to_data(image, output_type='data.frame')#check done
    ocr_df = ocr_df.dropna().assign(left_scaled=ocr_df.left * w_scale,
                    width_scaled=ocr_df.width * w_scale,
                    top_scaled=ocr_df.top * h_scale,
                    height_scaled=ocr_df.height * h_scale,
                    right_scaled=lambda x: x.left_scaled + x.width_scaled,
                    bottom_scaled=lambda x: x.top_scaled + x.height_scaled)
    float_cols = ocr_df.select_dtypes('float').columns
    ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
    ocr_df = ocr_df.replace(r'^\s*$!', np.nan, regex=True)
    ocr_df = ocr_df.dropna().reset_index(drop=True)
    words = list(ocr_df.text)
    return image, words


# load the images here

def Models_Load():
  nlp_trf = spacy.load("en_core_web_trf")
  nlp_trf = en_core_web_trf.load()
  model = lp.Detectron2LayoutModel(config_path = "../models/X101/X101.yaml",model_path = "../models/X101/model.pth",label_map={0: "Abstract", 1: "Author", 2: "Caption", 3:"Date", 4:"Equation",5: "Figure", 6: "Footer", 7: "List", 8:"Paragraph", 9:"Reference", 10: "Section", 11: "Table", 12:"Title"})
  return nlp_trf,model

# extract the information
def extract_informations(dir_image,model):
  image = cv2.imread(dir_image)
  image = image[..., ::-1] 
  #********************************************************
  layout = model.detect(image) # You need to load the image somewhere else, e.g., image = cv2.imread(...)
  #lp.draw_box(image, layout,)
  lp.draw_box(image, layout)
  #************************************************************
  title_blocks = lp.Layout([b for b in layout if b.type=="Title"])
  author_blocks = lp.Layout([b for b in layout if b.type=="Author"])
  date_blocks = lp.Layout([b for b in layout if b.type=="Date"])
  abstract_blocks = lp.Layout([b for b in layout if b.type=="Abstract"])
  #Paragraph_blocks = lp.Layout([b for b in layout if b.type=="Paragraph"])
  #*************************************************************
  return title_blocks, author_blocks, date_blocks, abstract_blocks, image


def Decode_text(TextBlock ,image):
  text_blocks = lp.Layout([b for b in TextBlock])
  #*************************************************************
  h, w = image.shape[:2]
  left_interval = lp.Interval(0, w/2*1.05, axis='x').put_on_canvas(image)
  left_blocks = text_blocks.filter_by(left_interval, center=True)
  left_blocks.sort(key = lambda b:b.coordinates[1])

  right_blocks = [b for b in text_blocks if b not in left_blocks]
  right_blocks.sort(key = lambda b:b.coordinates[1])
  # And finally combine the two list and add the index
  # according to the order
  text_blocks = lp.Layout([b.set(id = idx) for idx, b in enumerate(left_blocks + right_blocks)])
  ocr_agent = lp.TesseractAgent(languages='eng') 
  for block in text_blocks:
    segment_image = (block
                       .pad(left=5, right=5, top=5, bottom=5)
                       .crop_image(image))
      
    text = ocr_agent.detect(segment_image)
    block.set(text=text, inplace=True)
    Decoded_Text=text_blocks.get_texts()

    return Decoded_Text

# extract the information from the decoded texts

def extract_emails(doc):
  emails=[]
  regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
  for w in doc:
    if w.pos_ == 'X':
      if(re.fullmatch(regex, w.text)):
        emails.append(w.text)
  return emails

def extract_author_name(doc):
  authors_names=[]
  for ent in doc.ents:
    if ent.label_ == 'PERSON':
      authors_names.append(ent.text)
  return authors_names

def extract_author_department(doc):
  authors_department=[]
  for ent in doc.ents:
    if ent.label_ == 'ORG':
      authors_department.append(ent.text)
  return authors_department

def extract_paper_date(doc):
  paper_date=[]
  for ent in doc.ents:
    if ent.label_ == 'DATE':
      paper_date.append(ent.text)
  return paper_date

# make the citaion 

def Authers_and_Departments(AutherFromX101,TrfModel):
  extractAuthorName=[]
  authorDepartment=[]
  #paperDate=[]
  doc_trf = TrfModel(AutherFromX101)
  AuthorName = extract_author_name(doc_trf)
  authorDepartment = extract_author_department(doc_trf)
  #paperDate_trf = extract_paper_date(doc_trf)
  return AuthorName,authorDepartment

# citation formate section

def make_citation_APA_format(authorName,authorDepartment,paperDate,paperTitle):
  final_citation=''
  for name in authorName:
    final_citation=final_citation + name +","
  
  final_citation=final_citation + "."

  for date in paperDate:
    final_citation = final_citation +"("+ date +")" +"."
  
  final_citation=final_citation + paperTitle + "."

  for departmen in authorDepartment:
    final_citation=final_citation + departmen +", "
  
  final_citation=final_citation + "."
  
  return final_citation

def make_citation_MLA_format(authorName,authorDepartment,paperDate,paperTitle):
  final_citation=''
  # first Author
  for name in authorName:
    final_citation=final_citation + name +","
  
  final_citation=final_citation + "."
  # the second Title
  final_citation=final_citation + paperTitle + "."

  # the third Department or Publisher or organization
  for departmen in authorDepartment:
    final_citation=final_citation + departmen +", "

  final_citation=final_citation + "."
  # the fourth Date
  for date in paperDate:
    final_citation = final_citation +"("+ date +")" +"."
  
  return final_citation

def make_citation_Chicago_format(authorName,authorDepartment,paperDate,paperTitle):
  final_citation=''
  for name in authorName:
    final_citation=final_citation + name +","
  
  final_citation=final_citation + "."

  for date in paperDate:
    final_citation = final_citation +"("+ date +")" +"."
  
  final_citation=final_citation + paperTitle + "."

  for departmen in authorDepartment:
    final_citation=final_citation + departmen +", "
  
  final_citation=final_citation + "."
  
  return final_citation


"""# Test Function to  test all citation format"""


def extract_information_from_image(directory, model101, modelTRF):
    extracted_information = []
    if os.path.exists(directory):
        file_list = os.listdir(directory)
        for filename in file_list:
            if filename.endswith('_0_ori.jpg') or filename.endswith('_0_ori.jpeg'):
                image_path = os.path.join(directory, filename)
                title, author, date, abstract, image = extract_informations(image_path, model101)
                # ------------------------------------
                Title = Decode_text(title, image)
                Title = ' '.join([a if a is not None else '' for a in Title])
                print("Title:", Title)
                TotalAuthor = Decode_text(author, image)
                TotalAuthor = ' '.join([a if a is not None else '' for a in TotalAuthor])
                print("TotalAuthor:", TotalAuthor)
                Date = Decode_text(date, image)
                if Date is not None:
                    Date = ' '.join([a if a is not None else '' for a in Date])
                else:
                    Date = str(Date)
                Abstract = Decode_text(abstract, image)
                Abstract = ' '.join([a if a is not None else '' for a in Abstract])
                # ------------------------------------
                Author, Department = Authers_and_Departments(TotalAuthor, modelTRF)
                extracted_information.append((Title, Author, Department, Date, Abstract))
    else:
        print(f"Directory '{directory}' does not exist.")
    return extracted_information

def get_abstracts(extracted_information):
    abstracts = [info[4] for info in extracted_information]
    return abstracts

def generate_citation(extracted_information):
    citation_APA = ''
    citation_MLA = ''
    citation_Chicago = ''
    for info in extracted_information:
        Title, Author, Department, Date, Abstract = info
        citation_APA += make_citation_APA_format(Author, Department, Date, Title)
        citation_MLA += make_citation_MLA_format(Author, Department, Date, Title)
        citation_Chicago += make_citation_Chicago_format(Author, Department, Date, Title)
    return citation_APA, citation_MLA, citation_Chicago


def GenerateAllCitationTEST(directory, model101, modelTRF):
    extracted_information = extract_information_from_image(directory, model101, modelTRF)
    citation_APA, citation_MLA, citation_Chicago = generate_citation(extracted_information)
    return citation_APA, citation_MLA, citation_Chicago


"""# The main Function 

"""

def go_a_head(pdf_data_dir, x101model, trfmodel):
      images_path =''
      All_Pdf_citation = []
      pdf_files = list(os.listdir(pdf_data_dir))
      pdf_files = [t for t in pdf_files if t.endswith('.pdf')]
      
      for pdf_file in pdf_files:
        images_path =pdf_Splitter(pdf_file, pdf_data_dir)
      

      print(images_path)
      citaion_APA,citaion_MLA,citaion_Chicago = GenerateAllCitationTEST(images_path ,x101model , trfmodel)
      return citaion_APA,citaion_MLA,citaion_Chicago
