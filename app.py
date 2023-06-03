from flask import Flask, request, jsonify
from informationExtraction.infoExtraction import Models_Load, go_a_head, get_abstracts
from summarization.Final_Stage_Summarization import summarize_text

app = Flask(__name__)
x101model = None
trfmodel = None


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'message': 'No file found in the request'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No file selected'})

    # save the list of files
    pdf_files = request.files.getlist('file')
    for pdf_file in pdf_files:
        pdf_file.save('temp.pdf')
    return jsonify({'message': 'Files uploaded successfully'})


@app.route('/generate_citation', methods=['GET'])
def extract_info():
    global x101model, trfmodel
    if x101model is None or trfmodel is None:
        x101model, trfmodel = Models_Load()

    pdf_files = request.files.getlist('file') # get the list

    apa_citations = []
    mla_citations = []
    chicago_citations = []
    for pdf_file in pdf_files:
        apa_citation, mla_citation, chicago_citation = go_a_head(pdf_file, x101model, trfmodel)
        apa_citations.append(apa_citation)
        mla_citations.append(mla_citation)
        chicago_citations.append(chicago_citation)

    response = {
        'apa': apa_citations,
        'mla': mla_citations,
        'chicago': chicago_citations
    }

    return jsonify(response)


@app.route('/summarize', methods=['GET'])
def perform_summarization():
    text = get_abstracts()  

    summary = summarize_text(text)

    response = {'summary': summary}

    return jsonify(response)


if __name__ == '__main__':
    app.run()
