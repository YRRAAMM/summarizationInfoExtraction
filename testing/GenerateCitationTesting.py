import requests

url = 'http://localhost:5000/generate_citation'

# Create a file object for the PDF file you want to extract citations from.
file = open('myfile.pdf', 'rb') # todo change this

# Create a request object and set the `files` parameter to the file object.
request = requests.post(url, files={'file': file})

# Check the response status code.
if request.status_code == 200:
    response = request.json()
    print('Citations generated successfully!')
    print(response)
else:
    print('Error generating citations.')
