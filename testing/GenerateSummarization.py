import requests

url = 'http://localhost:5000/summarize'

# Create a file object for the PDF file you want to summarize.
file = open('myfile.pdf', 'rb')

# Create a request object and set the `files` parameter to the file object.
request = requests.post(url, files={'file': file})

# Check the response status code.
if request.status_code == 200:
    response = request.json()
    print('Summary generated successfully!')
    print(response)
else:
    print('Error generating summary.')
