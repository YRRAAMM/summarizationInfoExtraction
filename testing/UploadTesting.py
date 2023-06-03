import requests

url = 'http://localhost:5000/upload'

# Create a file object for the PDF file you want to upload.
file = open('/media/elit/2CA4481AA447E4C41/Users/youssif/Desktop/D folder/college/GP/flask projects/GP server/TheApplication/uploads/1802.01168 (1) (1).pdf', 'rb')

# Create a request object and set the `files` parameter to the file object.
request = requests.post(url, files={'file': file})

# Check the response status code.
if request.status_code == 200:
    print('File uploaded successfully!')
else:
    print('Error uploading file.')
