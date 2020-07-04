# ImageToExcel
ImageToExcel is a python script to convert image based table into a spreadsheet. It uses Google Cloud Vision API.

Please visit: https://cloud.google.com/vision/docs/quickstart-client-libraries for instructions on how to setup the api and download the JSON api key file needed to run the script.

Once you download the key file, update: os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = " path to json file containing api credentials" in image2excel.py on line 17. Provide the path to your image containing the table by replacing the value: file = r'table.jpg' in image2excel.py on line 132

Run the image2excel.py, and it will generate the excelsheet to a file output.xlsx
