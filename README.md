##1. Install requirements

#RUN: pip install -r requirements.txt

##2. Training

#RUN: SegNet_PyTorch_v2.ipynb

NOTE: For Testing, the test images are saved in "images" folder. The trained model is saved in "weights" folder

##3. Flask test server

#RUN: python3 SegNet_test.py <PORT>

where, <PORT> (optional argument, DEFAULT VALUE = 5000) -> the port on which you want the flask server to run.

##4. Front end

NOTE: First start flask server

#(On Browser) RUN: http://127.0.0.1:5000

Test on images saved in "images" folder.
