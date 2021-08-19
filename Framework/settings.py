

## global variables

# The text detector. Currently the EAST trained Neural network
detector = None

# A list of all the chat messages extracted through ocr. See Chatlog for the datastructure
CHATLOG_ARRAY = []

# TestNumber = 0

##### begin settings that need to be changed   ######

# path to the install location of tesseract. Make sure to include the / at the end as well
TESSERACT_LOCATION = "D:/Tesseract-ocr/"

# Path to a .pb file contains trained detector network.'
MODEL_DETECTOR = "D:/StellarisAI/frozen_east_text_detection.pb"

# folder which contains all the videos you want to analyze.
# Make sure to include the / at the end as well
# video_location = "D:/PythonCVtest/DataSets/Youtube/HarroVideos2/"
VIDEO_LOCATION = "D:/PythonCVtest/DataSets/Youtube/crew/"

##### end settings that need to be changed   ######