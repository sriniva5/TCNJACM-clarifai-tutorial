import os

from clarifai.rest import ClarifaiApp
from clarifai.rest import Video as ClVideo

#Set the environment variable to the API key from your account
os.environ["CLARIFAI_API_KEY"] = "INSERT_KEY_HERE"

#Test to see if the environment variable is set
#print(os.environ)
#print(os.environ["CLARIFAI_API_KEY"])

#Set the api key to the environment variable
api_key = os.environ["CLARIFAI_API_KEY"]
app = ClarifaiApp(api_key=api_key)

#Use general model
#Inspired by: https://blog.clarifai.com/tutorial-automatic-object-annotation-of-video
model = app.models.get("general-v1.3")

#Feel free to replace the video/gif with your own
#Pass video to prediction model -> get json of output
video = ClVideo(url="https://media.giphy.com/media/COoFzhnusyUVy/giphy.gif")
json = model.predict([video])

#Represent frames in video
frames = json["outputs"][0]["data"]["frames"]

#Parse frames and return concepts in a frame at a particular time (in seconds)
#Concepts with a confidence score > .85 are extracted
for x in range(len(frames)):
    #convert time to seconds
    timeInSeconds = frames[x]["frame_info"]["time"]/1000
    concepts = frames[x]["data"]["concepts"]
    for i in range(len(concepts)):
        # filters out confidence scores <.85
        if concepts[i]["value"] > .85:
            print("{}s {}".format(timeInSeconds, concepts[i]["name"]))
