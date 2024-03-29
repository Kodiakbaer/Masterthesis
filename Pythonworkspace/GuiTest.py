# Import packages
import cv2
import numpy as np
import csv

# Lists to store the bounding box coordinates
top_left_corner=[]
bottom_right_corner=[]

# function which will be called on mouse input
def mark_rectangle(action, x, y, flags, *userdata) :
  # Referencing global variables
  global top_left_corner, bottom_right_corner, tlc, brc
  # Mark the top left corner when left mouse button is pressed
  if action == cv2.EVENT_LBUTTONDOWN:
    top_left_corner = [(x,y)]
    # When left mouse button is released, mark bottom right corner
    tlc = [x,y]
    print(tlc)
  elif action == cv2.EVENT_LBUTTONUP:
    bottom_right_corner = [(x,y)]
    # Draw the rectangle
    cv2.rectangle(image, top_left_corner[0], bottom_right_corner[0], (0,255,0),2, 8)
    cv2.imshow("Window",image)
    brc = [x,y]
    print(brc)
    rect = [int(round(tlc[0]*OrigWidth/725, 0)), int(round(tlc[1]*OrigHeight/1288, 0)),
            int(round(brc[0]*OrigWidth/725, 0)), int(round(brc[1]*OrigHeight/1288, 0))]
    print(rect)
    holds.append(rect)
  #return [top_left_corner, bottom_right_corner]


# Read Images
image = cv2.imread("D:/MMichenthaler/Data_15-09-2021_Workspace/NewVideo1/NewVideo1_frame250.jpg")
holdsPath = 'D:/MMichenthaler/HandOverHold/ScriptTestNewData2/'

# Make a temporary image, will be useful to clear the drawing
global OrigWidth
global OrigHeight
OrigWidth = image.shape[1]
OrigHeight = image.shape[0]
image = cv2.resize(image, (725, 1288), interpolation = cv2.INTER_AREA)
temp = image.copy()
# Create a named window

cv2.namedWindow("Window")
# highgui function called when mouse events occur
holds = []
cv2.setMouseCallback("Window", mark_rectangle, holds)
tlc = []
brc = []


k = 0
# Close the window when key q is pressed
while k!=113:
  # Display the image
  cv2.imshow("Window", image)
  k = cv2.waitKey(0)

  if (k == 99):               # If c is pressed, clear the window, using the dummy image
    image = temp.copy()
    tempHolds = []
    cv2.imshow("Window", image)
  if (k == 32):               # key press " "q
    print(holds)
  if (k == 115):              # key press "s"
    with open(holdsPath + 'holds2.csv', 'w') as f:
      # create the csv writer
      writer = csv.writer(f)

      # write a row to the csv file
      writer.writerow(holds)


with open(holdsPath + "holds2.csv", "r") as file:
  holdsSt = []
  stHolds = list(csv.reader(file, delimiter = ','))
  print(stHolds)
  for elem in stHolds:
    for elem2 in elem:
      elem3 = elem2.replace('[', '')
      elem4 = elem3.replace(']', '')
      holdsSt.append(elem4.split(','))

  holds = [list(map(int,rec)) for rec in holdsSt]



  #for row in csv_reader:
  #  holds.append(row)


print(holds)

cv2.destroyAllWindows()