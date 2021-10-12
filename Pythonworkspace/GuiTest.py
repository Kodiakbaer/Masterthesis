# Import packages
import cv2
import numpy as np

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
    rect = [tlc[0]*OrigWidth/900, tlc[1]*OrigHeight/1600, brc[0]*OrigWidth/900, brc[1]*OrigHeight/1600]
    print(rect)
    holds.append(rect)
  #return [top_left_corner, bottom_right_corner]


# Read Images
image = cv2.imread("D:/MMichenthaler/Data_15-09-2021_Workspace/NewVideo2/NewVideo2_frame190.jpg")
# Make a temporary image, will be useful to clear the drawing
global OrigWidth
global OrigHeight
OrigWidth = image.shape[1]
OrigHeight = image.shape[0]
image = cv2.resize(image, (900, 1600), interpolation = cv2.INTER_AREA)
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
    image= temp.copy()
    cv2.imshow("Window", image)
  if (k == 32):               # key press " "
    print(holds)
  if (k == 115):              # key press "s"
    f = open("holds.txt", "w")
    f.write(str(holds))

cv2.destroyAllWindows()