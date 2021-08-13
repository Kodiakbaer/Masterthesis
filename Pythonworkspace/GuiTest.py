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
    rect = [tlc[0],tlc[1],brc[0],brc[1]]
    print(rect)
    holds.append(rect)
  #return [top_left_corner, bottom_right_corner]


# Read Images
image = cv2.imread("D:\MMichenthaler\VideoFrames\Video2\Video2_frame1000.jpg")
# Make a temporary image, will be useful to clear the drawing
temp = image.copy()
# Create a named window

cv2.namedWindow("Window")
# highgui function called when mouse events occur
holds = []
cv2.setMouseCallback("Window", mark_rectangel, holds)
tlc = []
brc = []


k = 0
# Close the window when key q is pressed
while k!=113:
  # Display the image
  cv2.imshow("Window", image)
  k = cv2.waitKey(0)
  # If c is pressed, clear the window, using the dummy image
  if (k == 99):
    image= temp.copy()
    cv2.imshow("Window", image)
  if (k == 32):
    print(holds)
  if (k == 115):
    f = open("holds.txt", "w")
    f.write(str(holds))

cv2.destroyAllWindows()