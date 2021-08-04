import cv2

vidcap = cv2.VideoCapture("D:\Studium\Master\MasterArbeit\Bilder_26-05-2021_Raw\Videos\P5261481.MOV")
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("D:\Studium\Master\MasterArbeit\Bilder_26-05-2021_Workspace\VideoFrames\Video5\Video5_frame%d.jpg" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
