import cv2
import numpy as np

# initialize the camera
cam = cv2.VideoCapture(0)

# Write cam feed to a file
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output.avi", fourcc, 20.0, (640, 480))

while True:
    ret, frame = cam.read()
    
    # Em escala de cinzentos
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # draw a line example
    #cv2.line(frame, (0,0), (640,640), (0,0,255), 10)
    #cv2.line(frame, (0,640), (640,0), (0,0,255), 10)
    
    # Write feed to a file
    out.write(frame)
    
    
    
    # Open cam feed window
    cv2.imshow("cam window", frame)
    #cv2.imshow("gray cam window", gray)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cam.release()
out.release()
cv2.destroyAllWindows()
    
    
