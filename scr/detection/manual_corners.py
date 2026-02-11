import cv2
import numpy as np

def get_manual_corners(image: np.ndarray, pattern_size: tuple[int, int]):

    original = image.copy() # in case the user wants to reset 
    instance = image.copy()

    clicked_points = []

    def click_event(event: int, x: int, y: int, flags: int, params):

        if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4: # if there are 4 clickes the next clicks are ignored
            
            clicked_points.append((x,y))
            cv2.circle(instance, (x,y), 5, (0, 0, 255), -1) # drawing a little red circel in the click aqrea 

            cv2.imshow("manual corners", instance) # showind the image updated 

    cv2.namedWindow("manual corners", cv2.WINDOW_NORMAL) 
    cv2.imshow("manual corners", instance)

    cv2.setMouseCallback("manual corners", click_event) 

    while True: # get all the four points
        
        key = cv2.waitKey(20) & 0xFF # waiting for the keyboard to press something

        if key == 27: # in case of ESC, we exit the program
            cv2.destroyWindow("manual corners")
            return None 

        if key == ord('r'): # in case the user presser 'r', we reset the image 
            clicked_points.clear()
            instance[:] = original # resetting each instance pixel to the original 
            cv2.imshow("manual corners", instance)
        
        if len(clicked_points) == 4: # all the corners have been pressed
            break
    
    cv2.destroyWindow("manual corners") # once we collected all the corners, we can kill the window

    # ordering the points

    pts = np.array(clicked_points, dtype=np.float32)
    
    s = pts.sum(axis=1) # summing x + y for each couple 
    diff = pts[:, 0] - pts[:, 1] # difference x - y for each couple 

    tl_idx = np.argmin(s)
    br_idx = np.argmax(s)
    tr_idx = np.argmin(diff)
    bl_idx = np.argmax(diff)

    tl = pts[tl_idx]
    br = pts[br_idx]
    tr = pts[tr_idx]
    bl = pts[bl_idx]

    # interpolling the grid
    cols, rows = pattern_size

    grid = []
    
    for j in range(rows):
        for i in range(cols):
            u = i/(cols-1)
            v = j/(rows-1)
            top = (1 - u) * tl + u * tr
            bottom = (1 - u) * bl + u * br
            p = (1 - v) * top + v * bottom
            grid.append(p)
    
    corners = np.array(grid, dtype=np.float32)
    corners = corners.reshape(-1, 1, 2)

    return corners






   

    


            
        
