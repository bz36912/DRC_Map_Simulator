"""running simAlgorithm runs the main(), which is different to the main() in mapSimulation.py
mapSimulation.py's main() is typically used and is the actual simulator
simAlgorithm.py's main() is used for debugging and code development.
"""

import cv2 as cv
#from cv2 import aruco
import numpy as np 
import sys
import matplotlib.pyplot as plt
import scipy as scipy
import time

np.set_printoptions(threshold=sys.maxsize)

EACH_STEP = True # This value is set in mapSimulation.py and used for debugging
def set_EACH_STEP(value:bool):
    global EACH_STEP
    EACH_STEP = value
    print("Enter/exit debug mode. EACH_STEP: ", EACH_STEP)

class TrackObject(object):
    #to select pixels of certain colour from a picture
    LOWER = 0
    UPPER = 1
    YELLOW_HSV = np.array([[20, 40, 70], [40, 255, 255]])
    BLUE_HSV = np.array([[100, 70, 80], [130, 255, 255]])

    #for debugging to annotate the track
    GREEN = (0, 255, 0)  # in BGR
    BLACK = (0, 0, 0)
    RED = (0, 0, 255)

    #index of array to select the type of track object
    LEFTTRACK = 0 #yellow
    RIGHTTRACK = 1 #blue
    OBSTACLE = 2 #purple
    CAR = 3 #red

    LIST_OF_TYPES = [LEFTTRACK, RIGHTTRACK]
    NUM_TYPES = len(LIST_OF_TYPES)

    colourBounds = {
        LEFTTRACK: BLUE_HSV,
        RIGHTTRACK: YELLOW_HSV,
    }

class Car(object):
    #maintain gap algorithm parameters
    # prevIsLeft = False
    state = "FOLLOW" #"TURN RIGHT", "TURN LEFT", "DRIVE STRAIGHT (LEFT)", "DRIVE STRAIGHT (RIGHT)"
    OPTIMAL_SIDE_DISTANCE = 40
    SEGMENT_SIZE = 5 #be a factor of 180

    WINDOW_HEIGHT = 240
    WINDOW_WIDTH = 200
    MAX_Y = round(WINDOW_HEIGHT / 2)
    MAX_X = round(WINDOW_WIDTH / 2)
    MAX_RANGE = np.sqrt(MAX_X**2 + MAX_Y**2)

obj = TrackObject()
car = Car()      

def points_from_canny(mask):
    canny = cv.Canny(mask, 60, 180)
    if EACH_STEP:
        cv.imshow('canny',canny)
        dummy = 0 #just a filler
    filtered = np.nonzero(canny)
    return filtered[1], filtered[0]

def points_from_contour(mask):
    """turns a line into contour/outline points
    This is an alternative method to points_from_skeleton()
    """
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # cv.drawContours(frame, contours, -1, (255, 0, 0), 3)

    contourArray = np.zeros((0, 2)) #just an empty array
    for c in contours:
        if cv.contourArea(c) > 100: #finds only the contours which are large (noise rejection)
            contourArray = np.concatenate((contourArray, np.squeeze(c)))

    # print("length of contourArray: ", len(contourArray))
    return contourArray[::, 0], contourArray[::, 1]

def blur_n_mask(frame, bounds):
    frame = cv.blur(frame, (6,6)) #to fill in small gaps/imperfections in the image
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, bounds[obj.LOWER], bounds[obj.UPPER])
    if EACH_STEP:
        # cv.imshow("mask, with bounds: " + str(bounds), mask)
        dummy = 0
    return mask

def split_into_two_cameras(frame):
    cropMask = np.zeros_like(frame)

    # rejects anypoints behind the car, since our car cannot see what is behind it
    height_cutoff = frame.shape[0] // 2
    cropMask[ :height_cutoff, :] = [255, 255, 255]
    cropMask[height_cutoff: , :] = [0, 0, 0]
    frontFrame = cv.bitwise_and(frame, cropMask)

    # splitting the image into left and right halves, to simulate that we have two cameras
    width_cutoff = frame.shape[1] // 2
    cropMask[ : , :width_cutoff] = [255, 255, 255]
    cropMask[ : , width_cutoff:] = [0, 0, 0]
    leftFrame = cv.bitwise_and(frontFrame, cropMask)
    cropMask[ :, :width_cutoff] = [0, 0, 0]
    cropMask[ :, width_cutoff:] = [255, 255, 255]
    rightFrame = cv.bitwise_and(frontFrame, cropMask)

    return leftFrame, rightFrame

def send_to_camera(frame, stepNum):
    leftFrame, rightFrame = split_into_two_cameras(frame)
    # cv.imshow('leftFrame', leftFrame)
    # cv.imshow('rightFrame', rightFrame)
    if EACH_STEP:
        # cv.imshow("left half pic" + str(stepNum), leftFrame)
        dummy = 0 #just a filler

    #initialising some numpy arrays. See explanation for how to use the variables in the documentation under the first heading
    leftX = np.zeros(obj.NUM_TYPES, dtype=np.ndarray)
    rightX = np.zeros(obj.NUM_TYPES, dtype=np.ndarray)
    leftY = np.zeros(obj.NUM_TYPES, dtype=np.ndarray)
    rightY = np.zeros(obj.NUM_TYPES, dtype=np.ndarray)
    
    for objType in obj.LIST_OF_TYPES:
        bounds = obj.colourBounds[objType]
        leftMask = blur_n_mask(leftFrame, bounds)
        rightMask = blur_n_mask(rightFrame, bounds)
        # express objects in picture into points with x,y coordinates
        leftX[objType], leftY[objType] = points_from_contour(leftMask)
        rightX[objType], rightY[objType] = points_from_contour(rightMask)

        # leftX[objType], leftY[objType] = points_from_canny(leftMask)
        # rightX[objType], rightY[objType] = points_from_canny(rightMask)

    return leftX, leftY, rightX, rightY

def SIM_get_target_position(frame, stepNum):
    # express objects in picture into points with x,y coordinates
    leftX, leftY, rightX, rightY = send_to_camera(frame, stepNum)

    # algorithm for processing coordinate. MOST of the code development is with the function below
    # targetX, targetY = quick_path_search(frame, stepNum, leftX[obj.LEFTTRACK], leftY[obj.LEFTTRACK], rightX[obj.RIGHTTRACK], rightY[obj.RIGHTTRACK])
    targetX, targetY = quick_path_search(frame, stepNum, leftX, leftY, rightX, rightY)

    #displays the  targetX, targetY in the car's frame
    if EACH_STEP:
        # correct for the offset, so the origin at the top left of the image
        (width, height) = (frame.shape[1], frame.shape[0])
        picTargetX = targetX + width / 2
        picTargetY = targetY + height / 2

        debugCroppedFrame = frame.copy() #do NOT annote on frame, which is reserved for analysis
        cv.circle(debugCroppedFrame, (round(picTargetX), round(picTargetY)), 7, obj.GREEN, -1)
        #to see which side is seen by the left camera, and which by the right camera:
        cv.line(debugCroppedFrame, (round(width / 2), 0), (round(width / 2), height), obj.RED, 1)
        cv.imshow("cropped pic" + str(stepNum), debugCroppedFrame)

    return targetX, targetY

def quick_path_search(frame, stepNum, leftX, leftY, rightX, rightY):
    '''
    This looks the point with the greatest y value from the left and the right cameras, then averages their positions to find a midpoint
    '''
    # the following if statement tells it to got straight if it can't find any points. probably not ideal.
    #if leftPoints[0] == [] and rightPoints[0] == []:
    #    return 0,0
    leftPoints = [(leftX[obj.LEFTTRACK] - car.MAX_X).tolist(), (leftY[obj.LEFTTRACK] - car.MAX_Y).tolist()]
    rightPoints = [(rightX[obj.RIGHTTRACK] - car.MAX_X).tolist(), (rightY[obj.RIGHTTRACK] - car.MAX_Y).tolist()]

    if leftPoints[0] == [] and rightPoints[0] == []: #if there is no data
        furthestPointX = 2
        furthestPointY = -10
    elif len(leftPoints[0]) < len(rightPoints[0]): #if there is more data on the right than on the left
        #use right data
        furthestPointX = min(rightPoints[0])
        furthestPointY = rightPoints[1][rightPoints[0].index(furthestPointX)]
        furthestPointX -= 40
    else:
        #use left data
        furthestPointX = max(leftPoints[0])
        furthestPointY = leftPoints[1][leftPoints[0].index(furthestPointX)]
        furthestPointX += 40

    # simulator adjustments (separate from the algorithm)
    normalise = np.sqrt(furthestPointX**2 + furthestPointY**2)
    if furthestPointY == 0:
        angle = np.rad2deg(np.arctan2(-0.01, furthestPointX))
    else:
        angle = np.rad2deg(np.arctan2(furthestPointY, furthestPointX))
    STEP_SIZE = 30
    if normalise == 0:
        normalisedX = 0
        normalisedY = -3
        if EACH_STEP:
            print("very little progress at StepNum: ", stepNum)
    elif angle > -45:
        normalisedX = STEP_SIZE * np.cos(np.deg2rad(-45))
        normalisedY = STEP_SIZE * np.sin(np.deg2rad(-45))
        if EACH_STEP:
            print("capped to -45 degrees turn at StepNum: ", stepNum)
    elif angle < -135:
        normalisedX = STEP_SIZE * np.cos(np.deg2rad(-135))
        normalisedY = STEP_SIZE * np.sin(np.deg2rad(-135))
        if EACH_STEP:
            print("capped to -135 degrees turn at StepNum: ", stepNum)
    else:
        normalisedX = round(furthestPointX / normalise * STEP_SIZE)
        normalisedY = round(furthestPointY / normalise * STEP_SIZE)


    # for debugging and graphical display
    if EACH_STEP:
        plt.suptitle("At stepNum: " + str(stepNum))
        #displays the rectangular coordinates
        plt.scatter(leftPoints[0], leftPoints[1], label = "leftTrack", color = "blue")
        plt.scatter(rightPoints[0], rightPoints[1], label = "rightTrack", color = "yellow")
        plt.scatter(normalisedX, normalisedY, label="target")
        plt.scatter(furthestPointX, furthestPointY, label="furthest (X, Y)")
        plt.xlabel('x (px)')
        plt.ylabel('y (px)')
        plt.axis('scaled')
        plt.legend()
        plt.show()
    return normalisedX, normalisedY 

########################################
"""running simAlgorithm runs the main(), which is different to the main() in mapSimulation.py
mapSimulation.py's main() is typically used and is the actual simulator
simAlgorithm.py's main() is used for debugging and code development.
"""
def main():
    print("please run the code in mapSimulation.py")

if __name__ == "__main__":
    main()