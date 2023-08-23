"""IMPORTANT INFORMATION
In the ./documentation folder, there is a manual for this simulator (mapSimulation manual.pdf). 
You may find it helpful.

To run this code correctly, mapSimulation.py must be in the current directory, since I (Jack) used
a relative file path for cv.imread() and included libraries as packages

The input images are stored in the simulation_img. These images made using MS paint, look at
the examples in the folder.
"""

import cv2 as cv

# from cv2 import aruco
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy import stats
import time

import simAlgorithm

EACH_STEP = False # set to true for debugging, it shows images from each cycle/step of simulation

GREEN = (0, 255, 0)  # in BGR
BLUE = (255, 0, 0)
BLACK = (0, 0, 0)
WINDOW_HEIGHT = 240
WINDOW_WIDTH = 200
# length of window. Window is the square section of the map that the car sees, 
# since it cannot see the entire map at once.
WINDOW_DIAGONAL = (int)(np.sqrt(WINDOW_HEIGHT**2 + WINDOW_WIDTH**2) + 1)


def crop_image(frame, newDimension, origin=None):
    """crops an image

        Args:
        frame (np.array): typically the return value of cv.imread()
        newDimension (tuple): (height, width) coordinates
        origin (tuple, optional): centre of the cropped image as (x,y) coordinates

     Returns:
        np.array: cropped image
    """
    height, width = newDimension
    halfHeight, halfWidth = (height / 2, width / 2)
    if origin == None:
        oldHeight, oldWidth = frame.shape[:2]
        (x, y) = (oldWidth / 2, oldHeight / 2)
    else:
        (x, y) = origin
    return frame[int(y - halfHeight) : int(y + halfHeight), int(x - halfWidth) : int(x + halfWidth)]

def rotate_image(frame, angle):
    # rotate image
    height, width = frame.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv.getRotationMatrix2D(center, angle, 1)
    rotatedImage = cv.warpAffine(frame, matrix, (width, height))
    return crop_image(rotatedImage, (WINDOW_HEIGHT, WINDOW_WIDTH))  # crop it to a square

def homogenous_frame_transform(inputX, inputY, dx, dy, theta):
    #you can Google homogenous_frame_transform, if you don't know what it is.
    #it is used to change from one coordinate system to another
    
    rad = np.deg2rad(theta)
    mapToCarM = np.array([[np.cos(rad), -np.sin(rad), dx],
                        [np.sin(rad), np.cos(rad), dy],
                        [0, 0, 1]])
    carCoordinate = np.zeros((3, 1))
    P = np.array([[inputX,], [inputY,], [1,]])
    carCoordinate = np.matmul(mapToCarM, P)
    return float(carCoordinate[0]), float(carCoordinate[1])

def car_to_map_frame(carX, carY, dx, dy, theta):
    # changes from the car's to map's coordinate frames
    # rotate
    rotX, rotY = homogenous_frame_transform(carX, carY, 0, 0, theta)
    # translate
    finalX, finalY = homogenous_frame_transform(rotX, rotY, dx, dy, 0)
    return finalX, finalY

def set_bounded_value(value, aMin, aMax):
    if value > aMax:
        return aMax
    if value < aMin:
        return aMin
    else:
        return value

class MapSimulation:
    def __init__(self, imageName, x:int=None, y:int=None, heading:int=None):
        (self.x, self.y, self.heading) = (x, y, heading)
        self.check_init_conditions(imageName)
        self.frame = cv.imread("./simulation_image_input/" + imageName)
        self.displayFrame = self.frame.copy() #frame is used for CV analysis. 
        # displayFrame is used for display, and is annotated on. The visual annotation interferes with analysis.
        # Thus, we have separate frames for display and analysis
        cv.circle(self.displayFrame, (self.x, self.y), 7, BLACK, -1)

    def isInteger(self, string):
        if string[0] == '-': #if a negative number
            return string[1:].isdigit()
        else:
            return string.isdigit()

    def check_init_conditions(self, imageName):
        splited = imageName.split(".") #removes the .jpg
        splited = splited[0].split("_")
        if len(splited) >= 4:
            if self.isInteger(splited[-3]) and self.x == None:
                self.x = int(splited[-3])
            if self.isInteger(splited[-2]) and self.y == None:
                self.y = int(splited[-2])
            if self.isInteger(splited[-1]) and self.heading == None:
                self.heading = int(splited[-1])
        else:
            if type(self.x) == None or self.y == None or self.heading == None:
                print("ERROR: Image name: ", imageName, " is not valid, it needs to be include \
                    the initial conditions in the form: <filename>_<x>_<y>_<heading>.jpg")

    def next_point(self, stepNum):
        x = self.x
        y = self.y

        # crop image
        cropped = crop_image(self.frame, (WINDOW_DIAGONAL, WINDOW_DIAGONAL), origin=(x, y))
        # rotate image
        rotated = rotate_image(cropped, 90 + self.heading)

        carX, carY = simAlgorithm.SIM_get_target_position(rotated, stepNum)
        newX, newY = car_to_map_frame(carX, carY, x, y, 90 + self.heading)

        self.heading = float(np.rad2deg(np.arctan2(newY - self.y, newX - self.x)))
        newX, newY = self.check_bounds(newX, newY) #to make sure the car does not go off the map

        #displays the new position of the car
        cv.line(self.displayFrame, (round(x), round(y)), (round(newX), round(newY)), GREEN, 3)
        if stepNum % 10 == 0:
            circleColour = BLUE
            org = round(newX), round(newY + 3)
            cv.putText(self.displayFrame, str(stepNum), org, cv.FONT_HERSHEY_SIMPLEX, 1, BLUE, 2, cv.LINE_AA)
        else:
            circleColour = GREEN
        cv.circle(self.displayFrame, (round(newX), round(newY)), 7, circleColour, -1)

        (self.x, self.y) = (newX, newY)
        if EACH_STEP:
            print("StepNum: ", stepNum, " carX, Y: ",  carX, carY)
            print("heading: ", round(self.heading), "(x, y): ", self.x, self.y)
    
    def check_bounds(self, x, y):
        """It checks if the point is out of bounds. If yes, it corrects for it.
        There needs to be sufficient space for the WINDOW_DIAGONAL
        The black lines on the ./simulation_image_input/yellowMapInit.jpg marks the bounds
        """
        height, width = self.frame.shape[:2]
        boundedY = set_bounded_value(y, WINDOW_DIAGONAL / 2, height - WINDOW_DIAGONAL / 2)
        boundedX = set_bounded_value(x, WINDOW_DIAGONAL / 2, width - WINDOW_DIAGONAL / 2)
        if (boundedX != x or boundedY != y):
            print("going out of bounds!!")
        return float(boundedX), float(boundedY)
        
    def show_result(self):
        resized = cv.resize(self.displayFrame, (1200, 600)) 
        cv.imshow("displayFrame", resized)


def main():
    global EACH_STEP
    simAlgorithm.set_EACH_STEP(EACH_STEP)
    imageName = "fullScaleGlare_215_351_0.jpg"
    # sim = MapSimulation(imageName, x=1999, y=275, heading=45)
    # sim = MapSimulation(imageName, x=215, y=370, heading=20)
    sim = MapSimulation(imageName)

    prevTime = time.time() #for measuring processing time of the algorithm. 
    # cv.imshow() and plt.show() add lag
    savedEACH_STEP = EACH_STEP
    for stepNum in range(5 if savedEACH_STEP else 270):
        if stepNum == -1: #switch to bebug mode. -1 to turn this off
            EACH_STEP = True
            simAlgorithm.set_EACH_STEP(True)
            prevTime = time.time()
        sim.next_point(stepNum)
        if EACH_STEP:
            sim.show_result()
            print("time to complete cycle: ", time.time() - prevTime)
            prevTime = time.time()

    sim.show_result()
    # cv.imshow("original frame", sim.frame)
    c = cv.waitKey(0)

if __name__ == "__main__":
    main()
