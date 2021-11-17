import numpy as np
import math
import cv2


### Camera coordinate to vehicle coordinate(ISO)
def RotX(Pitch):
      Pitch = np.deg2rad(Pitch)
      return [[1, 0, 0], [0, math.cos(Pitch), -math.sin(Pitch)], [0, math.sin(Pitch), math.cos(Pitch)]]

def RotY(Yaw):
      Yaw = np.deg2rad(Yaw)
      return [[math.cos(Yaw), 0, math.sin(Yaw)], [0, 1, 0], [-math.sin(Yaw), 0, math.cos(Yaw)]]

def RotZ(Roll):
      Roll = np.deg2rad(Roll)
      return [[math.cos(Roll), -math.sin(Roll), 0], [math.sin(Roll), math.cos(Roll), 0], [0, 0, 1]]      

### Camera parameter setting
ImageSize = (640, 480)
FocalLength = (672.5725981, 670.15018819)
PrinciplePoint = (299.72871914, 427.87676371)
# ImageSize = (720, 1280)
# FocalLength = (309.4362, 344.2161) #초점거리
# # FocalLength = (509.4362, 544.2161)
# PrinciplePoint = (618.9034, 477.5352) # 주점
IntrinsicMatrix = ((FocalLength[0], 0, 0), (0, FocalLength[1], 0), (PrinciplePoint[0], PrinciplePoint[1], 1))
print("IntrinsicMatrix : \n", IntrinsicMatrix)
'''  IntrinsicMatrix
      [ fx , cfx  , cx
        0  ,  fy  , cy
        0  ,  0   , 1 ]  
'''
Height = 2.798
Pitch = 2
Yaw = 1.4
Roll = 0

### Bird's eye view setting
DistAhead = 50 # 값이 오를 수록 멀리(?) 보임
# SpaceToOneSide = 6
SpaceToOneSide = 6 # 값이 오를수록 넓게 보임
BottomOffset = 4.0 # 이미지 아래 출발점?


OutView = (BottomOffset, DistAhead, -SpaceToOneSide, SpaceToOneSide)
# OutImageSize = [math.nan, 250]
OutImageSize = [math.nan, 250]
# print("OutImageSize : ", OutImageSize)

WorldHW = (abs(OutView[1]-OutView[0]), abs(OutView[3]-OutView[2])) # abs : 절대값 구하는 함수
# print("WorldHW : \n", WorldHW)

Scale = (OutImageSize[1]-1)/WorldHW[1]
ScaleXY = (Scale, Scale)
# print("ScaleXY : ", ScaleXY)

OutDimFrac = Scale*WorldHW[0]

# print("ODF : ", OutDimFrac)
OutDim = round(OutDimFrac)+1
# print("OD : ", OutDim)
OutImageSize[0] = OutDim

### Homography Matrix Compute

#Translation Vector
Rotation = np.matmul(np.matmul(RotZ(-Yaw),RotX(90-Pitch)),RotZ(Roll))
# print("Rotation : \n", Rotation)
TranslationinWorldUnits = (0, 0, Height)
Translation = [np.matmul(TranslationinWorldUnits, Rotation)]
print("Translation : \n", Translation)

#Rotation Matrix 회전변환행렬
RotationMatrix = np.matmul(RotY(180), np.matmul(RotZ(-90), np.matmul(RotZ(-Yaw), np.matmul(RotX(90-Pitch), RotZ(Roll)))))
print("RotationMatrix : \n", RotationMatrix)

#Camera Matrix (Extrinsic?)
CameraMatrix = np.matmul(np.r_[RotationMatrix, Translation], IntrinsicMatrix)
print("CameraMatrix : \n", CameraMatrix)
CameraMatrix2D = np.r_[[CameraMatrix[0]], [CameraMatrix[1]], [CameraMatrix[3]]]
# print("CameraMatrix2D : \n", CameraMatrix2D)

#Compute Vehicle-to-Image Projective Transform
VehicleHomo = np.linalg.inv(CameraMatrix2D)
# print("VH : \n", VehicleHomo)

AdjTransform = ((0, -1, 0), (-1, 0, 0), (0, 0, 1))
BevTransform = np.matmul(VehicleHomo, AdjTransform)
# print("BevTf \n",BevTransform)

DyDxVehicle = (OutView[3], OutView[1])
tXY = [a*b for a,b in zip(ScaleXY, DyDxVehicle)]

#test = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
ViewMatrix = ((ScaleXY[0], 0, 0), (0, ScaleXY[1], 0), (tXY[0]+1, tXY[1]+1, 1))
# print("ViewMatrix : \n", ViewMatrix)

T_Bev = np.matmul(BevTransform, ViewMatrix)
T_Bev = np.transpose(T_Bev)



### Main
src = cv2.imread("ttttt.jpg", cv2.IMREAD_COLOR)


BirdEyeView = cv2.warpPerspective(src, T_Bev, (OutImageSize[1], OutImageSize[0]))


### Distance

#Vehicle to Image
toOriginalImage = np.linalg.inv(np.transpose(T_Bev))
Trans = np.linalg.inv(np.matmul(toOriginalImage, VehicleHomo))
VehiclePoint = [[200, 0]]
XV = np.r_[VehiclePoint[0], np.shape(VehiclePoint)[0]]
UV = np.matmul(XV, Trans)

UV[0:2] = UV[0:2]/UV[2]
ImagePoints = list(map(int, UV[0:2]))

# print("ImagePoints : ",ImagePoints)
# annotatedBEV1 = cv2.drawMarker(BirdEyeView, ImagePoints, (0,0,255))
# cv2.putText(annotatedBEV1, "20 meters", ImagePoints, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0))

# cv2.imshow("Vehicle to Image", annotatedBEV1)

#Image to Vehicle
toOriginalImage = np.linalg.inv(np.transpose(T_Bev))
Trans = np.matmul(toOriginalImage, VehicleHomo)
ImagePoint = [[1250, 30]]

UI = ImagePoint
UI = np.r_[ImagePoint[0], np.shape(ImagePoint)[0]]
# print("UI : ", UI)
XI = np.matmul(UI, Trans)
# print("Trans : ", Trans)
# print("XI : ", XI)
XI[0:2] = XI[0:2]/XI[2]

XAhead = XI[0]

# annotatedBEV2 = cv2.drawMarker(BirdEyeView, ImagePoint[0], (0,0,255))
# cv2.putText(annotatedBEV2, str(round(XAhead, 2))+" meters", ImagePoint[0], cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0))
# print(str(round(XAhead,2)))
# cv2.imshow("Image to Vehicle", annotatedBEV2)
cv2.imshow("BEV", BirdEyeView)
cv2.imwrite('bev.png', BirdEyeView)


gray = cv2.cvtColor(BirdEyeView, cv2.COLOR_BGR2HSV)
cv2.imshow("gra",gray)
lower_white = np.array([0, 0, 110])
upper_white = np.array([205, 205, 205])
mask = cv2.inRange(gray, lower_white, upper_white)
cv2.imshow("mask", mask)
mmm = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
cv2.waitKey(0)
cv2.destroyAllWindows()