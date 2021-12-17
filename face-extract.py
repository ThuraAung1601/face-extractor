import argparse
import cv2
import numpy as np
import os
import timeit

# import prebuilt libraries with pre-trained models
from mtcnn.mtcnn import MTCNN
import cvlib
import face_detection

def mtcnn_detector(inputFile,outputPath):
  mtcnn_detector = MTCNN()
  cap = cv2.VideoCapture(inputFile)
  print("... Scanning faces ...")
  count = 0
  start = timeit.default_timer()
  while cap.isOpened():
          ret,frame = cap.read()
          if ret:
              faces = mtcnn_detector.detect_faces(frame)
              for result in faces:
                  startX, startY, fW, fH = result['box']
                  endX, endY = startX + fW, startY + h
                  face_crop = frame[startY:endY, startX:endX]
                  count = count + 1
                  outputFile = "{}.png".format(count)
                  outputPath = os.path.join(outputPath, outputFile)
                  cv2.imwrite(outputPath, face_crop)
                  print("Confidence score for {}.png : {} ".format(count,result['confidence']))
          else:
              break          
  cap.release()
  cv2.destroyAllWindows()
  print("... Done ... ")
  stop = timeit.default_timer()
  print('Time taken : ', stop - start, " seconds")  
  print("{} faces extracted using MTCNN".format(count))
  
def dlib_detector(inputFile,outputPath):
  cap = cv2.VideoCapture(inputFile)
  print("... Scanning faces ...")
  count = 0
  start = timeit.default_timer()
  while cap.isOpened():
          ret,frame = cap.read()
          if ret:
              faces, confidences = cvlib.detect_face(frame) 
              for face in faces:
                  (startX, startY, endX, endY) = face
                  face_crop = frame[startY:endY, startX:endX]
                  count = count + 1
                  outputFile = "{}.png".format(count)
                  outputPath = os.path.join(outputPath, outputFile)
                  cv2.imwrite(outputPath, face_crop)
                  print("Confidence score for {}.png : {} ".format(count,confidences))
          else:
              break

  cap.release()
  cv2.destroyAllWindows()
  print("... Done ... ")
  stop = timeit.default_timer()
  print('Time taken : ', stop - start, " seconds")  
  print("{} faces extracted using DLIB".format(count))

def dsfd_detector(inputFile,outputPath):
  dsfd_detector = face_detection.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)
  cap = cv2.VideoCapture(inputFile)
  print("... Scanning faces ...")
  count = 0
  start = timeit.default_timer()
  while cap.isOpened():
          ret,frame = cap.read()
          if ret:          
              # BGR to RGB
              frame = frame[:, :, ::-1]
              # DSFD detector
              faces = dsfd_detector.detect(frame)[:, :]
              for result in faces:
                  startX, startY, endX, endY,confidences = result
                  startX, startY, endX, endY = int(startX), int(startY), int(endX), int(endY)
                  face_crop = frame[startY:endY, startX:endX]
                  count = count + 1
                  outputFile = "{}.png".format(count)
                  outputPath = os.path.join(outputPath, outputFile)
                  cv2.imwrite(outputPath, face_crop)
                  print("Confidence score for {}.png: {} ".format(count,confidences))
          else:
              break

  cap.release()
  cv2.destroyAllWindows()
  print("... Done ... ")
  stop = timeit.default_timer()
  print('Time taken : ', stop - start, " seconds")  
  print("{} faces extracted using dsfd".format(count))
  
def main(inputFile,outputPath,modelUsed):
  if modelUsed = 'mtcnn': 
    mtcnn_detector(inputFile,outputPath)
  elif modelUsed = 'dlib':
    dlib_detector(inputFile,outputPath)
  elif modelUsed = 'dsfd':
    dsfd_detector(inputFile,outputPath)
  else:
    print("Detector choosen is not available")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  
  # options
  parser.add_argument("-i", "--input", required=True, help="path to input video")
  parser.add_argument("-o", "--output", default="output", help="path to output directory of faces")
  parser.add_argument("-m", "--model", default="dlib", help="model used to extract faces")
  
  inputFile = args['input']
  outputPath = args['output']
  modelUsed = args['model']
  
  cwd = os.getcwd()
  outputPath = os.path.join(cwd, outputPath)
  if not os.path.exists(outputPath):
    os.makedirs(outputPath)
    
  main(inputFile,outputPath,modelUsed)
