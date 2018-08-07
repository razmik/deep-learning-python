"""
Tute: https://www.learnopencv.com/deep-learning-based-text-recognition-ocr-using-tesseract-and-opencv/
Code: https://github.com/spmallick/learnopencv/blob/master/OCR/ocr_simple.py
Pathset: https://stackoverflow.com/questions/43041994/how-to-install-pytesseract-in-windows-8-1win64-visual-studio-2012pythonanac
Error of loading testdata path: https://github.com/madmaze/pytesseract/issues/50
"""
import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"

if __name__ == '__main__':

    # Read image path from command line
    # imPath = "data/computer-vision-1024x454.jpg"
    imPath = "data/receipt.png"

    # Uncomment the line below to provide path to tesseract manually
    # pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

    # Define config parameters.
    # '-l eng'  for using the English language
    # '--oem 1' sets the OCR Engine Mode to LSTM only.
    #
    #  There are four OCR Engine Mode (oem) available
    #  0    Legacy engine only.
    #  1    Neural nets LSTM engine only.
    #  2    Legacy + LSTM engines.
    #  3    Default, based on what is available.
    #
    #  '--psm 3' sets the Page Segmentation Mode (psm) to auto.
    #  Other important psm modes will be discussed in a future post.

    config = ('-l eng --oem 1 --psm 3 --tessdata-dir "C:/Program Files (x86)/Tesseract-OCR/tessdata"')

    # Read image from disk
    im = cv2.imread(imPath, cv2.IMREAD_COLOR)

    # Run tesseract OCR on image
    text = pytesseract.image_to_string(im, config=config)

    # Print recognized text
    print(text)