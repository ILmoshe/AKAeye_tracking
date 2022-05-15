"""
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^DESCRIPTION^^^^^^^^^^^^^^^^^^^^^^
Pupil-detection

AKA advanced technologies LTD

author: Moshe Mizrachi
last edited date: 14/05/22

status: in progress

-------------------------------SETUP-----------------------
python ver: 3.6 (not upper!)
NOTE: The install should be made from ANACONDA env.
libraries: install via command line (don't forget to activate venv!)
1.  conda install -c menpo opencv==3.4.2







FOR DEVELOPER:


---------------------------------------FOR INVERSE
let src = cv.imread('canvasInput');
let dst = cv.Mat.zeros(src.rows, src.cols, cv.CV_8U);
let circles = new cv.Mat();
let color = new cv.Scalar(255, 0, 0);
cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
// You can try more different parameters
TODO :cv.HoughCircles(src, circles, cv.HOUGH_GRADIENT,
                1, 9, 20, 40, 40, 0);
// draw circles
for (let i = 0; i < circles.cols; ++i) {
    let x = circles.data32F[i * 3];
    let y = circles.data32F[i * 3 + 1];
    let radius = circles.data32F[i * 3 + 2];
    let center = new cv.Point(x, y);
    cv.circle(dst, center, radius, color);
}
cv.imshow('canvasOutput', dst);
src.delete(); dst.delete(); circles.delete();

-----------------------FOR NORMAL

let src = cv.imread('canvasInput');
let dst = cv.Mat.zeros(src.rows, src.cols, cv.CV_8U);
let circles = new cv.Mat();
let color = new cv.Scalar(255, 0, 0);
cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);
// You can try more different parameters
TODO: cv.HoughCircles(src, circles, cv.HOUGH_GRADIENT,
                1, 40, 40, 50, 34, 0);
// draw circles


"""

import cv2
import numpy as np

cap = cv2.VideoCapture(0)  # TODO: decrease image size
# out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (500, 450))
CON = 80
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        roi = frame[0:200, 100:400]
        rows, cols, _ = roi.shape
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_roi = cv2.GaussianBlur(gray_roi, (11, 11), 0)
        gray_roi = cv2.medianBlur(gray_roi, 5)

        flipped = np.array([[((cell - 255) ** 2) / 255 for cell in row] for row in gray_roi])
        threshold = cv2.threshold(flipped, 115, 255, cv2.THRESH_BINARY_INV)[1]
        contours = cv2.findContours(np.uint8(threshold), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
        # circles = cv2.HoughCircles(np.uint8(gray_roi), cv2.HOUGH_GRADIENT, 1, 100,
        #                            param1=100, param2=10, minRadius=20, maxRadius=60)
        #
        # circles = cv2.HoughCircles(np.uint8(gray_roi), cv2.HOUGH_GRADIENT, 1, 100,
        #                            param1=100, param2=10, minRadius=20, maxRadius=60)

        #  circles = cv2.HoughCircles(np.uint8(gray_roi), cv2.HOUGH_GRADIENT,
        #                             1, 40, param1=40, param2=50, minRadius=34, maxRadius=0)
        #
        # # TODO: cv.HoughCircles(src, circles, cv.HOUGH_GRADIENT,
        #                  #      1, 9, 20, 40, 40, 0);

        circles = cv2.HoughCircles(np.uint8(flipped), cv2.HOUGH_GRADIENT,
                                   1, 11, param1=16, param2=35, minRadius=20, maxRadius=60)

        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # draw the outer circle
                cv2.circle(roi, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # draw the center of the circle
                cv2.circle(roi, (i[0], i[1]), 2, (0, 0, 255), 3)

        # TODO
        #  import numpy as np
        # from scipy.optimize import curve_fit
        #
        # x = np.array([1, 2, 3, 9])
        # y = np.array([1, 4, 1, 3])
        #
        # def fit_func(x, a, b,c):
        #     return a*(x^2) + b*x + c
        #
        # params = curve_fit(fit_func, x, y)
        #
        # [a, b, c] = params[0]
        # Max_point = -b/(2*a)
        # X1X2_dist = ((b^2 -4*a*c)^0.5)/a
        sum_row = np.sum(flipped, axis=0)
        points = [[index, cell / 200] for index, cell in enumerate(sum_row)]
        points = np.array(points)

        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            # cv2.circle(roi, (x + int(w / 2), y + int(h / 2)), int((h) / 3), (0, 0, 255), 2)
            cv2.line(roi, (x + int(w / 2), 0), (x + int(w / 2), rows), (50, 200, 0), 1)
            cv2.line(roi, (0, y + int(h / 2)), (cols, y + int(h / 2)), (50, 200, 0), 1)

            cv2.polylines(roi, np.int32([points]), isClosed=False, color=(128, 0, 200))

            cv2.putText(roi, text='[Press Q to Exit]', org=(int(cols - 180), int(rows - 15)),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.6, color=(0, 0, 0))
            cv2.putText(threshold, text='[Press Q to Exit]', org=(int(cols - 180), int(rows - 15)),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.6, color=(255, 255, 255))
            cv2.putText(gray_roi, text='[Press Q to Exit]', org=(int(cols - 180), int(rows - 15)),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.6, color=(0, 0, 0))
            break

        cv2.imshow("roi", roi)
        cv2.imshow('Threshold', threshold)
        # cv2.imshow('gray_roi', gray_roi)
        # out.write(roi)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if cv2.waitKey(15) & 0xFF == ord('q'):  # Press 'Q' on the keyboard to exit the playback
        break

cap.release()
cv2.destroyAllWindows()
