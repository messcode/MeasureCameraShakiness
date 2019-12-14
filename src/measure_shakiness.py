import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
import collections 

Transformation = collections.namedtuple('Transformation', ['dx', 'dy', 'count', 'timeline', 'dist'])

def get_grey(frame):
    grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    return grey

def extract_frame(cap, every_n=5, f=lambda x: x):
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % every_n == 0:
            yield count, f(frame)
        count += 1

def measure_camera_movement(file_path, verbose=False, every_n=3, stop_frame=30):
    cap= cv.VideoCapture(file_path)
    fps = cap.get(cv.CAP_PROP_FPS)
    tfs = []
    for count, frame in extract_frame(cap, every_n=3, f=get_grey):
        if count == 0:
            prev_count, prev_grey = count, frame
            continue
        curr_count, curr_grey = count, frame
        orb = cv.ORB_create()
        orb = cv.ORB_create()
        #feature extraction sift is scale invariant
        kp1, des1 = orb.detectAndCompute(curr_grey, None)
        kp2, des2 = orb.detectAndCompute(prev_grey, None)
        # match
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key = lambda x:x.distance)
        # filter out the matches
        good = []
        for m in matches:
            for n in matches:
                if m.distance < 0.2 * n.distance:
                    good.append(m)
        if len(good) > 10:
            #  find homography
            pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, status = cv.findHomography(pts1, pts2)
            h, w = curr_grey.shape[:2]
            pts = np.float32([[0,0], [0,h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dx = M[0, 2]
            dy = M[1, 2]
            count = curr_count
            timeline = curr_count * 1.0 / fps
            dist = math.sqrt(dx**2 + dy**2)
            # track
            tfs.append(Transformation(dx, dy, curr_count, timeline, dist))
            # plot  figure
            if verbose:
                print("time={:.2f} frame={} good={}||dx={} dy={}".fromat(timeline, curr_count, len(good),dx, dy))
        else:
        # good points is not enough 
            pass
        # set prev
        prev_count, prev_grey = curr_count, curr_grey 
        if count > stop_frame:
            break
    return tfs    

if __name__ == "__main__":
    file_path = "../data/gleicher2.avi"
    cap= cv.VideoCapture(file_path)
    fps = cap.get(cv.CAP_PROP_FPS)
    tfs = []
    for count, frame in extract_frame(cap, every_n=3, f=get_grey):
        if count == 0:
            prev_count, prev_grey = count, frame
            continue
        curr_count, curr_grey = count, frame
        orb = cv.ORB_create()
        orb = cv.ORB_create()
        #feature extraction sift is scale invariant
        kp1, des1 = orb.detectAndCompute(curr_grey, None)
        kp2, des2 = orb.detectAndCompute(prev_grey, None)
        # match
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key = lambda x:x.distance)
        # filter out the matches
        good = []
        for m in matches:
            for n in matches:
                if m.distance < 0.2 * n.distance:
                    good.append(m)
        if len(good) > 10:
            #  find homography
            pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, status = cv.findHomography(pts1, pts2)
            h, w = curr_grey.shape[:2]
            pts = np.float32([[0,0], [0,h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dx = M[0, 2]
            dy = M[1, 2]
            count = curr_count
            timeline = curr_count * 1.0 / fps
            dist = math.sqrt(dx**2 + dy**2)
            # track
            tfs.append(Transformation(dx, dy, curr_count, timeline, dist))
            # plot  figure
            for idx, point in enumerate(pts1[:10].squeeze()):
                prev_grey_dots = cv.circle(prev_grey, tuple(point), 10, (0,255,0), 2)
                prev_grey_dots = cv.putText(prev_grey_dots, "{}".format(idx), tuple(point.astype(int)), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            for idx, point in enumerate(pts2[:10].squeeze()):
                cur_grey_dots = cv.circle(curr_grey, tuple(point), 10, (0, 255, 0), 2)
                cur_grey_dots = cv.putText(cur_grey_dots, "{}".format(idx), tuple(point), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.imshow(prev_grey_dots)
            ax1.set_title("Time={:.2f} dx={} dy={}".format(timeline, dx, dy))
            ax2.imshow(cur_grey_dots)
            plt.show()
            plt.waitforbuttonpress()
            plt.close(1) 
        else:
        # good points is not enough 
            pass
        # set prev
        prev_count, prev_grey = curr_count, curr_grey 
        if count > 30:
            break
    # count_v = [t.count for t in tfs]
    # dx_v = [t.dx for t in tfs]
    # dy_v = [t.dy for t in tfs]
    # plt.plot(count_v, dx_v)
    # plt.plot(count_v, dy_v)
    # plt.show() 