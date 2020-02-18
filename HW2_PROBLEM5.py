import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

#function to compute homogrphy
def computeH(img1_pts,img2_pts):
    maxInliers1 = []
    finalH1 = None
    #run 500 iterations of ransac
    for i in range(500):
        #select 4 points in random form the feature matches
        corr1 = img1_pts[random.randrange(0, len(img1_pts))]
        corr2 = img1_pts[random.randrange(0, len(img1_pts))]
        randomFour = np.vstack((corr1, corr2))
        corr3 = img1_pts[random.randrange(0, len(img1_pts))]
        randomFour = np.vstack((randomFour, corr3))
        corr4 = img1_pts[random.randrange(0, len(img1_pts))]
        randomFour = np.vstack((randomFour, corr4))

        rows = []
        #form homography matrix for each set of four points
        for corr in randomFour:
            p1 = np.matrix([corr.item(0), corr.item(1), 1])
            p2 = np.matrix([corr.item(2), corr.item(3), 1])

            r2 = [0, 0, 0, -p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2),
                      p2.item(1) * p1.item(0), p2.item(1) * p1.item(1), p2.item(1) * p1.item(2)]
            r1 = [-p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2), 0, 0, 0,
                      p2.item(0) * p1.item(0), p2.item(0) * p1.item(1), p2.item(0) * p1.item(2)]
            rows.append(r1)
            rows.append(r2)

        matrixA = np.matrix(rows)

        ##perform svd to obtain the final homography matrix
        u, s, v = np.linalg.svd(matrixA)
        h = np.reshape(v[8], (3, 3))
        h = (1 / h.item(8)) * h
        #find inlines based on thier distance
        inliers1 = []
        for i in range(len(img1_pts)):
            d = Distance(img1_pts[i], h)
            if d < 5:
                inliers1.append(img1_pts[i])

        if len(inliers1) > len(maxInliers1):
            maxInliers1 = inliers1
            finalH1 = h

    maxInliers2 = []
    finalH2 = None
    #repeat the above steps for the second set of images
    for i in range(500):
        corr1 = img2_pts[random.randrange(0, len(img2_pts))]
        corr2 = img2_pts[random.randrange(0, len(img2_pts))]
        randomFour = np.vstack((corr1, corr2))
        corr3 = img2_pts[random.randrange(0, len(img2_pts))]
        randomFour = np.vstack((randomFour, corr3))
        corr4 = img2_pts[random.randrange(0, len(img2_pts))]
        randomFour = np.vstack((randomFour, corr4))

        rows = []
        for corr in randomFour:
            p1 = np.matrix([corr.item(0), corr.item(1), 1])
            p2 = np.matrix([corr.item(2), corr.item(3), 1])

            r2 = [0, 0, 0, -p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2),
                  p2.item(1) * p1.item(0), p2.item(1) * p1.item(1), p2.item(1) * p1.item(2)]
            r1 = [-p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2), 0, 0, 0,
                  p2.item(0) * p1.item(0), p2.item(0) * p1.item(1), p2.item(0) * p1.item(2)]
            rows.append(r1)
            rows.append(r2)

        matrixA = np.matrix(rows)

        # perform svd to obtain the final homography matrix
        u, s, v = np.linalg.svd(matrixA)
        h2 = np.reshape(v[8], (3, 3))
        h2 = (1 / h.item(8)) * h2

        inliers2 = []
        for i in range(len(img2_pts)):
            d = Distance(img2_pts[i], h2)
            if d < 5:
                inliers2.append(img2_pts[i])

        if len(inliers2) > len(maxInliers2):
            maxInliers2 = inliers2
            finalH2 = h2

    return finalH1,finalH2


def Distance(correspondence, h):

    point1 = np.transpose(np.matrix([correspondence[0].item(0), correspondence[0].item(1), 1]))
    estimate2 = np.dot(h, point1)
    estimate2 = (1/estimate2.item(2))*estimate2

    point2 = np.transpose(np.matrix([correspondence[0].item(2), correspondence[0].item(3), 1]))
    err = point2 - estimate2
    return np.linalg.norm(err)

def main():
    a=cv2.imread(r'C:\Users\Chinnu\Desktop\asu\classes\ciupr\keble_a.jpg')
    b=cv2.imread(r'C:\Users\Chinnu\Desktop\asu\classes\ciupr\keble_b.jpg')
    c=cv2.imread(r'C:\Users\Chinnu\Desktop\asu\classes\ciupr\keble_c.jpg')

    orb= cv2.ORB_create()

    # find the keypoints with ORB
    kp1, des1 = orb.detectAndCompute(a,None)
    kp2, des2 = orb.detectAndCompute(b,None)
    kp3, des3 = orb.detectAndCompute(c,None)

    # create BFMatcher object
    bf = cv2.BFMatcher()

    matches1 = bf.knnMatch(des2,des1,k=2)
    matches2 = bf.knnMatch(des2,des3,k=2)

    # store all the good matches as per Lowe's ratio test.
    good1 = []
    good1_org = []
    for m, n in matches1:
        if m.distance < 0.75 * n.distance:
            good1.append([m])
            good1_org.append(m)

    # store all the good matches as per Lowe's ratio test.
    good2 = []
    good2_org = []
    for m, n in matches2:
        if m.distance < 0.75 * n.distance:
            good2.append([m])
            good2_org.append(m)

    # plotting the best matches using Lowe's ratio
    img3 = cv2.drawMatchesKnn(b, kp2, a, kp1, good1, None, flags=2)
    img4 = cv2.drawMatchesKnn(b, kp2, c, kp3, good2, None, flags=2)
    img3=cv2.cvtColor(img3,cv2.COLOR_BGR2RGB)
    img4=cv2.cvtColor(img4,cv2.COLOR_BGR2RGB)
    plt.imshow(img3),plt.show()
    plt.imshow(img4),plt.show()

    keypoints1 = [kp2, kp1]
    correspondenceList1 = []
    for match in good1_org:
        (x1, y1) = keypoints1[0][match.queryIdx].pt
        (x2, y2) = keypoints1[1][match.trainIdx].pt
        correspondenceList1.append([x1, y1, x2, y2])

    corrs1 = np.matrix(correspondenceList1)

    keypoints2 = [kp2, kp3]
    correspondenceList2 = []
    for match in good2_org:
        (x1, y1) = keypoints2[0][match.queryIdx].pt
        (x2, y2) = keypoints2[1][match.trainIdx].pt
        correspondenceList2.append([x1, y1, x2, y2])

    corrs2 = np.matrix(correspondenceList2)

    H1,H2=computeH(corrs1,corrs2)

    warped1 = cv2.warpPerspective(a, H1, (a.shape[1]+a.shape[1]+a.shape[1],a.shape[0]), flags=cv2.INTER_LINEAR)
    warped1=cv2.cvtColor(warped1,cv2.COLOR_BGR2RGB)

    warped2 = cv2.warpPerspective(c, H2,(c.shape[1]+c.shape[1],c.shape[0]), flags=cv2.INTER_LINEAR)
    warped2=cv2.cvtColor(warped2,cv2.COLOR_BGR2RGB)

    plt.figure()
    plt.imshow(warped1)
    plt.show()

    plt.figure()
    plt.imshow(warped2)
    plt.show()
    b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
    warped1[0:b.shape[0], 583:1303] =b
    warped1[0:b.shape[0], 1203:1603] = warped2[0:warped2.shape[0],0:400]

    plt.figure()
    plt.imshow(warped1)
    plt.show()

if __name__ == "__main__":
    main()