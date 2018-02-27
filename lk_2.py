import numpy as np
import cv2
from scipy.ndimage import filters
from scipy import signal

feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )


image1 = cv2.imread('i1.jpg')
image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
image1 = cv2.resize(image1,(600,900))


sh = image1.shape
image2 = cv2.imread('i2.jpg')
image2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
image2 = cv2.resize(image2,(600,900))

p0 = cv2.goodFeaturesToTrack(image1, mask = None, **feature_params)




epsilon = 0.1

indices = np.indices(sh)
indices=np.transpose(indices,[1,2,0])
oness = np.ones(sh)

jacobian=np.zeros((sh[0],sh[1],2,6))
jacobian[:,:,0,0] = indices[:,:,1]
jacobian[:,:,1,1] = indices[:,:,1]
jacobian[:,:,0,2] = indices[:,:,0]
jacobian[:,:,1,3] = indices[:,:,0]
jacobian[:,:,0,4] = oness
jacobian[:,:,1,5] = oness
# print jacobian

win_size = 150

itera=0
sample_p = np.float32([[20,30],[30,20],[50,40]])

p = np.zeros((6,1))
p[0] = 1
p[3] = 1
# p[0] = np.random.randint(1)
# p[1] = np.random.randint(6)
# p[2] = np.random.randint(7)
# p[3] = np.random.randint(8)
# p[4] = np.random.randint(9)
# p[5] = np.random.randint(5)

def warp(x,y):
    wx = p[0]*x + p[2]*y + p[4]
    wy = p[1]*x + p[3]*y + p[5]
    return [wx,wy]

centre_x = int(p0[9][0][0])
centre_y = int(p0[9][0][1])
top_x = max(centre_x - win_size,0)
top_y = max(centre_y - win_size,0)
bottom_x = min(centre_x + win_size,sh[1])
bottom_y = min(centre_y + win_size,sh[0])
jacobian = jacobian[top_y:bottom_y,top_x:bottom_x,:,:]

inter= 0
while True:

    transformed_points = np.float32([warp(sample_p[0][0],sample_p[0][1]),warp(sample_p[1][0],sample_p[1][1]),warp(sample_p[2][0],sample_p[2][1])])
    M = cv2.getAffineTransform(sample_p,transformed_points)
    warped_image = cv2.warpAffine(image2,M,(sh[1],sh[0]))
    na = str(inter) + 'ty.jpg'
    cv2.imwrite(na,warped_image)
    inter+=1




        # name ='img_' + str(itera) + '.jpg'
        # cv2.imwrite(name,warped_image)
        # itera +=1
        # print itera


    error_image = image1 - warped_image
    error_window = error_image[top_y:bottom_y,top_x:bottom_x]



    grad_x,grad_y = np.gradient(image2)
    warped_grad_x = cv2.warpAffine(grad_x,M,(sh[1],sh[0]))
    warped_grad_y = cv2.warpAffine(grad_y,M,(sh[1],sh[0]))
    grad = np.stack((warped_grad_x,warped_grad_y),axis=2)
    grad_window = grad[top_y:bottom_y,top_x:bottom_x,:]
    grad_window = grad_window[:,:,np.newaxis,:]

    steep_desc = np.matmul(grad_window,jacobian)
    steep_desc_transpose = np.transpose(steep_desc,[0,1,3,2])
    hessian_elem = np.matmul(steep_desc_transpose,steep_desc)
    hessian = np.sum(np.sum(hessian_elem,axis=0),axis=0)
    error_window = error_window[:,:,np.newaxis,np.newaxis]
    mat_elem = np.matmul(steep_desc_transpose,error_window)
    mat = np.sum(np.sum(mat_elem,axis=0),axis=0)
    hessian_inv = np.linalg.pinv(hessian)
    delta_p = np.matmul(hessian_inv,mat)
    p = p + delta_p
    norm_delp = np.linalg.norm(delta_p,axis=0)
    print norm_delp
    if norm_delp < epsilon:
        break
