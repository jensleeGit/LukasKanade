import numpy as np
import cv2
from scipy.ndimage import filters
from scipy import signal
from moviepy.editor import VideoFileClip
import sys

videopath = sys.argv[1]
cap = cv2.VideoCapture(videopath)

length = 300
breadth = 200
ret,image2 = cap.read()
# image2 = cv2.imread('template.JPG')
image2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
image2 = image2.astype('float32')
sh = image2.shape
print sh
image2 = cv2.resize(image2,(length,breadth))
sh = image2.shape
# print sh
# def warp(p,x,y):
#     wx = p[0]*x + p[2]*y + p[4]
#     wy = p[1]*x + p[3]*y + p[5]
#     return [wx,wy]


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

grad_x = cv2.Sobel(image2, cv2.CV_32F, 1, 0)
grad_y = cv2.Sobel(image2, cv2.CV_32F, 0, 1)
# grad_x,grad_y = np.gradient(image2)
grad = np.stack((grad_x,grad_y),axis=2)
grad = grad[:,:,np.newaxis,:]
steep_desc = np.matmul(grad,jacobian)

steep_desc_transpose = np.transpose(steep_desc,[0,1,3,2])
hessian = np.matmul(steep_desc_transpose,steep_desc)
hessian = np.sum(np.sum(hessian,axis=0),axis=0)
hessian_inv = np.linalg.inv(hessian)
# print hessian_inv
# exit(0)


# sample_p = np.float32([[50,50],[50,200],[200,50]])


epsilon = 0.005
iterr_limit = 2500


def warp_image(image3):
    image1 = np.copy(image3)
    image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    image1 = image1.astype('float32')
    image1 = cv2.resize(image1,(length,breadth))
    p = np.zeros((6,1),dtype = np.float)
    p[0] = 1
    p[3] = 1
    iterr = 0
    while True:
        warped_image = cv2.warpAffine(image1, np.transpose(np.reshape(p,[3,2])), (sh[1],sh[0]))
        # transformed_points = np.float32([warp(sample_p[0][0],sample_p[0][1]),warp(sample_p[1][0],sample_p[1][1]),warp(sample_p[2][0],sample_p[2][1])])
        # M = cv2.getAffineTransform(sample_p,transformed_points)
        # warped_image = cv2.warpAffine(image1,M,(sh[1],sh[0]))

        error_image = warped_image - image2
        error_image = error_image[:,:,np.newaxis,np.newaxis]
        aux_val_elem = np.matmul(steep_desc_transpose,error_image)
        aux_val = np.sum(np.sum(aux_val_elem,axis=0),axis=0)
        del_p = np.matmul(hessian_inv,aux_val)
        iterr+=1
        first_part = p + del_p
        sec_part = np.zeros((6,1))
        sec_part[0]  = p[0] * del_p[0]
        sec_part[1]  = p[1] * del_p[0]
        sec_part[2]  = p[0] * del_p[2]
        sec_part[3]  = p[1] * del_p[2]
        sec_part[4]  = p[0] * del_p[4]
        sec_part[5]  = p[1] * del_p[4]
        third_part = np.zeros((6,1))
        third_part[0] = p[2] * del_p[1]
        third_part[1] = p[3] * del_p[1]
        third_part[2] = p[2] * del_p[3]
        third_part[3] = p[3] * del_p[3]
        third_part[4] = p[2] * del_p[5]
        third_part[5] = p[3] * del_p[5]
        p = first_part + sec_part + third_part
        norm_delp = np.linalg.norm(del_p,axis=0)
        # print norm_delp
        if norm_delp < epsilon:
            # print "done"
            warped_image = cv2.resize(warped_image,(1920,1080))
            # print warped_image.shape
            return warped_image
        if iterr>iterr_limit:
            warped_image = cv2.resize(warped_image,(1920,1080))
            return warped_image



vid_output='stablized.mp4'

clip1=VideoFileClip(videopath)

vid_clip=clip1.fl_image(warp_image)
vid_clip.write_videofile(vid_output, audio=False)
