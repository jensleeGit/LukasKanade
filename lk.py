import numpy as np
import cv2
from scipy.ndimage import filters
from scipy import signal

# img1 = cv2.imread('')
# img2 = cv2.imread('')

# sh = [5,9]
# image2 = cv2.imread('\0')


# image2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)


image1 = cv2.imread('corners.jpg')
image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
image1 = cv2.resize(image1,(300,168))
sh = image1.shape
# sh = [300,400]
# image1 = np.zeros(sh)
image2 = cv2.imread('movement.jpg')
image2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
image2 = cv2.resize(image2,(300,168))
# sh = iamge1.shape
# image2 = np.zeros(sh)
p = np.zeros((sh[0],sh[1],6))
win_size = 3
# print sh
epsilon = 0.1
indices = np.indices(sh)
indices=np.transpose(indices,[1,2,0])
oness = np.ones(sh)

jacobian=np.zeros((sh[0],sh[1],2,6))
jacobian[:,:,0,0] = indices[:,:,0]
jacobian[:,:,1,1] = indices[:,:,0]
jacobian[:,:,0,2] = indices[:,:,1]
jacobian[:,:,1,3] = indices[:,:,1]
jacobian[:,:,0,4] = oness
jacobian[:,:,1,5] = oness
# print jacobian


itera=0

while (True):
    new_indices = np.zeros((sh[0],sh[1],2),dtype=np.int)
    new_indices[:,:,0] = (1 + p[:,:,0]) * indices[:,:,0] + p[:,:,2] * indices[:,:,1] + p[:,:,4]
    new_indices[:,:,1] = (p[:,:,1]) * indices[:,:,0] + (1+p[:,:,3]) * indices[:,:,1] + p[:,:,5]

    less_zero_x = new_indices[:,:,0] < 0
    greater_shape_x  =  new_indices[:,:,0] >= sh[0]
    less_zero_y = new_indices[:,:,1] < 0
    greater_shape_y  =  new_indices[:,:,1] >= sh[1]

    new_indices[:,:,0] = (1 - less_zero_x) * new_indices[:,:,0] + less_zero_x * 0
    new_indices[:,:,0] = (1 - greater_shape_x) * new_indices[:,:,0] + greater_shape_x * (sh[0] - 1)

    new_indices[:,:,1] = (1 - less_zero_y) * new_indices[:,:,1] + less_zero_y * 0
    new_indices[:,:,1] = (1 - greater_shape_y) * new_indices[:,:,1] + greater_shape_y * (sh[1] - 1)



    warped_image = np.zeros(sh)
    warped_image = image1[new_indices[:,:,0],new_indices[:,:,1]]

    name ='img_' + str(itera) + '.jpg'
    cv2.imwrite(name,warped_image)
    itera +=1
    print itera


    error_image = image2 - warped_image


    grad_x,grad_y = np.gradient(image1)
    warped_grad_x = np.zeros(sh)
    warped_grad_x = grad_x[new_indices[:,:,0],new_indices[:,:,1]]
    warped_grad_y = np.zeros(sh)
    warped_grad_y = grad_y[new_indices[:,:,0],new_indices[:,:,1]]



    grad_final_temp = [warped_grad_x[:,:],warped_grad_y[:,:]]
    grad_final = np.transpose(grad_final_temp,[1,2,0])
    grad_final = grad_final[:,:,np.newaxis,:]

    steep_desc = np.matmul(grad_final,jacobian)

    steep_desc_transpose = np.transpose(steep_desc,[0,1,3,2])

    hessian_elem = np.matmul(steep_desc_transpose,steep_desc)


    parameters = hessian_elem.shape[2]
    kernel = np.ones((win_size,win_size))
    hessian = np.zeros((sh[0],sh[1],parameters,parameters))
    for i in range(parameters):
        for j in range(parameters):
            hessian[:,:,i,j] = signal.convolve2d(hessian_elem[:,:,i,j],kernel,boundary = 'symm',mode='same')


    error_image = error_image[:,:,np.newaxis,np.newaxis]

    mat_elem = np.matmul(steep_desc_transpose,error_image)
    mat = np.zeros((sh[0],sh[1],mat_elem.shape[2],mat_elem.shape[3]))
    for i in range(mat_elem.shape[2]):
        for j in range(mat_elem.shape[3]):
            mat[:,:,i,j] = signal.convolve2d(mat_elem[:,:,i,j],kernel,boundary = 'symm',mode='same')


    hessian_inv = np.zeros((sh[0],sh[1],parameters,parameters))

    # print hessian[:,:].shape
    for i in range(sh[0]):
        for j in range(sh[1]):
            hessian_inv[i,j] = np.linalg.pinv(hessian[i,j])
    # hessian_inv = np.linalg.pinv(hessian)

    delta_p = np.matmul(hessian_inv,mat)
    delta_p = delta_p[:,:,:,0]

    normm = np.linalg.norm(delta_p,axis=2)
    mask = normm > epsilon
    # mask = np.tile(mask,(1,6))
    # print mask.shape
    mask = np.dstack((mask,mask,mask,mask,mask,mask))
    p = p + mask * delta_p
    br_cond = np.sum(mask)
    if br_cond==0:
        break
