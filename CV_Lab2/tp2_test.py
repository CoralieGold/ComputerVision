import os
import sys

from tp2 import *


def tests():    

    

    q1 = ( (differencesMatrix1D(5).todense() == 
        np.array(
        [[-1.,  1.,  0.,  0.,  0.],
         [ 0., -1.,  1.,  0.,  0.],
         [ 0.,  0., -1.,  1.,  0.],
         [ 0.,  0.,  0., -1.,  1.]])).all())
    
    
    Dx,Dy=differencesMatrix2D(3,4)  
    q2a = (Dx.todense()==np.matrix([
        [-1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0., -1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0., -1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0., -1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0., -1.,  1.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0., -1.,  1.,  0.,  0.,  0.,  0.]])).all()
    
    q2b = (Dy.todense()==np.matrix([
        [-1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0., -1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0., -1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  1.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  1.,  0.]])).all()    
    
    im=np.array([[1,2,3,0],[3,2,1,0]])  
    dx,dy=imageGradient(im) 
    q3a = (dx==np.array([[ 1.,  1., -3.]])).all()
    q3b = (dy==np.array([[ 2.,  0., -2.]])).all()
    

    u=im+np.array([[1,-1,0,1],[0,1,1,0]])  
    q4= totalVariationApprox(u,epsilon=0)==6.4787086646190755
   
    
    q5a = costDenoise(im,u,lamb=0,epsilon=0.5)==5.0
    q5b = costDenoise(im,u,lamb=1,epsilon=0) ==11.478708664619075
 
    im_denoised_ls=denoiseImageLeastSquare(im,lamb=5)
    q6a=abs((im_denoised_ls - np.array([
           [ 1.8117829 ,  1.7761033 ,  1.65832824,  1.3819402 ],
           [ 2.00981908,  1.81341942,  1.54860686,  0.        ]]))).max()<1e-8  
    
    
    im_denoised_ls=denoiseImageLeastSquare(im,lamb=0)
    q6b=abs((im_denoised_ls - np.array([
        [ 1.,  2.,  3.,  0.],
        [ 3.,  2.,  1.,  0.]]))).max()<1e-8     
        
    
    im_denoised_ls=denoiseImageLeastSquare(im,lamb=1000000)
    q6c=abs((im_denoised_ls - np.array(
        [[ 1.71428673,  1.71428616,  1.71428502,  1.71428331],
         [ 1.71428802,  1.71428645,  1.71428431,  0.        ]]))).max()<1e-8      
    
    '''
    im=np.array([[1,2,3,0],[3,2,1,0]])  
    im_denoised_ROF,costs=denoiseImageROF(im,lamb=5,nb_iter=10,epsilon=0.5,display=True,tol=0.01,u0=None)
    q7_10a=np.max(abs(np.array(costs)-np.array(([35.246883904340372, 16.067452009595058, 14.181431579403343, 13.934999069773578, 13.904840013576258]))))<1e-8
    q7_10b=np.max(abs(np.array(im_denoised_ROF)-np.array(
       [[ 1.82061273,  1.78843083,  1.67915002,  1.32103773],
       [ 2.03211635,  1.82434463,  1.53430771,  0.        ]])))<1e-8
    
    
    
    #inpainting
    im=np.array([[1,2,3,0],[3,2,1,0]])  
    mask=np.ones(im.shape,dtype=np.bool)
    mask[:,1:2]=False
    im=im*mask
    plt.imshow(im,cmap=plt.cm.Greys_r)
    im_denoised_ROF,costs=im_inpaint,costs=denoiseImageROF(im,lamb=1,nb_iter=10,epsilon=0.2,display=True,tol=0.001,u0=None,dataWeights=10*mask)     
    q11=np.max(abs(np.array(im_denoised_ROF)-np.array(
       [[ 1.07091148,  2.69607282,  2.89812404,  0.04175379],
       [ 2.96186413,  2.69607282,  1.02734661,  0.        ]])))<1e-8    
    '''
    q7_10a = 0
    q7_10b = 0
    q11 = 0
    
    note=1*q1+1*q2a+1*q2b+1*q3a+1*q3b+2*q4+1*q5a+1*q5b+1*q6a+1*q6b+1*q6c+3*q7_10a + 3*q7_10b+2*q11
    print(note)
      
if __name__ == "__main__":
    tests()    
