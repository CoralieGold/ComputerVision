#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy import ndimage
from skimage.io import imread
from skimage.io import imsave
import pickle
from scipy import sparse
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve,cg
#import cv2


def pause():
    plt.draw() 
    plt.pause(0.001)
    raw_input("Press Enter to continue...")  
    

        
def plotImageEtResidu(im,u):        
    plt.subplot(1,3,1)
    plt.imshow(im,cmap=plt.cm.Greys_r)
    plt.subplot(1,3,2)
    plt.imshow(u,cmap=plt.cm.Greys_r)
    plt.subplot(1,3,3)
    plt.imshow(u-im,cmap=plt.cm.Greys_r)        
    plt.draw() 
   




def differencesMatrix1D(N):
    """ creates the sparse matrix of size N-1 by N that allows to compute
        the difference between successive elements of a vector through a simple multiplication
        differencesMatrix1D(5).todense() should give        
        matrix([[-1.,  1.,  0.,  0.,  0.],
                [ 0., -1.,  1.,  0.,  0.],
                [ 0.,  0., -1.,  1.,  0.],
                [ 0.,  0.,  0., -1.,  1.]])
    """
    #TO DO create the matrix using sparse.diags with the optional shape argument
    D = diags([0, -1, 1], [-1, 0, 1], shape=(N-1, N))
    return D


def differencesMatrix2D(H,W):
    # TO DO : code this function that allows to compute the two matrices Dx et Dy defined in the class presentation
    # for H=3 and W=4 we want
    #Dx.todense()=
    #matrix([[-1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
            #[ 0., -1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
            #[ 0.,  0., -1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
            #[ 0.,  0.,  0.,  0., -1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
            #[ 0.,  0.,  0.,  0.,  0., -1.,  1.,  0.,  0.,  0.,  0.,  0.],
            #[ 0.,  0.,  0.,  0.,  0.,  0., -1.,  1.,  0.,  0.,  0.,  0.]])     
    # Dy.todense() =
    #matrix([[-1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
            #[ 0., -1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
            #[ 0.,  0., -1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
            #[ 0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
            #[ 0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  1.,  0.,  0.],
            #[ 0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  1.,  0.]])
 
    # TO DO: code this function using sparse.kron and sparse.eye(with two arguments)
    Dx = sparse.kron(sparse.eye(H-1, n=H), differencesMatrix1D(W))
    Dy = sparse.kron(differencesMatrix1D(H), sparse.eye(W-1, n=W))
    return Dx,Dy

def imageGradient(u):
    """given an image of size M by N this function returns tow images dx and dy of size M-1 by N-1 that contain the graidnet in the x and in the y direction"""
    # Warning: in the sum i should stop at u.shape[0]-1 and j et u.shape[1]-1
    # you can use a convolution to get the gradient or simple numpy array operateurs with numpy indexing operations like a[:-1,:]  
    # example:
    # im=np.array([[1,2,3,0],[3,2,1,0]])    
    # dx,dy=imageGradient(im)    
    # dx
    # array([[ 1.,  1., -3.]])
    # dy
    # array([[ 2.,  0., -2.]])        
	
    dx = u[:-1, 1:] - u[:-1, :-1]
    dy = u[1:, :-1] - u[:-1, :-1]

    return dx,dy

def smoothNorm(x,y,epsilon):
    """this function implement the smooth version of the gradient norm for the TV approximation  seen in class
    it also return the influence function value """
    h=np.sqrt((x**2+y**2+epsilon**2).astype(float))
    psi=1./h
    return h,psi

def totalVariationApprox(u,epsilon):
    """this function calculare the approximated total variation of an image u using smoothing with epsilon 
    tv= sum_ij sqrt( epsilon^2+(u(i+1,j)-u(i,j))^2 + (u(i,j+1)-u(i,j))^2)
    """
    # TODO implement the approximation of the total variation by calling imageGradient
    dx, dy = imageGradient(u)
    tv = np.sum(np.sqrt( epsilon**2 + dx**2 + dy**2))
    return tv  



def costDenoise(f,u,lamb,epsilon,dataWeights=None): 
    """This function return the sum of the data term and the regularization term"""
    # TODO: code this function by calling totalVariationApprox for the regularisation term, use the dataWeigts to give 
    # differents weights to different pixels in the data term if provided
    
    data = (u-f)**2
    if(dataWeights):
        data *= dataWeights
  
    cost = np.sum(data) + lamb * np.sum(totalVariationApprox(u, epsilon))

    return cost



def denoiseImageLeastSquare(im,lamb,u0=None):
    #TODO implement the denoising method that uses regular least squares i.e base on the regularization 
    # term int_x int_y norm(gradient(u(x,y)))^2 dx
    # as seen in the class
    # use the function spsolve to solve the sparse linear system if u0==None 
    # otherwise use the function cg with x0=u0 and a tolerance of 1e-4 and maxiter=50
    
    Dx, Dy = differencesMatrix2D(im.shape[0], im.shape[1]);
    u = np.transpose(Dx)*Dx + np.transpose(Dy)*Dy
    u = u * lamb
    u = u + np.identity(u.shape[0])
    u = np.linalg.inv(u)*im.reshape(-1,1)
    u = u.reshape((im.shape[0], im.shape[1]))
    return u




def denoiseImageROF(im,lamb,nb_iter,epsilon=1,display=True,tol=0.15,u0=None,dataWeights=None):
   
    im=im.astype(float)
    Iv=im.flatten()
    nb_pixels=im.shape[0]*im.shape[1]
    
    # creating the Dx and Dy matrices
    Dx,Dy=differencesMatrix2D(im.shape[0],im.shape[1]) 
    
    #possiblity to provide a "hot start" u0 to the minimization
    if u0==None:
        u=im.astype(float).copy()      
    else:
        u=u0.copy()
    
    Uv=u.flatten() 
    costs=[]
    c=costDenoise(im, u,lamb,epsilon,dataWeights=dataWeights) 
    costs.append(c) 
    if display:
        plt.ion()    
        plt.figure()
    for t in range(nb_iter):
        print('.'),
        
        # TODO :compute the weights gamma_ij as seen in the reweigthed least sqare method to do ROF denoising
        # you can use the Dx and Dy matrices du compute the intensitie differences
        
       
        #TODO : construction the sparse matrix Gamma_d that conains the gammas in the row-major order on its diagonal 
       
        
        if dataWeights==None:
            #TODO compute the matrix M that need to be invered of used in a linear system solver
            # such that the minimum with respect to U_v is obtained as Uv=M^-1 Iv i.e by solving M*Uv= Iv
                 
            
            #TODO call cg with 10 iterations (in scipy.sparse.linalg) to solve the system system M*Uv= Iv
            pass
            
        else:
            # TODO: adapt the matrix M above in case there are different weights for each pixel provided by the input matrix dataWeights
            # warning : call cg with 100 iterations
            pass
            
        # check that the function decrease and stop if the decrease is small that some tolerance
        u=Uv.reshape((u.shape[0],u.shape[1]))
        nc=costDenoise(im, u,lamb,epsilon,dataWeights=dataWeights  )
        
        assert(nc<c+1e-8)
        costs.append(nc)
        if nc>c-tol*nb_pixels:
            break
        c=nc       
        
        if display:           
                plt.subplot(2,2,1)
                plt.imshow(im,cmap=plt.cm.Greys_r)
                plt.subplot(2,2,2)
                plt.imshow(u,cmap=plt.cm.Greys_r)
                plt.subplot(2,2,3)
                plt.imshow(u-im,cmap=plt.cm.Greys_r)   
                plt.subplot(2,2,4)
                plt.imshow(gammas,cmap=plt.cm.Greys_r)                 
                
    plt.draw() 
    return u,costs


def signalToNoiseRatio(im,im_denoised):
    #TODO implement the signal to noise ratio
    return SNR




def main():
    print '*************************'
    print 'differencesMatrix1D'
    print differencesMatrix1D(5).todense()
    #[[-1.  1.  0.  0.  0.]
    #[ 0. -1.  1.  0.  0.]
    #[ 0.  0. -1.  1.  0.]
    #[ 0.  0.  0. -1.  1.]] 
    
    print '*************************'
    print 'differencesMatrix2D'
    Dx,Dy=differencesMatrix2D(3,4)    
    print Dx.todense()
    #[[-1.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
    #[ 0. -1.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
    #[ 0.  0. -1.  1.  0.  0.  0.  0.  0.  0.  0.  0.]
    #[ 0.  0.  0.  0. -1.  1.  0.  0.  0.  0.  0.  0.]
    #[ 0.  0.  0.  0.  0. -1.  1.  0.  0.  0.  0.  0.]
    #[ 0.  0.  0.  0.  0.  0. -1.  1.  0.  0.  0.  0.]]    
    print Dy.todense()
    #[[-1.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.]
    #[ 0. -1.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.]
    #[ 0.  0. -1.  0.  0.  0.  1.  0.  0.  0.  0.  0.]
    #[ 0.  0.  0.  0. -1.  0.  0.  0.  1.  0.  0.  0.]
    #[ 0.  0.  0.  0.  0. -1.  0.  0.  0.  1.  0.  0.]
    #[ 0.  0.  0.  0.  0.  0. -1.  0.  0.  0.  1.  0.]]    
    
    print '*************************'
    print 'imageGradient'
    im=np.array([[1,2,3,0],[3,2,1,0]])  
    dx,dy=imageGradient(im)    
    print dx #array([[ 1.,  1., -3.]])
    print dy  #array([[ 2.,  0., -2.]])    
    
    print '*************************'
    print 'smoothNorm'
    x=np.array([[5,6,7],[1,2,3]])
    y=np.array([[2,1,1],[3,1,1]])          
    norm,influence=smoothNorm(x,y,0.5)
    print norm
    #[[ 5.40832691  6.10327781  7.08872344]
     #[ 3.20156212  2.29128785  3.20156212]] 
    print influence
    #[[ 0.18490007  0.16384638  0.14106912]
     #[ 0.31234752  0.43643578  0.31234752]] 
  
    print '*************************'
    print 'totalVariationApprox'
    u=im+np.array([[1,-1,0,1],[0,1,1,0]])  
    print totalVariationApprox(u,epsilon=0)
    #6.47870866462
    
    print '*************************'
    print 'costDenoise'
    print costDenoise(im,u,lamb=0,epsilon=0.5)
    #5.0
    print costDenoise(im,u,lamb=5,epsilon=0.5)
    #38.3178458537
    print costDenoise(im,u,lamb=1,epsilon=0)
    #11.4787086646    
    
    print '*************************'
    print 'denoiseImageLeastSquare'
    im_denoised_ls=denoiseImageLeastSquare(im,lamb=5)
    print im_denoised_ls
    #[[ 1.8117829   1.7761033   1.65832824  1.3819402 ]
    # [ 2.00981908  1.81341942  1.54860686  0.        ]]   
    im_denoised_ls=denoiseImageLeastSquare(im,lamb=0)
    print im_denoised_ls    
    #[[ 1.  2.  3.  0.]
    # [ 3.  2.  1.  0.]]  
    im_denoised_ls=denoiseImageLeastSquare(im,lamb=1000000)
    print im_denoised_ls      
    #[[ 1.71428673  1.71428616  1.71428502  1.71428331]
    # [ 1.71428802  1.71428645  1.71428431  0.        ]] 
    
    
    im_denoised_ROF,costs=denoiseImageROF(im,lamb=5,nb_iter=10,epsilon=0.5,display=False,tol=0.01,u0=None)
    print costs
    #[35.246883904340372, 16.067452009595062, 14.181431579403345, 13.934999069773578, 13.904840013576258]
    print im_denoised_ROF
    #[[ 1.82061273  1.78843083  1.67915002  1.32103773]
    # [ 2.03211635  1.82434463  1.53430771  0.        ]]   
    
    # debruitage de l'image
    im=np.array(imread('einstein.jpg'))
    im=scipy.ndimage.interpolation.zoom(im, 0.5)
    sigma=20
    im_noisy=im+np.random.normal(0,sigma,im.shape) 
    
    im_denoised_ls=denoiseImageLeastSquare(im_noisy,lamb=20)
    plt.figure()
    plt.ion()
    plotImageEtResidu(im_noisy,im_denoised_ls) 
    
    snr=[]
    lambdas=np.linspace(0,5,50)
    for lamb in lambdas:
        im_denoised_ls=denoiseImageLeastSquare(im_noisy,lamb=lamb,u0=im_denoised_ls)
        snr.append(signalToNoiseRatio(im,im_denoised_ls))
    plt.figure()
    #snr=[0.28157179640151553, 0.50161739818660078, 0.68209445978304939, 0.80829629834945049, 0.88544314866075624, 0.92597201259578055, 0.94182789039768355, 0.94203049824851515, 0.93280252381735118, 0.9181203573148552, 0.90050747578751622, 0.88151244058601275, 0.86212920433435891, 0.84291705332954325, 0.82421389522617083, 0.80620699781117167, 0.78898429549325733, 0.77259302368733174, 0.75702649996643567, 0.74226407797305427, 0.72827235828787418, 0.7150115807945957, 0.70243928256872079, 0.69050984859987907, 0.67918730765366808, 0.6684276485244065, 0.6581935076572567, 0.64844933385418579, 0.6391620236591824, 0.63030088531184669, 0.62183754007822922, 0.61374579258498729, 0.60599947930055831, 0.59858085507740977, 0.59146635152119165, 0.58463756059556227, 0.57807704024863116, 0.57176867639632756, 0.56569758195320252, 0.55984999122409873, 0.55421316313175994, 0.54877529296782035, 0.5435254321009152, 0.53845341502764188, 0.53354818842354901, 0.52880468622456911, 0.5242120844547864, 0.51976324900268644, 0.515451062309635, 0.51126886308082331]

    plt.plot(lambdas,snr)
    
    im_denoised_ROF,costs=denoiseImageROF(im,lamb=50,nb_iter=60,epsilon=0.1,display=True,tol=0.01,u0=None)    
    plotImageEtResidu(im,im_denoised_ROF)
    
    snr=[]
    lambdas=np.linspace(0,100,50)
    for lamb in lambdas:
        print lamb
        im_denoised_ROF,costs=denoiseImageROF(im_noisy,lamb=lamb,nb_iter=60,epsilon=0.1,display=False,tol=0.01,u0=im_denoised_ROF) 
        snr.append(signalToNoiseRatio(im,im_denoised_ROF))
    plt.figure()   
    #snr=[0.28157179640151558, 0.33097713759746783, 0.38974730877754654, 0.45886589871800787, 0.53816687763087945, 0.62819600279500876, 0.72628928924215297, 0.82702774019455294, 0.92661300123031876, 1.0155020415183087, 1.0898205635369946, 1.1431442605360018, 1.1751621061637179, 1.1876785607149793, 1.1836168133812877, 1.1677466368652414, 1.1440702910698552, 1.1168445550786334, 1.0865083103383895, 1.0560197023430606, 1.0257268657901999, 0.996559703961577, 0.96859775971124318, 0.94218510078482076, 0.91729470656537948, 0.89376772169351304, 0.87285899288352586, 0.85113489356685712, 0.83096409088343748, 0.81322334636720606, 0.79598703265204329, 0.77933137777958394, 0.76266203255881526, 0.748305645819904, 0.73441758146603309, 0.72108651130298074, 0.70822742648699644, 0.69605201783892712, 0.68444766884439356, 0.67329655101586972, 0.66259977189325958, 0.65240185062207612, 0.64267649159228801, 0.63330486812864883, 0.62432967160519104, 0.61570208733128984, 0.60739324849598786, 0.59941233125218563, 0.59173536905689716, 0.58434922099962794]
    plt.plot(lambdas,snr)   
    
    #inpainting
    im=np.array(imread('einstein.jpg')) [280:340,290:380]
    mask=np.ones(im.shape,dtype=np.bool)
    mask[10:50,40:50]=False
    im=im*mask
    plt.imshow(im,cmap=plt.cm.Greys_r)
    im_inpaint,costs=denoiseImageROF(im,lamb=1,nb_iter=100,epsilon=0.2,display=True,tol=0.001,u0=None,dataWeights=mask) 
      
if __name__ == "__main__":
    main()
