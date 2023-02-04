import numpy as np
import matplotlib.pyplot as plt

#vars for sigma and alpha
sigma=1*5e-1
alpha=1*5e-1
#size of dataset
N=50
#size of test values
M=100
#number of samples
S=25
#function
fname="0.15*exp(0.6*x-3)"

#posterior
'''funct posterior takes:
   design matrix dm, datapoints y, sigma, and alpha
   and returns:
   mean and cov matrix'''
def posterior(dm,y,sigma=sigma,alpha=alpha):
    Sigma_p=alpha*np.eye(dm.shape[0])
    Sigma_p_inv=np.linalg.inv(Sigma_p)
    A=(sigma**(-2))*dm.dot(dm.T)+Sigma_p_inv
    A_inv=np.linalg.inv(A)
    mean=(1/sigma**2)*A_inv.dot(dm).dot(y)
    cov=A_inv
    return mean,cov

'''funct epsilon takes:
   size and var sigma
   and returns:
   iid noise N(0,sigma^2)'''
def epsilon(size,sigma=sigma):
    epsilon=np.random.normal(loc=0,scale=sigma**2,size=size)
    return epsilon


#underlying functions  


'''funct linear_f takes:
   value x and sigma for noise
   and returns:
   target y+noise of linear funct'''
def linear_f(x,sigma=sigma):
    y=0.3+0.5*x+epsilon(x.shape,sigma)
    return y


def exponential_f(x,sigma=sigma):
    y=0.7*np.exp(x)+epsilon(x.shape,sigma)
    return y

def exponential_f2(x,sigma=sigma):
    y=0.15*np.exp(0.6*x-3)+epsilon(x.shape,sigma)
    return y

def logistic_f(x,sigma=sigma):
    y=(3/(1+np.exp(-2*x+10)))+epsilon(x.shape,sigma)
    return y

def sinusoidal_f(x,sigma=sigma):
    y=0.32*(x+np.sin(x))+epsilon(x.shape,sigma)
    return y

'''
the following function are basisfunctions for the designmatrix
'''
def identitybf(x):
    return x

def linearbf1(x):
    y=2+0.5*x
    return y

def linearbf2(x):
    y=1+2*x
    return y

def logisticbf1(x):
    y=1-(1/(np.exp(x)))
    return y

def logisticbf11(x):
    y=(1/(1+np.exp(-x)))
    return y

def logisticbf2(x):
    y=1/(0.3+np.exp(-x+6))
    return y

def logisticbf3(x):
    y=1/(0.15+np.exp(-x+3))
    return y

def logisticbf4(x):
    y=1/(np.exp(x))
    return y

def exponentialbf1(x):
    y=0.15*np.exp(0.6*x-3)
    return y

def exponentialbf2(x):
    y=0.2*np.exp(0.3*x)
    return y

def exponentialbf3(x):
    y=0.3*np.exp(0.3*x)
    return y

def exponentialbf4(x):
    y=np.exp(x)
    return y

def tanhbf1(x):
    y=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    return y

def tanhbf2(x):
    y=tanhbf1(x-2)+1
    return y

def tanhbf3(x):
    y=tanhbf1(x-3)+1
    return y

def tanhbf4(x):
    y=tanhbf1(x-4)+1
    return y

def tanhbf5(x):
    y=tanhbf1(x-5)+1
    return y


'''
funct design_matrix takes
transposed vector x, basis functions bfs
and returns:
Matrix with len(bfs)+1 rows
'''
def design_matrix(x, bfs):
    return np.concatenate([np.ones(x.shape)]+[bf(x) for bf in bfs],axis=1).T

#fit model to exponential function

#evenly spaced values
input=np.linspace(0,10,N).reshape(-1,1)
#y positions of X
y=exponential_f2(input,sigma)
#M test values between
test_input=np.linspace(0,10,M).reshape(-1,1)
# linear funct without noise 
y_noiseless=exponential_f2(test_input,sigma=0) 
#functs for design matrix
monofuncts=np.array([identitybf,linearbf1,logisticbf1,tanhbf1,exponentialbf4])  
# design matrix of test values
dm_test=design_matrix(test_input,monofuncts)

#plot

plt.figure(figsize=(9,6))

#convert input to design matrix
dm=design_matrix(input,monofuncts)

#predict
mean,cov=posterior(dm,y,sigma,alpha)

#vars
count=0
w_s=[]
non_mon=0

#draw S monotone samples
while count<S:
    x=np.random.multivariate_normal(mean.ravel(),cov).T
    if x[len(x)-1]<0 or x[len(x)-2]<0 or x[len(x)-3]<0 or x[len(x)-4]<0 or x[len(x)-5]<0:
       non_mon+=1
       if non_mon%1000==0:
        print("nomono:{}".format(non_mon)) 
    else:
        w_s.append(list(x))
        count+=1
        print("{}/{} monotonic trajectories".format(count,S))
w_s=np.array(w_s).T
y_s=dm_test.T.dot(w_s)



plt.scatter(input,y,color='k',marker='x')
plt.plot(test_input,y_noiseless,'k--')
plt.plot(test_input, y_s, 'r-', alpha=0.1)
#vars for rmse and sd:
RMSE_ges=0
lst=[]
for i in range(1,y_s.shape[1]):
    #calc rmse and sd
    diff=np.subtract(y_noiseless,y_s[i])
    square=np.square(diff)
    MSE=square.mean()
    RMSE=np.sqrt(MSE)
    RMSE_ges+=RMSE
    lst.append(RMSE)
    plt.plot(test_input,y_s[:,i],'r-',alpha=0.5)
RMSE_mean=RMSE_ges/y_s.shape[1]
sd=np.array(lst).std()
plt.legend(["observations","function","predictions"])
plt.xlim(0,10)
plt.title("Samples={},Datapoints={},Sigma={},Function={}\n RMSE={:.2f}(x100),sd={:.2f}(x100),nomon={}".format(S,N,sigma,fname,RMSE_mean*100,sd*100,non_mon))
plt.grid()
plt.show()
