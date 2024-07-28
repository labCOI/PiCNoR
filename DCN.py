import numpy as np
import copy

class DCN:
    def __init__(self,p0,p1):
        self.p0 = p0
        self.p1 = p1
    def copy(self):
        return copy.deepcopy(self)
    def scalar_mult(self, w):
        return DCN(self.p0*w,self.p1*w)
    def conj(self):
        return DCN(np.conj(self.p0),self.p1) 
    # def DualLog(self):
    #     return DCN(np.log(self.p0), self.p1/self.p0)
    def DualLog(self):
        theta = np.angle(self.p0)
        return DCN(
            theta*1j,
            (theta/np.sin(theta))*self.p1
            )
    def exp(self):
        # print(self.p0,np.exp(self.p0))
        # res = DCN(
        #     np.exp(self.p0),
        #     ((np.exp(self.p0)-np.exp(np.conj(self.p0)))/(self.p0 - np.conj(self.p0)))*self.p1
        # )
        # return res
        theta = np.imag(self.p0)
        return DCN(
            np.exp(theta*1j),
            (np.sin(theta)/theta)*self.p1
        )
        
    def actedby(self,d):
        res = DCN(1+0*1j,0)
        
        res.p1 = (np.real(d.p0)*np.real(d.p0)-np.imag(d.p0)*np.imag(d.p0))*np.real(self.p1)
        +2*(np.real(d.p0)*np.real(d.p1) - np.imag(d.p0)*np.imag(d.p1)-np.real(d.p0)*np.imag(d.p0)*np.imag(self.p1))
        +((np.real(d.p0)*np.real(d.p0)-np.imag(d.p0)*np.imag(d.p0))*np.imag(self.p1)
            +2*(np.real(d.p0)*np.imag(d.p0)*np.real(self.p1) + np.real(d.p0)*np.imag(d.p1) + np.imag(d.p0)*np.real(d.p1)))*1j
        # print(res)
        return res
    def inv(self):
        return DCN(
            np.conj(self.p0),
            -1*self.p1
        )
    def mag(self):
        return abs(self.p0)
    def norm(self):
        norm = np.sqrt(np.real(self.p0)*np.real(self.p0)+np.imag(self.p0)*np.imag(self.p0))
        return norm
    def normalised(self):
        norm = np.sqrt(np.real(self.p0)*np.real(self.p0)+np.imag(self.p0)*np.imag(self.p0))
        return DCN(self.p0/norm,self.p1/norm)
    def __mul__(self,d):
        return DCN(self.p0*d.p0, (self.p1*np.conj(d.p0) + self.p0*d.p1))
    def __truediv__(self,w):
        return DCN(self.p0/w, self.p1/w)
    def __add__(self, d):
        return DCN(self.p0+d.p0,self.p1+d.p1)
    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)
    def isLess(self,e):
        if np.real(self.p0)<e and np.imag(self.p0)<e and np.real(self.p1)<e and np.imag(self.p1)<e:
            return True
        return False
    def toPower(self, power):
        p0_p = self.p0**(power)
        p1_p = (self.p0**(power-1)) * self.p1 * power
        return DCN(p0_p,p1_p)
    def toTransform(self):
        M = np.zeros([3,3])
        M[0,0] = np.real(self.p0*self.p0)
        M[1,1] = np.real(self.p0*self.p0)
        M[1,0] = np.imag(self.p0*self.p0)
        M[0,1] = np.imag(self.p0*self.p0) * -1
        M[0,2] = np.real(2*self.p0*self.p1)
        M[1,2] = np.imag(2*self.p0*self.p1)
        M[2,2] = 1
        return M
    def __str__(self):
        return f'{self.p0} + {self.p1}E'
    def __repr__(self):
        return str(self)