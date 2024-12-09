#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('conda info --envs')


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy import interpolate
import pandas as pd
import os
import emcee
import corner
from getdist import plots, gaussian_mixtures, MCSamples
import time


# # Data

# In[3]:


class Data:
    
    def __init__(self):
        #binned Pantheon SNe Ia data
        self.SNe_full_data = pd.read_csv('Pantheon_SNeIa_dataset/lcparam_DS17f.txt',sep=' ')
        self.SNe_redshift = self.SNe_full_data['zcmb']
        self.SNe_appmag = self.SNe_full_data['mb']
        self.SNe_appmag_unc = self.SNe_full_data['dmb']
        self.distmod = self.SNe_appmag - (-19.25)  #-19.25 = absolute magnitude for Type 1a SNe
        self.distmod_unc = self.SNe_appmag_unc
        
        #CC data
        self.CC_full_data = pd.read_excel('CC_dataset/CC_data.xlsx')
        self.CC_redshift = self.CC_full_data['redshift']
        self.CC_Hub = self.CC_full_data['H(z)']
        self.CC_Hub_unc = self.CC_full_data['sigma']
        
        #BAO data
        self.BAO_full_data = pd.read_excel('BAO_dataset/BAO_data.xlsx')
        self.BAO_redshift = self.BAO_full_data['redshift']
        self.BAO_Thetdeg = self.BAO_full_data['Theta [deg]']
        self.BAO_Thetdeg_unc = self.BAO_full_data['sigma [deg]']
        
        #CMB data
        self.CMB_redshift = 1089
        self.CMB_R = 1.7661
        self.CMB_la = 301.7293
        self.CMB_v_obs = np.array([self.CMB_R,self.CMB_la])
        self.CMB_C = 10**(-8)*np.array([[33483.54, -44417.15],[-44417.15, 4245661.67]]) #covariance matrix
        self.CMB_C_inv = np.linalg.inv(self.CMB_C)
        
        #BBN data (Our "data" is the hubble parameter for LambdaCDM at z=10^9)
        self.BBN_redshift = 10**9
        self.BBN_HLCDM = 67.4*np.sqrt(5*10**(-5)*(1+self.BBN_redshift)**4) #(67.4 is the average Hubble param value today)
    
    
    def plot_SNe_Data(self):
        #plot
        plt.figure()
        plt.errorbar(self.SNe_redshift, self.distmod, yerr=self.distmod_unc, capsize=3, fmt='r.', label='Pantheon data')
        plt.xlabel('$z$')
        plt.ylabel(r'$\mu$')
        plt.legend()
        plt.show()
        return
    
    
    def plot_CC_Data(self):
        #plot
        plt.figure()
        plt.errorbar(self.CC_redshift, self.CC_Hub, yerr=self.CC_Hub_unc, capsize=3, fmt='r.', label='CC data')
        plt.xlabel('$z$')
        plt.ylabel(r'$H\,\,\,[km/s/Mpc]$')
        plt.legend()
        plt.show()
        return
    
    
    def plot_BAO_Data(self):
        #change to radians
        Thet = self.BAO_Thetdeg*np.pi/180 
        Thet_unc = self.BAO_Thetdeg_unc*np.pi/180
        
        #True observable with uncertainty
        self.BAO_DA_rdrag = 1/((1+self.BAO_redshift)*Thet)
        self.BAO_DA_rdrag_unc = Thet_unc/((1+self.BAO_redshift)*Thet**2)
        
        #plot
        plt.figure()
        plt.errorbar(self.BAO_redshift, self.BAO_DA_rdrag, yerr=self.BAO_DA_rdrag_unc, capsize=3, fmt='r.', label='BAO data')
        plt.xlabel('$z$')
        plt.ylabel(r'$D_A/r_{drag}$')
        plt.legend()
        plt.show()
        return
    
    
    def print_CMB_Data(self):
        print('CMB redshift =',self.CMB_redshift)
        print('R =',self.CMB_R)
        print('la =',self.CMB_la)
        print('Covariance matrix =',self.CMB_C)
        return
    
    
    def print_BBN_Data(self):
        print('BBN redshift =',self.BBN_redshift)
        print('Hubble parameter at BBN =',self.BBN_HLCDM)
        return


# # Numerical Solution

# In[4]:


class AXIS_interval:
    
    def __init__(self, x_iv):
        self.m    = x_iv[2]+1
        self.x    = np.linspace(x_iv[0],x_iv[1],self.m)
        self.xrev = np.linspace(x_iv[1],x_iv[0],self.m)
        #self.y    = np.zeros((3,self.m))
        self.ms   = 0
        self.me   = 0
        return


class AXIS:
    
    def __init__(self, x_iv):
        self.AXI = []
        self.niv = len(x_iv)    #no. of intervals
        #Input of xs,xe,n_sample per interval----------
        n = 0
        for i in range(self.niv):
            self.AXI.append(AXIS_interval(x_iv[i]))
            self.AXI[i].ms = n
            n += self.AXI[i].m-1
            self.AXI[i].me = n
        self.m = 0
        for i in range(self.niv):
            self.m += self.AXI[i].m
        self.m -= self.niv-1
        
        self.x = np.zeros(self.m)
        k = 0
        for i in range(self.niv):
            for j in range(self.AXI[i].m):
                self.x[k+j] = self.AXI[i].x[j]
            k += self.AXI[i].m-1
        return

        
    def get_index(self, xi):
        x = self.x
        m = self.m
        mx = 0
        if   xi <= x[0]:   mx = 0
        elif xi >= x[m-1]: mx = m-1
        else:
            for i in range(self.m):
                if x[i] > xi:
                    break
                mx = i
        return mx
    
    
    def get_interval(self, index):
        for i in range(self.niv):
            ms  = self.AXI[i].ms
            me1 = self.AXI[i].me+1
            for j in range(ms,me1):
                if index == j: return i
        return -1

    
    def print_t_axis(self):
        print('\n --- t-axis --------------------------------------')
        print(' t[0] = {0:.3f}'.format(self.x[0]))
        for i in range(self.niv):
            me = self.AXI[i].me
            print(' t[{0:d}] = {1:.3f}'.format(me,self.x[me]))
        print(' -------------------------------------------------')
        return


# In[5]:


class FLEQ:

    def __init__(self, GR, AX):
        self.GR     = GR
        self.AX     = AX
        self.eps    = 1
        self.neq    = 4 
        self.y      = np.zeros((self.neq, AX.m))
        self.yp     = np.zeros(self.neq)
        self.imin   = 0
        self.imax   = AX.m
        self.mv     = 1.e20
        self.prc    = 1.e-12
        self.Om     = 0
        self.Orad   = 0
        self.Ol     = 0
        self.Ok     = 0
        self.h      = 0
        self.Os     = 0
        self.Og     = 0
        self.s1     = 0
        self.Sg1    = 0
        self.H0     = 0
        self.H100   = 2.13312e-42   #[GeV]
        Mpc_to_km   = 3.08567758149137e19
        GeV_to_s    = 6.58212e-25
        self.kmMpcs = Mpc_to_km/GeV_to_s
        self.d_H    = 0
        self.fl_ks  = 0   #flag controlling the integration of the ODE sytem
        self.fl_sw  = 0   #flag controlling the integration of the ODE sytem
        return

    
    def set_prm(self, *args):
        self.Om   = Om   = args[0]
        self.Orad = Orad = args[1]
        self.Ol   = Ol   = args[2]
        self.Ok   = Ok   = args[3]
        self.h    = h    = args[4]
        self.Os   = Os   = args[5]
        self.Og   = Og   = args[6]
        self.s1   = s1   = args[7]
        self.H0   = h*self.H100   #[GeV]
        self.d_H  = 2999.2458/h   # [Mpc]

        # Sigma_1
        M0        = Om/4+Ol
        Og1       = M0*(3*Om/4+Orad)/(Og-M0)
        Ot1       = 1-Om-Orad-Ol-Ok-Og1
        ds1       = (1-Om-Orad-Ol-Ok-Og1+(Os-1)*s1**2)*(Og-M0)
        ds1       = ds1+s1**2+0.5*Os*s1**2*(1+(0.5*Os-1)*s1**2-Ok)
        ds1       = 2*np.sqrt(ds1)-s1
        self.Sg1  = ds1/s1
        return
    
    
    #GR system of equations    
    def FL_equations_GR(self, x, y):
        yp = self.yp
        z  = y[0]
        H  = y[1]
        
        z1 = z+1
        if z < 0 or 1/z1 < 0 or z > self.mv:
            return yp, 1
        
        z2 = z1**2
        z3 = z1**3
        H2 = H**2
        yp[0] = -z1*H
        yp[1] = -2*H2+0.5*self.Om*z3+2*self.Ol+self.Ok*z2
        return yp, 0


    #CCGG system of equations
    def FL_equations_CG(self, x, y):
        Om   = self.Om
        Orad = self.Orad
        Ol   = self.Ol
        Ok   = self.Ok
        Og   = self.Og
        Os   = self.Os
        Os2  = Os/2
        yp   = self.yp
        mv   = self.mv
        flks = self.fl_ks
        flsw = self.fl_sw
      
        z  = y[0]
        H  = y[1]
        s  = y[2]
        w  = y[3]
        if z < 0 or z > 1/self.prc or abs(H) > mv or abs(s) > mv:
            return yp, 1
        
        z1 = z+1
        z2 = z1**2
        z3 = z1**3
        H2 = H**2
        s2 = s**2
        M  = 0.25*Om*z3+Ol
        yp[0] = -z1*H
        yp[1] = -2*H2+2*M+Ok*z2-(Os-1)*s2

        V0 = -Om*z1-Orad*z2-Ol/z2
        ks = (H2*s2-M*(0.75*Om*z3+Orad*z1**4)
              +Os2*s2*(H2+(Os2-1)*s2-Ok*z2)
              +(Og-M)*(H2+V0*z2-Ok*z2+(Os-1)*s2))
        if ks > self.prc:
            yp[2] = -H*s+self.eps*2*np.sqrt(ks)
            flks  = 0
        else: 
            if flsw != 0: return yp, 2
            yp[2] = w*s
            flks  = 1
        s2Hw = s2*(H+w)
        if abs(s2Hw) > self.prc and abs(w) < mv:
            E = 1.5*Om*H*z3-s2*(5*H-(2*Os-1)*w)
            E *= (H2-Ok*z2-2*M+(Os-1)*s2)/s2Hw
            yp[3] = -w**2-2*s2+5*H2-3*H*w+4*Og*(Os-1)-2*Ok*z2+E
            flsw  = 0
        else:            
            if flks != 0: return yp, 3
            flsw = 1
        
        self.fl_ks = flks
        self.fl_sw = flsw
        return yp, 0


    def RK4(self, f, x, y):
        #4th order Runge/Kutta
        #Abramowitz/Stegun (1964/1972), p.896, 25.5.10
        m    = len(x)         
        h    = x[1]-x[0]
        err  = 0
        nerr = -1
        
        for i in range(1, m):
            yh = y[:, i-1]
            k1, err1 = f(x[i-1],     yh)
            k1 = k1*h
            k2, err2 = f(x[i-1]+h/2, yh+k1/2)
            k2 = k2*h
            k3, err3 = f(x[i-1]+h/2, yh+k2/2)
            k3 = k3*h
            k4, err4 = f(x[i-1]+h,   yh+k3)
            k4 = k4*h
            y[:,i] = y[:,i-1]+(k1+2*k2+2*k3+k4)/6

            if err1+err2+err3+err4 != 0:
                #print('RK4: ', err1,err2,err3,err4)
                err  = 1
                nerr = i
                break
        return err, nerr


    def solve_FLeq(self):
        AX  = self.AX
        m   = AX.m
        y   = self.y
        neq = self.neq
        ode_sys = self.FL_equations_CG if self.GR == 'n' else self.FL_equations_GR
        
        # Initial condition for z,H,s,w at tau=1
        y0 = [0,1,self.s1,self.Sg1]
        
        imin = 0
        m1  = AX.get_index(1)
        miv = AX.get_interval(m1)
        k = m1
        y[:,k] = y0
        for i in range(miv,-1,-1):
            xi = AX.AXI[i].xrev
            yi = np.zeros((neq, len(xi)))
            yi[:,0] = y[:,k]
            err, nerr = self.RK4(ode_sys, xi, yi)
            for j in range(AX.AXI[i].m):
                self.y[:,k-j] = yi[:,j]
            if err != 0:
                imin = k-nerr
                break
            k -= AX.AXI[i].m-1

        # Validity range of solution
        self.imin = m-1
        for i in range(m-2,imin-1,-1):
            if y[0,i] > 0 and (1/(y[0,i]+1)) > self.prc and y[0,i] < self.mv:
                self.imin = i
            else: break
        print(self.imin, y[0,self.imin-2:self.imin+2])
        
        # reset for next run
        self.fl_ks = 0
        self.fl_sw = 0
        return

    
    def print_parameters(self):
        print('\n --- Cosmological parameters ---------------------')
        print(' Ω_m                  = %11.3e  []'%self.Om)
        print(' Ω_r                  = %11.3e  []'%self.Orad)
        print(' Ω_Λ                  = %11.3e  []'%self.Ol)
        print(' Ω_K                  = %11.3e  []'%self.Ok)
        print(' h                    = %11.3e  []'%self.h)
        print(' Ω_s                  = %11.3e  []'%self.Os)
        print(' Ω_g                  = %11.3e  []'%self.Og)
        print(' s(τ=1)               = %11.3e  []'%self.s1)
        print(' Σ(τ=1)               = %11.3e  []'%self.Sg1)
        print(' -------------------------------------------------')
        return
    

    def print_solution(self):
        print('\n --- Solution of FL eq. --------------------------')
        print(' τ_min                = %11.3e  []'%self.AX.x[self.imin])
        print(' τ_max                = %11.3e  []'%self.AX.x[-1])
        print(' a(τ=τ_min)           = %11.3e  []'%(1/(self.y[0,self.imin]+1)))
        print(' a(τ=τ_max)           = %11.3e  []'%(1/(self.y[0,-1]+1)))
        print(' z(τ=τ_min)           = %11.3e  []'%self.y[0,self.imin])
        print(' z(τ=τ_max)           = %11.3e  []'%self.y[0,-1])
        print(' H(τ=τ_min)           = %11.3e  []'%self.y[1,self.imin])
        print(' H(τ=τ_max)           = %11.3e  []'%self.y[1,-1])
        print(' s(τ=τ_min)           = %11.3e  []'%self.y[2,self.imin])
        print(' s(τ=τ_max)           = %11.3e  []'%self.y[2,-1])
        print(' -------------------------------------------------')
        return

    
    def draw_zHs(self, xmin, xmax, ymin, ymax):
        imin = self.imin
        imax = self.imax
        a = np.zeros(self.AX.m)
        for i in range(imin,imax):
            a[i] = 1/(self.y[0,i]+1)
        GRCG = 'CG' if self.GR == 'n' else 'GR'
        
        x = self.AX.x
        y = self.y
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        plt.xlabel('$\\tau$')
        plt.ylabel('$z,H,s,a$')
        plt.xscale('linear')
        ax.text(0.05, 0.925, GRCG, ha='left', va='bottom',
                color='k', transform = ax.transAxes)
        ax.axvline(x[imin],color="grey", ls='dotted', zorder=-1)
        ax.axvline(1.,color="grey", ls='dotted', zorder=-1)
        ax.axhline(0.,color="grey", ls='dotted', zorder=-1)
        ax.axhline(1.,color="grey", ls='dotted', zorder=-1)
        plt.plot(x[imin:imax], y[0,imin:imax], label='z')
        plt.plot(x[imin:imax], y[1,imin:imax], label='H')
        plt.plot(x[imin:imax], y[2,imin:imax], label='s')
        plt.plot(x[imin:imax], a[imin:imax], label='a')
        plt.legend(loc='upper right')
        plt.show()
        return


# In[6]:


# distance modulus
class MDLS:

    def __init__(self, FL, GR_asym, Ol_GR):
        self.FL      = FL
        self.GR_asym = GR_asym
        self.Ol_GR   = Ol_GR
        self.zh      = np.zeros(FL.AX.m)
        self.Hh      = np.zeros(FL.AX.m)
        self.muh     = np.zeros(FL.AX.m)
        self.rcdh    = np.zeros(FL.AX.m)
        self.baoh    = np.zeros(FL.AX.m)
        self.ztr     = 1089   # transparency
        self.rtr     = 0      # rs(ztr)
        self.ntr     = 0
        self.zdrag   = 1020   # baryon drag epoch
        self.rdrag   = 0      # rs(zdrag)
        self.ndrag   = 0
        self.z       = 0
        self.Hz      = 0
        self.rcd     = 0
        self.mz      = 0
        self.mu      = 0
        self.rs      = 0
        self.R       = 0
        self.la      = 0
        self.bao     = 0
        self.HBBN_GR = 0
        self.HBBN_CG = 0
    

    # distance modulus
    def calc_mu(self):
        FL  = self.FL
        m   = FL.AX.m
        zh  = self.zh
        Hh  = self.Hh
        muh = self.muh

        for i in range(m):
            zh[i] = FL.y[0,m-1-i]
            Hh[i] = FL.y[1,m-1-i]
        
        # Identify domain where z grows strictly monotonely
        izmin = 0
        izmax = m-FL.imin-1
        for i in range(1,m):
            if zh[i] < zh[i-1]:
                izmax = i-1
                break
                
       #izmax = m-FL.imin
       #izmin = 0
       #for i in range(1,izmax):
       #    if zh[i] < zh[i-1]: izmin = i

        muh[izmin] = 0
        for i in range(izmin+1,izmax):
            muh[i] = muh[i-1]+0.5*(zh[i]-zh[i-1])*(1/Hh[i]+1/Hh[i-1])
        for i in range(izmin+1,izmax):
            muh[i] = 5*np.log10((1+zh[i])*muh[i]*self.FL.d_H)+25

        self.mz  = izmax-izmin            
        self.z   = zh[izmin:izmax]
        self.Hz  = Hh[izmin:izmax]
        self.mu  = muh[izmin:izmax]
        self.rcd = self.rcdh[izmin:izmax]
        self.bao = self.baoh[izmin:izmax]
        print(self.z[izmax-3:izmax])
        return


    # S_k(r)
    def S_k(self, Ok, r):
        # dim(r) = [Mpc]
        res = 0
        if abs(Ok) < self.FL.prc:
            res = r
        elif Ok < -self.FL.prc:
            rf = self.FL.d_H/np.sqrt(-Ok)
            res = rf*np.sin(np.sqrt(-Ok)*r/rf)
        else:
            rf = self.FL.d_H/np.sqrt(-Ok)
            res = rf*np.sinh(np.sqrt(Ok)*r/rf)
        return res
   
    
    # CMB distance priors
    def calc_R_la(self):
        z     = self.z
        Hz    = self.Hz
        mz    = self.mz
        rcd   = self.rcd
        Om    = self.FL.Om
        Ok    = self.FL.Ok
        ztr   = self.ztr
        rtr   = self.rtr
        zdrag = self.zdrag
        rdrag = self.rdrag
       
        # co-moving distance rcd(z)
        ntr    = 0
        ndrag  = 0
        rcd[0] = 0
        for i in range(1,mz):
            dz2 = 0.5*(z[i]-z[i-1])
            rcd[i] = rcd[i-1]+dz2*(1/Hz[i]+1/Hz[i-1])
            if ntr   == 0 and z[i] > ztr:   ntr   = i-1
            if ndrag == 0 and z[i] > zdrag: ndrag = i-1
        rcd *= self.FL.d_H
        self.ntr   = ntr
        self.ndrag = ndrag
        #print('ntr,ndrag: ', ntr,ndrag)

        if ntr != 0:
            # sound horizon rs
            rtr  = rcd[ntr]+(ztr-z[ntr])*(rcd[ntr+1]-rcd[ntr])/(z[ntr+1]-z[ntr])
            rs   = 0
            igr0 = 1/(np.sqrt(1+660/(1+z[ntr]))*Hz[ntr])
            for i in range(ntr+1,mz):
                dz2 = (z[i]-z[i-1])/2
                igr = 1/(np.sqrt(1+660/(1+z[i]))*Hz[i])
                rs += dz2*(igr+igr0)
                igr0 = igr
            self.rtr = rtr
            self.rs  = self.FL.d_H*rs/np.sqrt(3)
    
            # R, la
            skr = self.S_k(Ok, rtr)
            self.R  = np.sqrt(Om)*skr/self.FL.d_H
            if abs(rs) < self.FL.prc:
                self.la  = 0
            else: self.la = np.pi*skr/self.rs
        
        if ndrag != 0:
            # BAO
            rdrag   = 0
            Hdrag = Hz[ndrag]+(zdrag-z[ndrag])*(Hz[ndrag+1]-Hz[ndrag])/(z[ndrag+1]-z[ndrag])
            igr0  = 1/(np.sqrt(1+660/(1+zdrag))*Hdrag)
            #igr0 = 1/(np.sqrt(1+660/(1+z[ndrag]))*Hz[ndrag])
            for i in range(ndrag+1,mz):
                dz2 = 0.5*(z[i]-z[i-1])
                igr = 1/(np.sqrt(1+660/(1+z[i]))*Hz[i])
                rdrag += dz2*(igr+igr0)
                igr0 = igr
            self.rdrag = self.FL.d_H*rdrag/np.sqrt(3)
            
            for i in range(1,mz):
                self.bao[i] = self.S_k(Ok, rcd[i])/((1+z[i])*self.rdrag)
        return

    
    def calc_HBBN(self):
        FL   = self.FL
        Om   = FL.Om
        Orad = FL.Orad
        Ol   = FL.Ol
        Ok   = FL.Ok
        z    = self.z
        Hz   = self.Hz
        
        zB   = 1.e9
        self.HBBN_GR = np.sqrt(Om*zB**3+Orad*zB**4+Ol+Ok*zB**2)
        if FL.GR == 'n':
            i = 0
            for j in range(self.mz):
                if z[j] > zB:
                    i = j
                    break
            if i > 0: self.HBBN_CG = Hz[i]
        return
    

    def print_results(self):
        RPl  = 1.7661
        laPl = 301.7293
        print('\n --- Results -------------------------------------')
        print(' zmin                 = %11.3e  []'%self.z[0])
        print(' zmax                 = %11.3e  []'%self.z[-1])
        print(' z(transparency)      = %11.3e  []'%self.ztr)
        print(' r(transparency)      = %11.3e  [Mpc]'%self.rtr)
        print(' z_drag               = %11.3e  []'%self.zdrag)
        print(' r_drag               = %11.3e  [Mpc]'%(self.rdrag))
        print(' H(zmin)              = %11.3e  []'%self.Hz[0])
        print(' H(zmax)              = %11.3e  []'%self.Hz[-1])
        print(' H_GR(z=10^9)         = %11.3e  []'%self.HBBN_GR)
        print(' H_CG(z=10^9)         = %11.3e  []'%self.HBBN_CG)
        print(' r_sound              = %11.3e  []'%self.rs)
        print(' R                    = %11.3e  []'%self.R)
        print(' R(Planck)            = %11.3e  []'%RPl)
        print(' la                   = %11.3e  []'%self.la)
        print(' la(Planck)           = %11.3e  []'%laPl)
        print(' -------------------------------------------------')
        return


    def draw_H(self, xmin, xmax, ymin, ymax):
        z    = self.z
        Hz   = self.Hz*self.FL.kmMpcs*self.FL.H0
        GRCG = 'CG' if self.FL.GR == 'n' else 'GR'
        
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        plt.xlabel(r'$z$')
        plt.ylabel(r'$H \, [\mathrm{km}/(\mathrm{s} \cdot \mathrm{Mpc})]$')
        ax.text(0.05, 0.925, GRCG, ha='left', va='bottom',
                color='k', transform = ax.transAxes)
        ax.axvline(z[-1], color="grey", ls='dotted', zorder=-1)
        ax.plot(z, Hz,'k', label=r'$H(z)$')
        plt.show()
        return


    def draw_rcd(self, xmin, xmax, ymin, ymax):
        if self.ntr == 0 or self.ndrag == 0: return
        z     = self.z
        rcd   = self.rcd
        rtr   = self.rtr
        GRCG = 'CG' if self.FL.GR == 'n' else 'GR'
        
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        plt.xlabel(r'$z$')
        plt.ylabel(r'$r_{cd} [Mpc]$')
        ax.text(0.05, 0.925, GRCG, ha='left', va='bottom',
                color='k', transform = ax.transAxes)
        ax.axvline(self.ztr, color="grey", ls='dotted', zorder=-1)
        ax.plot(z, rcd, 'k', label=r'$r_{cd}$')
        ax.plot(self.ztr, rtr, 'o', color='r', label=r'$r_{tr}$')
        plt.legend()
        plt.show()
        return

    
    def draw_mu(self, xmin, xmax, ymin, ymax, obs_redshift, obs_distmod, obs_distmod_unc):
        z    = self.z
        mu   = self.mu
        GRCG = 'CG' if self.FL.GR == 'n' else 'GR'
        
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        plt.xlabel(r'$z$')
        plt.ylabel(r'$\mu \, (z)$')
        ax.text(0.05, 0.925, GRCG, ha='left', va='bottom',
                color='k', transform = ax.transAxes)
        plt.errorbar(obs_redshift, obs_distmod, yerr=obs_distmod_unc, capsize=3, fmt='r.', label='Pantheon data')
        plt.plot(z, mu, color='b',label=r'$\mu$')
        plt.legend()
        plt.show()
        return 


    def draw_bao(self, xmin, xmax, ymin, ymax, BAO_redshift, BAO_DA_rdrag, BAO_DA_rdrag_unc):
        if self.ndrag == 0: return
        z    = self.z
        bao  = self.bao
        GRCG = 'CG' if self.FL.GR == 'n' else 'GR'
        
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        plt.xlabel(r'$z$')
        plt.ylabel(r'$D_A/r_{drag} \, (z)$')
        ax.text(0.05, 0.925, GRCG, ha='left', va='bottom',
                color='k', transform = ax.transAxes)
        plt.errorbar(BAO_redshift, BAO_DA_rdrag, yerr=BAO_DA_rdrag_unc, capsize=3, fmt='r.', label='BAO data')
        plt.plot(z, bao, color='k',label=r'$D_A/r_{drag}$')
        plt.legend()
        plt.show()
        return 


# # MCMC

# In[7]:


class MCMCRunner:  
    #global class variables for CMB
    ls_av = 299792.458 #lightspeed in km/s
    z_CMB = 1089
    R = 1.7661
    la = 301.7293
    v_obs = np.array([R,la])
    C = 10**(-8)*np.array([[33483.54, -44417.15],[-44417.15, 4245661.67]])
    C_inv = np.linalg.inv(C)

    #global class variables for BBN
    redshift_BBN = 1e9
    H0_av = 67.4
    HLCDM = H0_av*np.sqrt(5*10**(-5)*(1 + redshift_BBN)**4)

    #global class variables for BAO
    zmax_BAO = 2.225

    def __init__(self, parameters, fixed_values, prior_range, datasets_used, calc_mcmc_input, n_iterations):
        """
        Initializes the MCMCRunner class.

        Args:
            parameters (dict): A dictionary where keys are parameter names, and values are either 'y' (optimize) or 'n' (do not optimize).
        """
        self.parameters = parameters
        self.optimized_params = [name for name, flag in parameters.items() if flag == 'y']
        self.fixed_params = [name for name, flag in parameters.items() if flag == 'n']

        # Initial ranges for the uniform prior
        self.initial = prior_range

        # Assign values of fixed parameters from the fixed_values dictionary
        self.values_of_fixed = {name: fixed_values[name] for name in self.fixed_params}

        # Number of parameters to optimise
        self.numberofparams = len(self.optimized_params)

        # Which datasets we are using
        self.whichdata = datasets_used

        self.calc_mcmc_input = calc_mcmc_input

        self.n_iterations = n_iterations


    def printparams(self):
        print('\n --- Parameters/Datasets -------------------------------------')
        print(' \n #Parameters to optimise               =', self.numberofparams)
        print(' Parameters to optimise                =', self.optimized_params)
        print(' Prior ranges                          =', self.initial)
        print(' \n Fixed Parameters                      =', self.fixed_params)
        print(' Values of fixed Parameters            =', self.values_of_fixed)
        print(' \n Datasets used                         =', self.whichdata)
        print(' \n -------------------------------------------------------------')
        return


    # Uniform log_prior
    def log_prior(self, CurrentParamVal):
        log_prior_val = 0.0
        for i, param in enumerate(self.optimized_params):
            value = CurrentParamVal[i]
            
            # Get the prior bounds for the parameter from self.initial
            param_range = self.initial.get(param)
            
            # Check if the value is within the bounds of the uniform prior
            if param_range[0] <= value <= param_range[1]:
                continue  # Continue if the parameter is within the prior bounds
            else:
                return -np.inf  # Return -infinity if the value is out of bounds

            # Uniform prior density (not necessary to return the value, we assume uniform prior in range)
            log_prior_val += -np.log((param_range[1] - param_range[0]))  # Uniform prior density

        return log_prior_val


    # Define the log_likelihood functions
    def log_likelihood(self, CurrentParamVal, MD, dataset, FL):
        log_likelihood_val = 0.0
        args = []

        optimized_index = 0  # To track position in CurrentParamVal

        for param_name, flag in self.parameters.items():
            if flag == 'y':  # Optimized parameter
                args.append(CurrentParamVal[optimized_index])
                optimized_index += 1
            else:  # Fixed parameter
                args.append(self.values_of_fixed.get(param_name))
        #print('current args = ', args)
        
        for dataset_name, flag in self.whichdata.items():
            if flag == 'n':
                continue
                
            if dataset_name == 'SNeIa':
                z, mu, R, la, bao, HBBN = self.calc_mcmc_input(MD, *args)

                #print("z range:", z.min(), z.max())
                #print("dataset.SNe_redshift.values range:", dataset.SNe_redshift.values.min(), dataset.SNe_redshift.values.max())
                #print("args:", args)

                # Check if z contains values greater than or equal to 2
                if max(z) < 2:
                    # Return -\infty to make the MCMC take a new step
                    return -np.inf
    
                f = interpolate.interp1d(z, mu, kind = 'linear')

                log_likelihood_val += -0.5*np.sum(((dataset.distmod.values - f(dataset.SNe_redshift.values))/dataset.distmod_unc.values)**2 + np.log(2*np.pi*dataset.distmod_unc.values**2))

            if dataset_name == 'CC':
                z, mu, R, la, bao, HBBN = self.calc_mcmc_input(MD, *args)

                # Check if z contains values greater than or equal to 2
                if max(z) < 2:
                    # Return -\infty to make the MCMC take a new step
                    return -np.inf
                
                #f = interpolate.interp1d(FL.y[0,:], FL.y[1,:], kind = 'linear')
                f = interpolate.interp1d(z, MD.Hz, kind = 'linear')
    
                log_likelihood_val += -0.5*np.sum(((dataset.CC_Hub.values - f(dataset.CC_redshift.values))/dataset.CC_Hub_unc)**2 + np.log(2*np.pi*dataset.CC_Hub_unc**2))
            
            if dataset_name == 'BAO':
                z, mu, R, la, bao, HBBN = self.calc_mcmc_input(MD, *args)
                # Check if z contains values greater than or equal to 2.225
                if max(z) < MCMCRunner.zmax_BAO:
                    # Return -\infty to make the MCMC take a new step
                    return -np.inf
                
                f = interpolate.interp1d(z, bao, kind = 'linear')
    
                log_likelihood_val += -0.5*np.sum(((dataset.BAO_DA_rdrag.values - f(dataset.BAO_redshift.values))/dataset.BAO_DA_rdrag_unc)**2 + np.log(2*np.pi*dataset.BAO_DA_rdrag_unc**2))
            
            if dataset_name == 'CMB': 
                z, mu, R, la, bao, HBBN = self.calc_mcmc_input(MD, *args)

                # Check if z contains values greater than or equal to 1089
                if max(z) < MCMCRunner.z_CMB:
                    # Return -\infty to make the MCMC take a new step
                    return -np.inf

                v = np.array([R,la])

                log_likelihood_val += -0.5*np.matmul((v - MCMCRunner.v_obs).transpose(), np.matmul(MCMCRunner.C_inv, v - MCMCRunner.v_obs))
            
            if dataset_name == 'BBN': 
                z, mu, R, la, bao, HBBN = self.calc_mcmc_input(MD, *args)

                # Check if z contains values greater than or equal to 10^9
                if max(z) < MCMCRunner.redshift_BBN:
                    # Return -\infty to make the MCMC take a new step
                    return -np.inf

                log_likelihood_val += -0.5*((MCMCRunner.HLCDM - HBBN)/MCMCRunner.HLCDM)**2
                
        return log_likelihood_val


    def log_sum(self, CurrentParamVal, MD, dataset, FL):
        # Compute the log-prior
        lp = self.log_prior(CurrentParamVal)
        if lp == -np.inf:
            # If the prior is zero (i.e. log is -infinity), return negative infinity
            return -np.inf
        
        # Compute the log-likelihood
        ll =  self.log_likelihood(CurrentParamVal, MD, dataset, FL)
        
        # Compute the log-posterior as the sum of the log-prior and log-likelihood
        return lp + ll



    def runmcmc(self, MD, dataset, FL):
        nwalkers = 3*self.numberofparams

        # Generate random starting positions for the walkers within the defined range
        p0 = np.zeros((nwalkers, self.numberofparams))
        for i, param_name in enumerate(self.initial):
            lower, upper = self.initial[param_name]  # Extract the bounds from the prior_range dictionary
            p0[:, i] = np.random.uniform(lower, upper, size=nwalkers)  # Sample the values 

        # Initialize the sampler
        self.sampler = emcee.EnsembleSampler(nwalkers, self.numberofparams, self.log_sum, args=(MD, dataset, FL))

        # Run the sampler for the specified number of iterations
        self.sampler.run_mcmc(p0, self.n_iterations, progress=True)



    def mcmc_corner_plot(self):
        # Get the chain of samples produced by the sampler
        samples = self.sampler.chain[:, :, :]
        #print(sampler.chain[:,:,:].shape)
        
        # Discard the first few samples as burn-in
        burnin = 100
        final_samples = samples[:,burnin:, :]
        #print(final_samples.shape)

        # Flatten the chain of samples
        self.flat_samples = final_samples.reshape(-1, self.numberofparams)
        #print(flat_samples.shape)
        
        self.labels = []
        for param in self.optimized_params:
                self.labels.append(r'$\Omega_{' + param.split('_')[1] + '}$')
                
        figure = corner.corner(self.flat_samples,bins=50, labels=self.labels)
        
        # Show the plot
        plt.show()


    def mcmc_get_dist_plot(self):
        names = self.optimized_params
        labels = [r'\Omega_{' + param.split('_')[1] + '}' for param in self.optimized_params]
        
        samples_new = MCSamples(samples=np.array(self.flat_samples),names=names,labels=labels)
        
        g = plots.get_subplot_plotter()
        g.triangle_plot(samples_new, filled=True)

        # Show the plot
        plt.show()



    def print_best_parameters(self):
        self.m = []
        self.std = []
        
        for i in range(self.numberofparams):
            self.m.append(np.mean(self.flat_samples[:,i]))
            self.std.append(np.std(self.flat_samples[:,i]))

        for i in range(self.numberofparams):
            print(f'Mean of {self.optimized_params[i]} = {self.m[i]}')
            print(f'Standard deviation of {self.optimized_params[i]} = {self.std[i]}')



    def plot_best_fit_SNe(self, MD, dataset):
        args_fit = []

        for param_name, flag in self.parameters.items():
            if flag == 'y':  # Optimized parameter
                param_index = self.optimized_params.index(param_name)
                args_fit.append(self.m[param_index])
            else:  # Fixed parameter
                args_fit.append(self.values_of_fixed.get(param_name))

        z_fit, mu_fit, R_fit, la_fit, bao_fit, HBBN_fit = self.calc_mcmc_input(MD, *args_fit)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(0.01, 2)
        ax.set_ylim(32, 46)
        plt.errorbar(dataset.SNe_redshift, dataset.distmod, yerr=dataset.distmod_unc, capsize=3, fmt='r.', label='Pantheon data')
        plt.plot(z_fit,mu_fit,label='fit')
        plt.xlabel('$z$')
        plt.ylabel(r'$\mu$')
        plt.legend()
        plt.show()


    def plot_best_fit_CC(self, MD, dataset):
        args_fit = []

        for param_name, flag in self.parameters.items():
            if flag == 'y':  # Optimized parameter
                param_index = self.optimized_params.index(param_name)
                args_fit.append(self.m[param_index])
            else:  # Fixed parameter
                args_fit.append(self.values_of_fixed.get(param_name))

        z_fit, mu_fit, R_fit, la_fit, bao_fit, HBBN_fit = self.calc_mcmc_input(MD, *args_fit)

        fig = plt.figure()
        plt.errorbar(dataset.CC_redshift, dataset.CC_Hub, yerr=dataset.CC_Hub_unc, capsize=3, fmt='r.', label='CC data')
        plt.plot(MD.z,MD.Hz*67.4,label='fit') #check the 67.4!!!
        plt.xlim(0,2)
        plt.ylim(0,300)
        plt.xlabel('$z$')
        plt.ylabel('$H(z)$')
        plt.legend()
        plt.show()


    def plot_best_fit_BAO(self, MD, dataset):
        args_fit = []

        for param_name, flag in self.parameters.items():
            if flag == 'y':  # Optimized parameter
                param_index = self.optimized_params.index(param_name)
                args_fit.append(self.m[param_index])
            else:  # Fixed parameter
                args_fit.append(self.values_of_fixed.get(param_name))

        z_fit, mu_fit, R_fit, la_fit, bao_fit, HBBN_fit = self.calc_mcmc_input(MD, *args_fit)

        fig = plt.figure()
        plt.errorbar(dataset.BAO_redshift, dataset.BAO_DA_rdrag, yerr=dataset.BAO_DA_rdrag_unc, capsize=3, fmt='r.', label='BAO data')
        plt.plot(z_fit,bao_fit,label='fit') 
        plt.xlim(0,2.5)
        plt.ylim(0,15)
        plt.xlabel(r'$z$')
        plt.ylabel(r'$D_A/r_{drag}$')
        plt.legend()
        plt.show()


    def print_CMB_priors(self, MD):
        args_fit = []
        
        for param_name, flag in self.parameters.items():
            if flag == 'y':  # Optimized parameter
                param_index = self.optimized_params.index(param_name)
                args_fit.append(self.m[param_index])
            else:  # Fixed parameter
                args_fit.append(self.values_of_fixed.get(param_name))
                
        z_fit, mu_fit, R_fit, la_fit, bao_fit, HBBN_fit = self.calc_mcmc_input(MD, *args_fit)
        
        print('\n --- CMB distance priors -------------------------------------')
        print(' \n [R_obs, la_obs]                   =', MCMCRunner.v_obs)
        print(' [R_fitted, la_fitted]                =', [R_fit,la_fit])
        print(' \n -------------------------------------------------------------')
        

    def print_HBBN(self, MD):
        args_fit = []
        
        for param_name, flag in self.parameters.items():
            if flag == 'y':  # Optimized parameter
                param_index = self.optimized_params.index(param_name)
                args_fit.append(self.m[param_index])
            else:  # Fixed parameter
                args_fit.append(self.values_of_fixed.get(param_name))
                
        z_fit, mu_fit, R_fit, la_fit, bao_fit, HBBN_fit = self.calc_mcmc_input(MD, *args_fit)
        
        print('\n --- Hubble param at BBN -------------------------------------')
        print(' \n HBBN from LCDM               =', MCMCRunner.HLCDM)
        print(' HBBN from fit                =', HBBN_fit)
        print(' \n -------------------------------------------------------------')
                

