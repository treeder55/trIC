import pyvisa as visa
from pyvisa.constants import Parity
import numpy as np
import matplotlib.pyplot as plt
from struct import unpack
import datetime as dt
import time
from threading import Timer
from scipy.optimize import curve_fit

def sinefunc(xdata,amp,freq,phase,bg):
    return amp*np.sin(freq*xdata+phase)+bg
def expfuncinc(xdata,amp,tau,bg):
    return amp*(1-np.exp(-xdata/tau))+bg
def expfuncdec(xdata,amp,tau,bg):
    return guess[0]*(np.exp(-xdata[xlim[0]:xlim[1]]/guess[1]))+guess[2]

    # this will be our standard class all devices will be using.
class VISADevice:
    def __init__(self, address_string):
        self.address = address_string
        #self.dev = visa.ResourceManager('@py').open_resource(self.address, resource_pyclass = MessageBasedResource)
        self.dev = visa.ResourceManager().open_resource(self.address)
        print('initialized')    
    def whoami(self):
        print(self.dev.query('*IDN?'))
    def query(self, cmd_string):
        return self.dev.query(cmd_string)        
    def write(self, cmd_string):
        self.dev.write(cmd_string)
    def read(self):
        return self.dev.read()
    def read_raw(self):
        return self.dev.read_raw()
    def timeout(self, t = 1): # args in seconds
        self.dev.timeout = t*1000 # pyvisa uses the units, milliseconds
        print(str(self.dev.timeout))
    def read_termination(self, cmd_string):
        self.dev.read_termination = cmd_string
    def parity(self, par):
        self.dev.parity = par
    def data_bits(self, b):
        self.dev.data_bits = b
class magnet:
    startfield = 0.0
    endfield = 0.0
    fld = 0.0
    scantime = 0.0
    numberofsteps = 0
    H = []
    H2 = []
    t = []
    def __init__(self, address_string):
        self.address = address_string
        #self.dev = visa.ResourceManager('@py').open_resource(self.address, resource_pyclass = MessageBasedResource)
        self.dev = visa.ResourceManager().open_resource(self.address,read_termination='\n')
        print('initialized')
    def whoami(self):
        print(self.dev.query('*IDN?'))
    def query(self, cmd_string):
        return self.dev.query(cmd_string)        
    def write(self, cmd_string):
        self.dev.write(cmd_string)        
    def read(self):
        return self.dev.read()
    def read_raw(self):
        return self.dev.read_raw()
    def setfield(self, field):
        self.startfield = self.query('get field sample')
        self.startfield = float(self.startfield[3:11])
        self.endfield = field
        print(self.query('set field setpoint %f' % field))
    def getfield(self):
        self.fld= str(self.query('get field sample'))
        self.fld = float(self.fld[3:11])
    def getfieldscan(self,timestep):
        self.scantime = np.abs(self.endfield - self.startfield)/(0.0045) # in seconds 
        self.numberofsteps = int(self.scantime/timestep)
        self.H = []
        self.t = []
        starttime = time.time()
        for i in range(self.numberofsteps+5):
            time.sleep(timestep)
            tempH = str(self.query('get field sample'))
            self.t.append(time.time()-starttime)
            self.H.append(float(tempH[3:11]))
        print("scan complete")
    def plotfield(self):
        fig1 = plt.figure()
        ax1 = fig1.add_axes([0,0,1,1])
        ax1.plot(self.t,self.H,marker='o',color='mediumpurple')
        ax1.set_xlabel('t (s)')
        ax1.set_ylabel('H (T)')
    def getfieldsetpoint(self):
        print(self.query('get field setpoint'))
    def getfieldstatus(self):
        print(self.query('get field status'))
class LakeShore10T:
    def __init__(self, address_string):
        self.address = address_string
        #self.dev = visa.ResourceManager('@py').open_resource(self.address, resource_pyclass = MessageBasedResource)
        self.dev = visa.ResourceManager().open_resource(self.address,baud_rate=9600, data_bits=7, read_termination='\r', 
    parity=Parity.odd)
        print('initialized')
    def whoami(self):
        print(self.dev.query('*IDN?'))
    def query(self, cmd_string):
        return self.dev.query(cmd_string)        
    def write(self, cmd_string):
        self.dev.write(cmd_string)        
    def read(self):
        return self.dev.read()
    def setT(self,T):
        self.write('SETP 1,%f' % T)
    def readsampleT(self):
        self.T = float(self.query('KRDG? A'))
    def readvtiT(self):
        self.vtiT = float(self.query('KRDG? B'))
class osc(VISADevice):
    I = []
    Q = []
    Det = []
    data = np.zeros((2,1),dtype=float)
    data1 = np.zeros((2,1),dtype=float)
    data2 = np.zeros((2,1),dtype=float)
    data3 = np.zeros((2,1),dtype=float)
    def SetHorizontalScale(self, *scale):
        self.write('Hor:scal %f' % scale[0])
        print(self.query('Hor:scal?'))
    def settriggermode(self, trigmode):  # 'normal' or 'auto'
        self.write('trig:A:mod %s' % trigmode)
    def measure(self, channel):
        self.dev.write('DATA:SOU CH%i' % channel) #select source channel
        self.dev.write('DATA:WIDTH 1') #select data width in bytes per point
        self.dev.write('DATA:ENC RPB') #specifies outgoing data format. RPB is binary
           
        ymult = float(self.dev.query('WFMPRE:YMULT?')) 
        yzero = float(self.dev.query('WFMPRE:YZERO?'))
        yoff = float(self.dev.query('WFMPRE:YOFF?'))
        x_increment = float(self.dev.query('WFMPRE:XINCR?'))
            
        self.dev.write('CURVE?') #this is the command that queries for the data
        self.data = self.dev.read_raw()
        headerlen = 2+int(self.data[1])
            
        ADC_wave = self.data[headerlen:-1] #get rid of header
        ADC_wave = np.array(unpack('%sB' % len(ADC_wave),ADC_wave))  #unpack binary data
        Volts = (ADC_wave - yoff)*ymult +yzero   # map data from sent integers to the correct values
        Time = np.arange(0, x_increment*len(Volts), x_increment)
        self.data = np.column_stack((Time, Volts)) #make 2d array of data
        
        self.data = np.transpose(self.data)
        
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.plot(self.data[0],self.data[1],marker = 'x')
        ax.set_ylabel('Voltage (check units)')
        ax.set_xlabel('t (s)')
        
        print(len(self.data[0]))
    def triggermodenormal(self):
        self.write('trig:A:mod normal')
    def triggermodeauto(self):
        self.write('trig:A:mod auto')
    def measureFFT(self):
        self.write('MATH:DEFINE "FFT( CH2 )"') #select source channel
        self.write('DATA:SOU MATH')
        #osci.write('DATA:WIDTH 1') #select data width in bytes per point
        #osci.write('DATA:ENC RPB') #specifies outgoing data format. RPB is binary
        self.write('WFMO:ENC BINARY')
        self.write('WFMO:BYT_N 2')
        ymult = float(self.query('WFMPRE:YMULT?')) 
        yzero = float(self.query('WFMPRE:YZERO?'))
        yoff = float(self.query('WFMPRE:YOFF?'))
        x_increment = float(self.query('WFMPRE:XINCR?'))
            
        self.write('CURVE?') #this is the command that queries for the data
        self.dataa = self.read_raw()
        headerlen = 2+int(self.dataa[1])
            
        FFT_curve = self.dataa[headerlen:-1] #get rid of header
        FFT_curve = np.array(unpack('>%sH' % int(len(FFT_curve)/2),FFT_curve))
#        FFT_curve = np.array(unpack('%sB' % len(FFT_curve),FFT_curve))  #unpack binary data
        Voltsrms = (FFT_curve - yoff)*ymult +yzero   # map data from sent integers to the correct values
        frequency = np.arange(0, x_increment*len(Voltsrms), x_increment)
        self.dataa= np.column_stack((frequency, Voltsrms)) #make 2d array of data
        
        self.dataa = np.transpose(self.dataa)
       
        fig = plt.figure()
        ax1 = fig.add_axes([0,1,0.8,0.8])
        ax1.plot(self.dataa[0],self.dataa[1],marker = 'x',color='peru')
        ax1.set_ylabel('Voltage (check units)')
        ax1.set_xlabel('f (Hz)')
        ax1.set_title('FFT')
    def saveFFT(self,prefix):
        np.save(prefix+'_FFT_I_'+time.strftime("%H%M_%m%d%Y"),self.dataa)
    def AFG_pulse(self,width,period,offset,amp):
        self.dev.write('AFG:OFFS %f' % offset)
        self.dev.write('AFG:FUNC PULSE')
        self.dev.write('AFG:PERI %f' % period)
        self.dev.write('AFG:PUL:WID %f' % width)
        self.dev.write('AFG:AMPL %f' % amp)
    def AFG_DC(self,val):
        self.dev.write('AFG:FUNC DC')
        self.dev.write('AFG:OFFS %f' % val)        
    def measureall(self,yfactor):
        self.data = {}
        for i in range(4):
            self.write('DATA:SOU CH%i' % (i+1)) #select source channel
            self.write('DATA:START 0')
            self.write('DATA:STOP 100000')
             #select data width in bytes per point
            #osci.dev.write('DATA:ENC RPB') #specifies outgoing data format. RPB is binary
            self.write('WFMO:ENC BINARY')
            #osci.dev.write('WFMO:BN_F RP')
            #osci.dev.write('DATA:WIDTH 8')
            #self.dev.write('WFMO:BIT_N 8')
            #self.write('WFMO:BYT_N 1')
            self.write('WFMO:BYT_N 2')
            ymult = float(self.query('WFMPRE:YMULT?'))*yfactor
            yzero = float(self.query('WFMPRE:YZERO?'))
            yoff = float(self.query('WFMPRE:YOFF?'))
            x_increment = float(self.query('WFMPRE:XINCR?'))
                
            self.write('CURVE?') #this is the command that queries for the data
            self.data[i] = self.read_raw()
            headerlen = 2+int(self.data[i][1])
            
            ADC_wave = self.data[i] [headerlen:-1] #get rid of header
            ADC_wave = np.array(unpack('>%sH' % int(len(ADC_wave)/2),ADC_wave))
            #ADC_wave = np.array(unpack('>%sB' % int(len(ADC_wave)),ADC_wave))
            Volts = (ADC_wave - yoff)*ymult +yzero   # map data from sent integers to the correct values
            Time = np.arange(0, x_increment*len(Volts), x_increment)
            self.data[i]= np.column_stack((Time, Volts)) #make 2d array of data
            
            self.data[i] = np.transpose(self.data[i])
    def plotdata(self):
        ch = ['Q','I','A','B']
        colors = ['gold','blue','purple','green']
        for i in range(4):
            fig = plt.figure()
            ax1 = fig.add_axes([0,i,0.8,0.8])
            ax1.plot(self.data[i][0],self.data[i][1],marker = 'x',color=colors[i])
            ax1.set_ylabel(ch[i]+' (V)')
            ax1.set_xlabel('t (s)')
            ax1.set_title('Quadrature')
    def savedata(self,filename):
        ch = ['Q','I','A','B']
        for i in range(4):
            np.save(filename+'_'+ch[i]+'_'+time.strftime("%m%d%Y"),self.data[i])
    def savedataQ(self,filename):
        np.save(filename+'_Q_'+time.strftime("%m%d%Y"),self.data[0])
    def savedataI(self,filename):
        np.save(filename+'_I_'+time.strftime("%m%d%Y"),self.data[1])
    def savedataD(self,filename):
        np.save(filename+'_D_'+time.strftime("%m%d%Y"),self.data['D'])
    def savedata4(self,filename):
        np.save(filename+'_4_'+time.strftime("%m%d%Y"),self.data['4'])
    def fittosine(self,channel,guess):
        if channel == 1:
            est, covar = curve_fit(sinefunc,self.data1[0],self.data1[1],p0=guess)
            yfit = sinefunc(self.data1[0],est[0],est[1],est[2],est[3])
            fig = plt.figure()
            ax1 = fig.add_axes([0,1,0.8,0.8])
            ax1.plot(self.data1[0],self.data1[1],marker = 'x',color='gold')
            ax1.plot(self.data1[0],yfit,color='black')
            ax1.set_ylabel('Voltage (check units)')
            ax1.set_xlabel('t (s)')
            ax1.set_title('Q (V)')
            ax1.text(0,0.0205,'frequency = '+str(est[1]/(2*3.14159)) +' Hz \n amp = '+str(est[2])+' V')
        if channel == 2:
            est, covar = curve_fit(sinefunc,self.data2[0],self.data2[1],p0=guess)
            yfit = sinefunc(self.data2[0],est[0],est[1],est[2],est[3])
            fig = plt.figure()
            ax1 = fig.add_axes([0,1,0.8,0.8])
            ax1.plot(self.data2[0],self.data2[1],marker = 'x',color='blue')
            ax1.plot(self.data2[0],yfit,marker = 'x',color='gold')
            ax1.set_ylabel('Voltage (check units)')
            ax1.set_xlabel('t (s)')
            ax1.set_title('I (V)')      #        
        if channel == 3:
            est, covar = curve_fit(sinefunc,self.data3[0],self.data3[1],p0=guess) 
            yfit = sinefunc(self.data3[0],est[0],est[1],est[2],est[3])
            fig = plt.figure()
            ax1 = fig.add_axes([0,1,0.8,0.8])
            ax1.plot(self.data3[0],self.data3[1],marker = 'x',color='purple')
            ax1.plot(self.data3[0],yfit,marker = 'x',color='gold')
            ax1.set_ylabel('Voltage (check units)')
            ax1.set_xlabel('t (s)')
            ax1.set_title('Det (V)')     #        
        if channel == 4:
            est, covar = curve_fit(sinefunc,self.data4[0],self.data4[1],p0=guess) 
            yfit = sinefunc(self.data4[0],est[0],est[1],est[2],est[3])
            fig = plt.figure()
            ax1 = fig.add_axes([0,1,0.8,0.8])
            ax1.plot(self.data4[0],self.data4[1],marker = 'x',color='red')
            ax1.plot(self.data4[0],yfit,marker = 'x',color='gold')
            ax1.set_ylabel('Voltage (check units)')
            ax1.set_xlabel('t (s)')
            ax1.set_title('signal')     #    
    def fittoexp(self,channel,guess,xlim,INCorDEC):
        if INCorDEC == 'inc':
            if channel == 1:
                est, covar = curve_fit(expfuncinc,self.data1[0][xlim[0]:xlim[1]],self.data1[1][xlim[0]:xlim[1]],p0=guess)
                self.fit1 = expfuncinc(self.data1[0][xlim[0]:xlim[1]],est[0],est[1],est[2])
                fig = plt.figure()
                ax1 = fig.add_axes([0,1,0.8,0.8])
                ax1.plot(self.data1[0],self.data1[1],marker = 'x',color='orange')
                ax1.plot(self.data1[0][xlim[0]:xlim[1]],self.fit1,color='black')
                ax1.set_ylabel('Voltage (check units)')
                ax1.set_xlabel('t (s)')
                ax1.set_title('  amp = ' + str(est[0]) + '  tau = ' + str(est[0]) + '  start = ' + str(est[2])) 
            if channel == 2:
                est, covar = curve_fit(expfuncinc,self.data2[0][xlim[0]:xlim[1]],self.data2[1][xlim[0]:xlim[1]],p0=guess)
                self.fit2 = expfuncinc(self.data2[0][xlim[0]:xlim[1]],est[0],est[1],est[2])
                fig = plt.figure()
                ax1 = fig.add_axes([0,1,0.8,0.8])
                ax1.plot(self.data2[0],self.data2[1],marker = 'x',color='blue')
                ax1.plot(self.data2[0][xlim[0]:xlim[1]],self.fit2,color='black')
                ax1.set_ylabel('Voltage (check units)')
                ax1.set_xlabel('t (s)')
                ax1.set_title('  amp = ' + str(est[0]) + '  tau = ' + str(est[0]) + '  start = ' + str(est[2]))     #        
            if channel == 3:
                est, covar = curve_fit(expfuncinc,self.data3[0][xlim[0]:xlim[1]],self.data3[1][xlim[0]:xlim[1]],p0=guess) 
                self.fit3 = expfuncinc(self.data3[0][xlim[0]:xlim[1]],est[0],est[1],est[2])
                fig = plt.figure()
                ax1 = fig.add_axes([0,1,0.8,0.8])
                ax1.plot(self.data3[0],self.data3[1],marker = 'x',color='red')
                ax1.plot(self.data3[0][xlim[0]:xlim[1]],self.fit3,marker = 'x',color='purple')
                ax1.set_ylabel('Voltage (check units)')
                ax1.set_xlabel('t (s)')
                ax1.set_title('  amp = ' + str(est[0]) + '  tau = ' + str(est[0]) + '  start = ' + str(est[2]))    #        
            if channel == 4:
                est, covar = curve_fit(expfuncinc,self.data4[0][xlim[0]:xlim[1]],self.data4[1][xlim[0]:xlim[1]],p0=guess)
                self.fit4 = expfuncinc(self.data4[0],est[0],est[1],est[2])
                fig = plt.figure()
                ax1 = fig.add_axes([0,1,0.8,0.8])
                ax1.plot(self.data4[0],self.data4[1],marker = 'x',color='red')
                ax1.plot(self.data4[0][xlim[0]:xlim[1]],self.fit4,marker = 'x',color='red')
                ax1.set_ylabel('Voltage (check units)')
                ax1.set_xlabel('t (s)')
                ax1.set_title('  amp = ' + str(est[0]) + '  tau = ' + str(est[0]) + '  start = ' + str(est[2]))    #            
        if INCorDEC == 'dec':
            if channel == 1:
                est, covar = curve_fit(expfuncdec,self.data1[0],self.data1[1],p0=guess) 
                self.fit1 = expfuncdec(self.data1[0],est[0],est[1],est[2])
                fig = plt.figure()
                ax1 = fig.add_axes([0,1,0.8,0.8])
                ax1.plot(self.data1[0],self.data1[1],marker = 'x',color='gold')
                ax1.plot(self.data1[0][xlim[0]:xlim[1]],self.fit1,marker = '-',color='black')
                ax1.set_ylabel('Voltage (check units)')
                ax1.set_xlabel('t (s)')
                ax1.set_title('  amp = ' + str(est[0]) + '  tau = ' + str(est[0]) + '  start = ' + str(est[2]))   #        
            if channel == 2:
                est, covar = curve_fit(expfuncdec,self.data2[0],self.data2[1],p0=guess) 
                self.fit2 = expfuncdec(self.data2[0],est[0],est[1],est[2])
                fig = plt.figure()
                ax1 = fig.add_axes([0,1,0.8,0.8])
                ax1.plot(self.data2[0],self.data2[1],marker = 'x',color='red')
                ax1.plot(self.data2[0][xlim[0]:xlim[1]],self.fit2,marker = 'x',color='blue')
                ax1.set_ylabel('Voltage (check units)')
                ax1.set_xlabel('t (s)')
                ax1.set_title('  amp = ' + str(est[0]) + '  tau = ' + str(est[0]) + '  start = ' + str(est[2]))   #        
            if channel == 3:
                est, covar = curve_fit(expfuncdec,self.data3[0],self.data3[1],p0=guess) 
                self.fit3 = expfuncdec(self.data2[0],est[0],est[1],est[2])
                fig = plt.figure()
                ax1 = fig.add_axes([0,1,0.8,0.8])
                ax1.plot(self.data2[0],self.data2[1],marker = 'x',color='red')
                ax1.plot(self.data2[0][xlim[0]:xlim[1]],self.fit3,marker = 'x',color='purple')
                ax1.set_ylabel('Voltage (check units)')
                ax1.set_xlabel('t (s)')
                ax1.set_title('  amp = ' + str(est[0]) + '  tau = ' + str(est[0]) + '  start = ' + str(est[2]))   #        
            if channel == 4:    
                est, covar = curve_fit(expfuncdec,self.data4[0],self.data4[1],p0=guess)
                self.fit4 = expfuncdec(self.data4[0],est[0],est[1],est[2])
                fig = plt.figure()
                ax1 = fig.add_axes([0,1,0.8,0.8])
                ax1.plot(self.data4[0],self.data4[1],marker = 'x',color='red')
                ax1.plot(self.data4[0][xlim[0]:xlim[1]],self.fit4,marker = 'x',color='red')
                ax1.set_ylabel('Voltage (check units)')
                ax1.set_xlabel('t (s)')
                ax1.set_title('  amp = ' + str(est[0]) + '  tau = ' + str(est[0]) + '  start = ' + str(est[2]))
    def updateplot(self):
        starttime = time.time()
        fig, (fig1,fig2,fig3) = plt.subplots(3,1,figsize=(6,14))
        fig1.set_xlabel('t (s)')
        fig2.set_xlabel('t (s)')
        fig3.set_xlabel('t (s)')
        fig1.set_ylabel('Receiver Q (V)')
        fig1.set_title('Receiver Q (V)')
        fig2.set_ylabel('Receiver I (V)')
        fig2.set_title('Receiver I (V)')
        fig3.set_ylabel('Detector (V)')
        fig3.set_title('Detector (V)')
        while time.time() - starttime < 3600:
            time.sleep(1)
            self.write(':MEASU:IMM:SOU CH1')
            self.write(':MEASU:IMM:TYP MEan') # this mean integrates over the entire waveform, so the integration time is controlled by the horizontal scale on the oscilloscope.  
            b = float(str(self.query(':MEASU:IMM:VAL?')))
            a = time.time()-starttime
            fig1.plot(a,b,marker='o',color='mediumpurple')
            fig1.figure.canvas.draw()
            self.write(':MEASU:IMM:SOU CH2')
            self.write(':MEASU:IMM:TYP MEan') # this mean integrates over the entire waveform, so the integration time is controlled by the horizontal scale on the oscilloscope.  
            d = float(str(self.query(':MEASU:IMM:VAL?')))
            c = time.time()-starttime
            fig2.plot(c,d,marker='o',color='mediumpurple')
            fig2.figure.canvas.draw()  
            self.write(':MEASU:IMM:SOU CH3')
            self.write(':MEASU:IMM:TYP MEan') # this mean integrates over the entire waveform, so the integration time is controlled by the horizontal scale on the oscilloscope.  
            f = float(str(self.query(':MEASU:IMM:VAL?')))
            e = time.time()-starttime
            fig3.plot(e,f,marker='o',color='mediumpurple')
            fig3.figure.canvas.draw()
class awg:
    def __init__(self,address_string):
        self.address = address_string
        self.dev = visa.ResourceManager().open_resource(self.address,read_termination='\n')
        print('initialized')
    def whoami(self):
        print(self.dev.query('*IDN?'))
    def query(self, cmd_string):
        return self.dev.query(cmd_string)        
    def write(self, cmd_string):
        self.dev.write(cmd_string)        
    def read(self):
        return self.dev.read()
    def read_raw(self):
        return self.dev.read_raw()
    def run(self,ch):
        self.write('OUTP%i 1' % ch)
    def stop(self,ch):
        self.write('OUTP%i 0' % ch)
    def queryfuncs():
        print('{SINusoid|SQUare|TRIangle|RAMP|PULSe|PRBS|NOISe|ARB|DC}')
    def setfuncpulse(self,ch,amplitude,offset,period,dcycle):
        self.write('SOUR%i:VOLT %f' % (ch,amplitude))
        self.write('SOUR%i:VOLT:OFFS %f' % (ch,offset))
        self.write('SOUR%i:FUNC PULS' % ch)
        self.write('SOUR%i:FUNC:PULS:PER %f' % (ch,period)) #units: seconds
        #self.write('SOUR%i:FUNC:PULS:WIDT %f' % (ch,width)) #units: percent
        self.write('SOUR%i:FUNC:PULS:DCYC %f' % (ch,dcycle)) #units: percent
    def setmodulation(self,ch,typ,freq,depth):
        d = {'SUM':'AMPL','AM':'DEPT'}
        self.write('SOUR%i:%s:SOUR INT' % (ch,typ))
        self.write('SOUR%i:%s:INT:FREQ %f' % (ch,typ,freq))
        self.write('SOUR%i:%s:%s %f' % (ch,typ,d[typ],depth))
        self.write('SOUR%i:%s:STAT 1' % (ch,typ))
    def turnoffmodulation(self,ch,typ):
        self.write('SOUR%i:%s:STAT 0' % (ch,typ))
class keithley(VISADevice):
    dc = 0
    def ACvolt(self):
        self.write(':abort:conf:volt:ac')
        time.sleep(2)
        print(self.query(':read?'))
        
    def DCvolt(self):
        temp = str(self.query('meas:volt:dc?'))
        self.dc = float(temp[0:-1])
    
class pulsegen(VISADevice):
    # the default channel is set to 1
    ch = 1
    
    # equivalent to pressing the Run/Stop button on the pulse generator
    # input needs to be either 'ON' or 'OFF'
    def run(self):
        self.write(':inst:stat ON')
        self.read()
    def stop(self):
        self.write(':inst:stat OFF')
        self.read()
    
    def channels(self):
        print(self.query(':inst:full?'))
    def setchannel(self,chan):
        self.ch = chan
    # changes edge to trigger on. This command changes ALL of the channels
    # input string must be either 'rising' or 'falling'
    def triggeredge(self,inputstring):
        if inputstring == 'rising':
            self.write(':puls0:edg:ris')
            self.read()
        if inputstring == 'falling':
            self.write(':puls0:edg:fall')
            self.read()
    def setmux(self,m):
        command = ':puls' + str(self.ch) + ':mux ' + str(m)
        self.write(command)
    # for channel, ch, change the amplitude of the pulse (in Volts) from 2 V 
    # to 20 V
    def setamp(self,a):
        self.write(':puls%i:outp:ampl %f' % (self.ch, a))
        self.read()
        
    # for all of the channels sets the pulse width, t (in s, from 10 ns to 
    # 999.9 s) (cannot set this for individual channels), and for channel, ch, 
    # the period p, (in s, from 50 ns to 999.9 s) 
    def setpulse(self, t, delay = 0):
        widthcommand = ':puls' + str(self.ch) + ':widt ' + str(t)
        delaycommand = ':puls' + str(self.ch) + ':delay ' + str(delay)
        self.write(widthcommand)
        self.read()
        self.write(delaycommand)
        self.read()
    def setperiod(self,p):
        self.write(':spulse:per %s' % p)
        self.read()
    def bncmode(self,mode):
        self.write(':puls%i:cmod %s' % (self.ch,mode))
        self.read()
    def dutycycle(self,non,noff,nwait):
        self.write(':puls%i:cmod dcycle' % self.ch)
        self.read() 
        self.write(':puls%i:pco %i' % (self.ch,non))
        self.read() 
        self.write(':puls%i:oco %i' % (self.ch,noff))
        self.read() 
        self.write(':puls%i:wco %i' % (self.ch,nwait))
        self.read() 
    # starts pulsing channel ch
    def runchannel(self):
        command = ':puls' + str(self.ch) + ':state ON'
        self.write(command)
        self.read()
    # stops pulsing channel ch
    def stopchannel(self):
        command = ':puls' + str(self.ch) + ':state OFF'
        self.write(command)
        self.read()
    def setpolarity(self,polarity): # inv, norm, or comp
        self.write(':puls%i:pol %s' % (self.ch,polarity))
        self.read()
        
class AgilentE3640(VISADevice):
    def __init__(self,address_string):
        VISADevice.__init__(self,address_string)
        self.write('*RCL 1')
    def setvoltage(self,V):
        self.write('VOLT %f' % V)
    def queryVandI(self):
        print(self.query('APPL?'))
        
class lockin(VISADevice):
    def __init__(self,address_string):
        VISADevice.__init__(self,address_string)
        self.read_termination('\r')
    lockdata = np.zeros((2,1),dtype=float)
    datanumber = np.zeros((2,1),dtype=float)
    def savesetup(self, spot):
        self.write('SSET $i' % spot) # save current setup to buffer 1<=spot<=9.
    def loadsetup(self, buff):
        self.write('RSET %i' % buff) # load setup from buffer 1<=buff<=9
    def setsensitivity(self, sens):
        self.write('SENS %i' % sens) # set sensitivity from 2 nV (0) to 1 V (26) rms full scale.
    def setdisplayXYRtheta(self, chan, disp):
        self.write('DDEF %i, %i' % (chan,disp))  # set channel 1 (1) or channel 2 (2) to XY, Rtheta, XnYn, or Aux 1,3 or Aux 2,4 (0,1,2,3,4)
    def expand(self, channel, offset, expand):
        self.write('OEXP %i, %f, %i' % (channel,offset,expand)) # increases the gain by 1, 10, or 100.
    def autogain(self):
        self.write('AGAN') # equivalent to pressing autogain button
    # following commands are for data storage and access. The buffer has 16383 storage bins
    def buffershot0orloop1(self, inp):
        self.write('SEND %i' % inp) # changes between shot mode, i=0, (ends data storage after buffer is full) or loop mode, i=1, (keeps taking data to replace buffer after buffer is full)
    def startscan(self):
        self.write('STRT')
    def pausescan(self):
        self.write('PAUS')
    def stopscanandclear(self):
        self.write('REST') # stops scan and clears buffer storage
    def getdatafastmode(self):
        self.write('FAST 2;STRD')
        self.lockdata = self.read()   # need to finish this function by reading to variable.
    def setsamplerate(self, rate):
        convertedrate = int(np.log2(rate)+4)
        self.write('SRAT %i' % convertedrate) # 0<=i<=14 corresponding to 0.0625,0.125,0.25,0.5,1,2,4,8,16,32,64,128,256,512,"Trigger" in units Hz
    def getdataASCIIfp(self,args=[0,16383]):   # args = [startbin,endbin], which are the bins of data to request.
        self.datanumber = np.arange(args[0],args[1])
        #self.filledbins = int(self.query('SPTS ?')[:-1])
        x = self.query('TRCA? 1,%i,%i' % (args[0],args[1])) # must pause before getting data this way. 
        y = self.query('TRCA? 2,%i,%i' % (args[0],args[1]))
        self.lockdata = np.zeros((2,args[1]-args[0]),dtype=float)
        self.lockdata[0] = [float(i) for i in x[:-2].split(',')]
        self.lockdata[1] = [float(i) for i in y[:-2].split(',')]
    def plot(self):
        fig1 = plt.figure()
        ax1 = fig1.add_axes([0,0,1,0.5])
        ax2 = fig1.add_axes([0,0.55,1,0.5])
        ax1.plot(self.datanumber,self.lockdata[0],marker='o')
        ax2.plot(self.datanumber,self.lockdata[1],marker='o')
        ax1.set_xlabel('index')
        ax1.set_ylabel('in phase V_rms (V)')
        ax2.set_ylabel('out of phase V_rms (V)')
    def filledbins(self):
        print(self.query('SPTS ?'))
    def savelockdata(self,filename):
        np.save(filename,self.lockdata) # saves data as .npy format.
          
def scanmag(oscilloscope,magnet,keithley,timestep,endfield,ioff,qoff,savefile):
    magnet.setfield(endfield)
    magnet.H = np.zeros(0)
    magnet.H2 = np.zeros(0)
    magnet.t = np.zeros(0)
    #ls.temp = np.zeros(0)
    oscilloscope.I = np.zeros(0)
    oscilloscope.Q = np.zeros(0)
    oscilloscope.D = np.zeros(0)
    fig, (fig1,fig2,fig3,fig4) = plt.subplots(4,1,figsize=(6,14))
    fig1.set_xlabel('t (s)')
    fig2.set_xlabel('t (s)')
    fig3.set_xlabel('t (s)')
    fig4.set_xlabel('t (s)')
    fig1.set_ylabel('Receiver Q (V)')
    fig1.set_title('Receiver Q (V)')
    fig2.set_ylabel('Receiver I (V)')
    fig2.set_title('Receiver I (V)')
    fig3.set_ylabel('Detector (V)')
    fig3.set_title('Detector (V)')
    fig4.set_ylabel('H (T)')
    fig4.set_title('Field')
    plott = 0
    #lock.lockdata = np.zeros((2,numberofsteps+5),dtype=float)
    starttime = time.time()
    for i in range(99999999):
    #for i in range(9):
        #print(time.time()-starttime)
        for g in range(10000000):
            if i*timestep > (time.time()-starttime):
                time.sleep(0.001)
            else:
                break
        oscilloscope.write(':MEASUrement:IMMed:SOUrce CH1')
        oscilloscope.write(':MEASUrement:IMMed:TYPe MEan')
        tempQ = float(str(oscilloscope.query(':MEASU:IMM:VAL?')))
        #osci.Q[i] = float(str(osci.query(':MEASU:IMM:VAL?')))
        oscilloscope.Q = np.append(oscilloscope.Q,[tempQ])
        oscilloscope.write(':MEASUrement:IMMed:SOUrce CH2')
        oscilloscope.write(':MEASUrement:IMMed:TYPe MEan') # this mean integrates over the entire waveform, so the integration time is controlled by the horizontal scale on the oscilloscope.  
        tempI = float(str(oscilloscope.query(':MEASU:IMM:VAL?')))
        #osci.I[i] = float(str(osci.query(':MEASU:IMM:VAL?')))
        oscilloscope.I = np.append(oscilloscope.I,[tempI])
        oscilloscope.write(':MEASUrement:IMMed:SOUrce CH3')
        oscilloscope.write(':MEASUrement:IMMed:TYPe MEan')
        tempD = float(str(oscilloscope.query(':MEASU:IMM:VAL?')))
        #osci.Det[i] = float(str(osci.query(':MEASU:IMM:VAL?')))
        oscilloscope.D = np.append(oscilloscope.D,[tempD])
        tempH = float(str(keithley.query('sens:data?')))
        tempH = tempH/(0.00200098*11.078)
        magnet.H = np.append(magnet.H,[tempH])
        tempt = time.time()-starttime
        magnet.t = np.append(magnet.t,[tempt])
        #ls.readsampleT()
        #tempT = ls.T
        #ls.temp = np.append(ls.temp,[tempT])
        if tempt>plott:   
            np.save(savefile +'_H_'+time.strftime("%m%d%Y"),magnet.H)   #field acquired throught the keithly
            #np.save(prefix+'_B_'+time.strftime("%H%M_%m%d%Y"),mag.H2)  #field acquired through the labview program
            np.save(savefile +'_t_'+time.strftime("%m%d%Y"),magnet.t)   #time using time.time in program
            np.save(savefile +'_Q_'+time.strftime("%m%d%Y"),oscilloscope.Q)  #quadrature component from receiver signal using the oscilloscope
            np.save(savefile +'_I_'+time.strftime("%m%d%Y"),oscilloscope.I)  #quadrature component from receiver signal using the oscilloscope
            np.save(savefile +'_D_'+time.strftime("%m%d%Y"),oscilloscope.D)#Detector signal using the oscilloscope
            #np.save(savefile +'_K_'+time.strftime("%m%d%Y"),ls.temp)
            fig1.plot(tempt,tempQ,marker='o',color='blue')
            fig1.figure.canvas.draw()
            fig2.plot(tempt,tempI,marker='o',color='red')
            fig2.figure.canvas.draw()  
            #fig3.plot(tempt,tempD,marker='o',color='black')
            #fig3.figure.canvas.draw()
            fig3.plot(tempH,np.sqrt((tempI-qoff)**2+(tempQ-qoff)**2),marker='o',color='black')   #offsets currently hardcoded in -_-
            fig3.figure.canvas.draw()
            fig4.plot(tempt,tempH,marker='o',color='green')
            fig4.figure.canvas.draw()
            plott = plott+0
        magnet.getfield()
        if np.abs(magnet.fld - endfield)<0.01:
            break
def tempscan(oscilloscope,keithley,ls,ts,Ttimestep,timestep,totaltime,savefile):
    oscilloscope.I = np.zeros(0)
    oscilloscope.Q = np.zeros(0)
    oscilloscope.A = np.zeros(0)
    oscilloscope.B = np.zeros(0)
    oscilloscope.H = np.zeros(0)
    oscilloscope.T = np.zeros(0)
    oscilloscope.s = np.zeros(0)
    fig, (fig1,fig2,fig3,fig4,fig5,fig6) = plt.subplots(6,1,figsize=(6,14))
    fig1.set_xlabel('t (s)')
    fig2.set_xlabel('t (s)')
    fig3.set_xlabel('t (s)')
    fig4.set_xlabel('t (s)')
    fig3.set_xlabel('t (s)')
    fig4.set_xlabel('t (s)')
    fig1.set_ylabel('Receiver Q (V)')
    fig1.set_title('Receiver Q (V)')
    fig2.set_ylabel('Receiver I (V)')
    fig2.set_title('Receiver I (V)')
    fig3.set_ylabel('A (V)')
    fig3.set_title('Channel 3 (V)')
    fig4.set_ylabel('B (V)')
    fig4.set_title('Channel 4 (V)')
    fig5.set_ylabel('H (T)')
    fig5.set_title('Field')
    fig6.set_ylabel('T (K)')
    fig6.set_title('Temperature')
    plott = 0
    #lock.lockdata = np.zeros((2,numberofsteps+5),dtype=float)
    starttime = time.time()
    qq=int(0)
    for i in range(99999999):
    #for i in range(9):
        #print(time.time()-starttime)
        for g in range(10000000):
            if (time.time()-starttime) > qq*Ttimestep:
                ls.setT(ts[int(qq)])
                qq=int(qq+1)
            if (time.time()-starttime) < i*timestep:
                time.sleep(0.001)
            else:
                break
        oscilloscope.write(':MEASUrement:IMMed:SOUrce CH1')
        oscilloscope.write(':MEASUrement:IMMed:TYPe MEan')
        tempQ = float(str(oscilloscope.query(':MEASU:IMM:VAL?')))
        oscilloscope.write(':MEASUrement:IMMed:SOUrce CH2')
        oscilloscope.write(':MEASUrement:IMMed:TYPe MEan') # this mean integrates over the entire waveform, so the integration time is controlled by the horizontal scale on the oscilloscope.  
        tempI = float(str(oscilloscope.query(':MEASU:IMM:VAL?')))
        oscilloscope.write(':MEASUrement:IMMed:SOUrce CH3')
        oscilloscope.write(':MEASUrement:IMMed:TYPe MEan')
        tempA = float(str(oscilloscope.query(':MEASU:IMM:VAL?')))
        oscilloscope.write(':MEASUrement:IMMed:SOUrce CH4')
        oscilloscope.write(':MEASUrement:IMMed:TYPe MEan')
        tempB = float(str(oscilloscope.query(':MEASU:IMM:VAL?')))
        temps = time.time()-starttime
        tempH = float(str(keithley.query('sens:data?')))
        tempH = ((tempH/(0.00200098*11.078))+0.011)*(1/0.948)
        ls.readsampleT()
        tempT = ls.T
        oscilloscope.Q = np.append(oscilloscope.Q,[tempQ])
        oscilloscope.I = np.append(oscilloscope.I,[tempI])
        oscilloscope.A = np.append(oscilloscope.A,[tempA])
        oscilloscope.B = np.append(oscilloscope.B,[tempB])
        oscilloscope.H = np.append(oscilloscope.H,[tempH])
        oscilloscope.T = np.append(oscilloscope.T,[tempT])
        oscilloscope.s = np.append(oscilloscope.s,[temps])
        if temps>plott:   
            np.save(savefile +'_Q_'+time.strftime("%m%d%Y"),oscilloscope.Q)  #quadrature component from receiver signal using the oscilloscope
            np.save(savefile +'_I_'+time.strftime("%m%d%Y"),oscilloscope.I)  #quadrature component from receiver signal using the oscilloscope
            np.save(savefile +'_A_'+time.strftime("%m%d%Y"),oscilloscope.A)#Detector signal using the oscilloscope
            np.save(savefile +'_B_'+time.strftime("%m%d%Y"),oscilloscope.B)#Detector signal
            np.save(savefile +'_H_'+time.strftime("%m%d%Y"),oscilloscope.H)#Detector signal
            np.save(savefile +'_T_'+time.strftime("%m%d%Y"),oscilloscope.T)#Detector signal using the oscilloscope
            np.save(savefile +'_s_'+time.strftime("%m%d%Y"),oscilloscope.s)
            #np.save(savefile +'_K_'+time.strftime("%m%d%Y"),ls.temp)
            fig1.plot(temps,tempQ,marker='o',color='gold')
            fig1.figure.canvas.draw()
            fig2.plot(temps,tempI,marker='o',color='blue')
            fig2.figure.canvas.draw()  
            fig3.plot(temps,tempA,marker='o',color='purple')
            fig3.figure.canvas.draw()
            fig4.plot(temps,tempB,marker='o',color='green')
            fig4.figure.canvas.draw()
            fig5.plot(temps,tempH,marker='o',color='red')
            fig5.figure.canvas.draw()
            fig6.plot(temps,tempT,marker='o',color='black')
            fig6.figure.canvas.draw()
            plott = plott+0
        if (time.time() - starttime)>totaltime:
            break
def longscanH(oscilloscope,keithley,ls,Hsetp,timestep,savefile):
    oscilloscope.I = np.zeros(0)
    oscilloscope.Q = np.zeros(0)
    oscilloscope.A = np.zeros(0)
    oscilloscope.B = np.zeros(0)
    oscilloscope.H = np.zeros(0)
    oscilloscope.T = np.zeros(0)
    oscilloscope.s = np.zeros(0)
    fig, (fig1,fig2,fig3,fig4,fig5,fig6) = plt.subplots(6,1,figsize=(6,14))
    fig1.set_xlabel('t (s)')
    fig2.set_xlabel('t (s)')
    fig3.set_xlabel('t (s)')
    fig4.set_xlabel('t (s)')
    fig3.set_xlabel('t (s)')
    fig4.set_xlabel('t (s)')
    fig1.set_ylabel('Receiver Q (V)')
    fig1.set_title('Receiver Q (V)')
    fig2.set_ylabel('Receiver I (V)')
    fig2.set_title('Receiver I (V)')
    fig3.set_ylabel('A (V)')
    fig3.set_title('Channel 3 (V)')
    fig4.set_ylabel('B (V)')
    fig4.set_title('Channel 4 (V)')
    fig5.set_ylabel('H (T)')
    fig5.set_title('Field')
    fig6.set_ylabel('T (K)')
    fig6.set_title('Temperature')
    plott = 0
    #lock.lockdata = np.zeros((2,numberofsteps+5),dtype=float)
    starttime = time.time()
    for i in range(99999999):
        for g in range(10000000):
            if i*timestep > (time.time()-starttime):
                time.sleep(0.001)
            else:
                break
        oscilloscope.write(':MEASUrement:IMMed:SOUrce CH1')
        oscilloscope.write(':MEASUrement:IMMed:TYPe MEan')
        tempQ = float(str(oscilloscope.query(':MEASU:IMM:VAL?')))
        oscilloscope.write(':MEASUrement:IMMed:SOUrce CH2')
        oscilloscope.write(':MEASUrement:IMMed:TYPe MEan') # this mean integrates over the entire waveform, so the integration time is controlled by the horizontal scale on the oscilloscope.  
        tempI = float(str(oscilloscope.query(':MEASU:IMM:VAL?')))
        oscilloscope.write(':MEASUrement:IMMed:SOUrce CH3')
        oscilloscope.write(':MEASUrement:IMMed:TYPe MEan')
        tempA = float(str(oscilloscope.query(':MEASU:IMM:VAL?')))
        oscilloscope.write(':MEASUrement:IMMed:SOUrce CH4')
        oscilloscope.write(':MEASUrement:IMMed:TYPe MEan')
        tempB = float(str(oscilloscope.query(':MEASU:IMM:VAL?')))
        temps = time.time()-starttime
        tempH = float(str(keithley.query('sens:data?')))
        tempH = ((tempH/(0.00200098*11.078))+0.011)*(1/0.948)
        ls.readsampleT()
        tempT = ls.T
        oscilloscope.Q = np.append(oscilloscope.Q,[tempQ])
        oscilloscope.I = np.append(oscilloscope.I,[tempI])
        oscilloscope.A = np.append(oscilloscope.A,[tempA])
        oscilloscope.B = np.append(oscilloscope.B,[tempB])
        oscilloscope.H = np.append(oscilloscope.H,[tempH])
        oscilloscope.T = np.append(oscilloscope.T,[tempT])
        oscilloscope.s = np.append(oscilloscope.s,[temps])
        if temps>plott:
            try:
                np.save(savefile +'_Q_'+time.strftime("%m%d%Y"),oscilloscope.Q)
            except IOError:
                print('Permission denied to write to Q file')
            try:
                np.save(savefile +'_I_'+time.strftime("%m%d%Y"),oscilloscope.I)
            except IOError:
                print('Permission denied to write to I file')
            try:
                np.save(savefile +'_A_'+time.strftime("%m%d%Y"),oscilloscope.A)
            except IOError:
                print('Permission denied to write to A file')
            try:
                np.save(savefile +'_B_'+time.strftime("%m%d%Y"),oscilloscope.B)
            except IOError:
                print('Permission denied to write to B file')
            try:
                np.save(savefile +'_H_'+time.strftime("%m%d%Y"),oscilloscope.H)
            except IOError:
                print('Permission denied to write to H file')
            try:
                np.save(savefile +'_T_'+time.strftime("%m%d%Y"),oscilloscope.T)
            except IOError:
                print('Permission denied to write to T file')
            try:
                np.save(savefile +'_s_'+time.strftime("%m%d%Y"),oscilloscope.s)
            except IOError:
                print('Permission denied to write to s file')
            #np.save(savefile +'_K_'+time.strftime("%m%d%Y"),ls.temp)
            fig1.plot(temps,tempQ,marker='o',color='gold')
            fig1.figure.canvas.draw()
            fig2.plot(temps,tempI,marker='o',color='blue')
            fig2.figure.canvas.draw()  
            fig3.plot(temps,tempA,marker='o',color='purple')
            fig3.figure.canvas.draw()
            fig4.plot(temps,tempB,marker='o',color='green')
            fig4.figure.canvas.draw()
            fig5.plot(temps,tempH,marker='o',color='red')
            fig5.figure.canvas.draw()
            fig6.plot(temps,tempT,marker='o',color='black')
            fig6.figure.canvas.draw()
            plott = plott+0
        if np.abs(tempH-Hsetp)<0.1:
            time.sleep(10)
            break
def longscan(oscilloscope,keithley,ls,timestep,totaltime,savefile):
    oscilloscope.I = np.zeros(0)
    oscilloscope.Q = np.zeros(0)
    oscilloscope.A = np.zeros(0)
    oscilloscope.B = np.zeros(0)
    oscilloscope.H = np.zeros(0)
    oscilloscope.T = np.zeros(0)
    oscilloscope.s = np.zeros(0)
    fig, (fig1,fig2,fig3,fig4,fig5,fig6) = plt.subplots(6,1,figsize=(6,14))
    fig1.set_xlabel('t (s)')
    fig2.set_xlabel('t (s)')
    fig3.set_xlabel('t (s)')
    fig4.set_xlabel('t (s)')
    fig3.set_xlabel('t (s)')
    fig4.set_xlabel('t (s)')
    fig1.set_ylabel('Receiver Q (V)')
    fig1.set_title('Receiver Q (V)')
    fig2.set_ylabel('Receiver I (V)')
    fig2.set_title('Receiver I (V)')
    fig3.set_ylabel('A (V)')
    fig3.set_title('Channel 3 (V)')
    fig4.set_ylabel('B (V)')
    fig4.set_title('Channel 4 (V)')
    #fig5.set_ylabel('H (T)')
    fig5.set_ylabel('vti T (K)')
    fig5.set_title('Field')
    fig6.set_ylabel('T (K)')
    fig6.set_title('Temperature')
    plott = 0
    #lock.lockdata = np.zeros((2,numberofsteps+5),dtype=float)
    starttime = time.time()
    for i in range(99999999):
    #for i in range(9):
        #print(time.time()-starttime)
        for g in range(10000000):
            if i*timestep > (time.time()-starttime):
                time.sleep(0.001)
            else:
                break
        oscilloscope.write(':MEASUrement:IMMed:SOUrce CH1')
        oscilloscope.write(':MEASUrement:IMMed:TYPe MEan')
        tempQ = float(str(oscilloscope.query(':MEASU:IMM:VAL?')))
        oscilloscope.write(':MEASUrement:IMMed:SOUrce CH2')
        oscilloscope.write(':MEASUrement:IMMed:TYPe MEan') # this mean integrates over the entire waveform, so the integration time is controlled by the horizontal scale on the oscilloscope.  
        tempI = float(str(oscilloscope.query(':MEASU:IMM:VAL?')))
        oscilloscope.write(':MEASUrement:IMMed:SOUrce CH3')
        oscilloscope.write(':MEASUrement:IMMed:TYPe MEan')
        tempA = float(str(oscilloscope.query(':MEASU:IMM:VAL?')))
        oscilloscope.write(':MEASUrement:IMMed:SOUrce CH4')
        oscilloscope.write(':MEASUrement:IMMed:TYPe MEan')
        tempB = float(str(oscilloscope.query(':MEASU:IMM:VAL?')))
        temps = time.time()-starttime
        #tempH = float(str(keithley.query('sens:data?')))
        #tempH = ((tempH/(0.00200098*11.078))+0.011)*(1/0.948)
        ls.readvtiT()
        tempH = ls.vtiT
        ls.readsampleT()
        tempT = ls.T
        oscilloscope.Q = np.append(oscilloscope.Q,[tempQ])
        oscilloscope.I = np.append(oscilloscope.I,[tempI])
        oscilloscope.A = np.append(oscilloscope.A,[tempA])
        oscilloscope.B = np.append(oscilloscope.B,[tempB])
        oscilloscope.H = np.append(oscilloscope.H,[tempH])
        oscilloscope.T = np.append(oscilloscope.T,[tempT])
        oscilloscope.s = np.append(oscilloscope.s,[temps])
        if temps>plott:
            try:
                np.save(savefile +'_Q_'+time.strftime("%m%d%Y"),oscilloscope.Q)
            except IOError:
                print('Permission denied to write to Q file')
            try:
                np.save(savefile +'_I_'+time.strftime("%m%d%Y"),oscilloscope.I)
            except IOError:
                print('Permission denied to write to I file')
            try:
                np.save(savefile +'_A_'+time.strftime("%m%d%Y"),oscilloscope.A)
            except IOError:
                print('Permission denied to write to A file')
            try:
                np.save(savefile +'_B_'+time.strftime("%m%d%Y"),oscilloscope.B)
            except IOError:
                print('Permission denied to write to B file')
            try:
                np.save(savefile +'_H_'+time.strftime("%m%d%Y"),oscilloscope.H)
            except IOError:
                print('Permission denied to write to H file')
            try:
                np.save(savefile +'_T_'+time.strftime("%m%d%Y"),oscilloscope.T)
            except IOError:
                print('Permission denied to write to T file')
            try:
                np.save(savefile +'_s_'+time.strftime("%m%d%Y"),oscilloscope.s)
            except IOError:
                print('Permission denied to write to s file')
            #np.save(savefile +'_K_'+time.strftime("%m%d%Y"),ls.temp)
            fig1.plot(temps,tempQ,marker='o',color='gold')
            fig1.figure.canvas.draw()
            fig2.plot(temps,tempI,marker='o',color='blue')
            fig2.figure.canvas.draw()  
            fig3.plot(temps,tempA,marker='o',color='purple')
            fig3.figure.canvas.draw()
            fig4.plot(temps,tempB,marker='o',color='green')
            fig4.figure.canvas.draw()
            fig5.plot(temps,tempH,marker='o',color='red')
            fig5.figure.canvas.draw()
            fig6.plot(temps,tempT,marker='o',color='black')
            fig6.figure.canvas.draw()
            plott = plott+0
        if (time.time() - starttime)>totaltime:
            break
def longscanjustT(ls,timestep,totaltime,savefile):
    T = np.zeros(0)
    s = np.zeros(0)
#    fig, (fig1) = plt.subplots(1,1,figsize=(6,4))
#    fig1.set_xlabel('t (s)')
#    fig1.set_ylabel('T (K)')
#    fig1.set_title('Temperature')
    plott = 0
    #lock.lockdata = np.zeros((2,numberofsteps+5),dtype=float)
    starttime = time.time()
    for i in range(99999999):
    #for i in range(9):
        #print(time.time()-starttime)
        for g in range(10000000):
            if i*timestep > (time.time()-starttime):
                time.sleep(0.001)
            else:
                break
        temps = time.time()-starttime
        ls.readsampleT()
        tempT = ls.T
        T = np.append(T,[tempT])
        s = np.append(s,[temps])
        if temps>plott:
            try:
                np.save(savefile +'_T_'+time.strftime("%m%d%Y"),T)
            except IOError:
                print('Permission denied to write to T file')
            try:
                np.save(savefile +'_s_'+time.strftime("%m%d%Y"),s)
            except IOError:
                print('Permission denied to write to s file')
#            fig1.plot(temps,tempT,marker='o',color='black')
#            fig1.figure.canvas.draw()
            plott = plott+0
        if (time.time() - starttime)>totaltime:
            break
            
def noisescan(oscilloscope,timestep,totaltime,savefile):
    oscilloscope.I = np.zeros(0)
    oscilloscope.Q = np.zeros(0)
    oscilloscope.A = np.zeros(0)
    oscilloscope.B = np.zeros(0)
    oscilloscope.t = np.zeros(0)
    fig, (fig1,fig2,fig3,fig4) = plt.subplots(4,1,figsize=(6,14))
    fig1.set_xlabel('t (s)')
    fig2.set_xlabel('t (s)')
    fig3.set_xlabel('t (s)')
    fig4.set_xlabel('t (s)')
    fig1.set_ylabel('Receiver Q (V)')
    fig1.set_title('Receiver Q (V)')
    fig2.set_ylabel('Receiver I (V)')
    fig2.set_title('Receiver I (V)')
    fig3.set_ylabel('A (V)')
    fig3.set_title('Channel 3 (V)')
    fig4.set_ylabel('B (V)')
    fig4.set_title('Channel 4 (V)')
    plott = 0
    #lock.lockdata = np.zeros((2,numberofsteps+5),dtype=float)
    starttime = time.time()
    for i in range(99999999):
    #for i in range(9):
        #print(time.time()-starttime)
        for g in range(10000000):
            if i*timestep > (time.time()-starttime):
                time.sleep(0.001)
            else:
                break
        oscilloscope.write(':MEASUrement:IMMed:SOUrce CH1')
        oscilloscope.write(':MEASUrement:IMMed:TYPe MEan')
        tempQ = float(str(oscilloscope.query(':MEASU:IMM:VAL?')))
        oscilloscope.write(':MEASUrement:IMMed:SOUrce CH2')
        oscilloscope.write(':MEASUrement:IMMed:TYPe MEan') # this mean integrates over the entire waveform, so the integration time is controlled by the horizontal scale on the oscilloscope.  
        tempI = float(str(oscilloscope.query(':MEASU:IMM:VAL?')))
        oscilloscope.write(':MEASUrement:IMMed:SOUrce CH3')
        oscilloscope.write(':MEASUrement:IMMed:TYPe MEan')
        tempA = float(str(oscilloscope.query(':MEASU:IMM:VAL?')))
        oscilloscope.write(':MEASUrement:IMMed:SOUrce CH4')
        oscilloscope.write(':MEASUrement:IMMed:TYPe MEan')
        tempB = float(str(oscilloscope.query(':MEASU:IMM:VAL?')))
        tempt = time.time()-starttime
        oscilloscope.Q = np.append(oscilloscope.Q,[tempQ])
        oscilloscope.I = np.append(oscilloscope.I,[tempI])
        oscilloscope.A = np.append(oscilloscope.A,[tempA])
        oscilloscope.B = np.append(oscilloscope.B,[tempB])
        oscilloscope.t = np.append(oscilloscope.t,[tempt])
        if tempt>plott:   
            np.save(savefile +'_Q_'+time.strftime("%m%d%Y"),oscilloscope.Q)  #quadrature component from receiver signal using the oscilloscope
            np.save(savefile +'_I_'+time.strftime("%m%d%Y"),oscilloscope.I)  #quadrature component from receiver signal using the oscilloscope
            np.save(savefile +'_A_'+time.strftime("%m%d%Y"),oscilloscope.A)#Detector signal using the oscilloscope
            np.save(savefile +'_B_'+time.strftime("%m%d%Y"),oscilloscope.B)#Detector signal using the oscilloscope
            np.save(savefile +'_t_'+time.strftime("%m%d%Y"),oscilloscope.t)
            #np.save(savefile +'_K_'+time.strftime("%m%d%Y"),ls.temp)
            fig1.plot(tempt,tempQ,marker='o',color='gold')
            fig1.figure.canvas.draw()
            fig2.plot(tempt,tempI,marker='o',color='blue')
            fig2.figure.canvas.draw()  
            fig3.plot(tempt,tempA,marker='o',color='purple')
            fig3.figure.canvas.draw()
            fig4.plot(tempt,tempB,marker='o',color='green')
            fig4.figure.canvas.draw()
            plott = plott+0
        if (time.time() - starttime)>totaltime:
            break
            
def scanmagwhilepulsing(oscilloscope,magnet,keithley,ls,timestep,endfield,ioff,qoff,savefile):
    magnet.setfield(endfield)
    magnet.H = np.zeros(0)
    magnet.H2 = np.zeros(0)
    magnet.t = np.zeros(0)
    ls.temp = np.zeros(0)
    oscilloscope.I = np.zeros(0)
    oscilloscope.Q = np.zeros(0)
    oscilloscope.D = np.zeros(0)
    fig, (fig1,fig2,fig3,fig4) = plt.subplots(4,1,figsize=(6,14))
    fig1.set_xlabel('t (s)')
    fig2.set_xlabel('t (s)')
    fig3.set_xlabel('t (s)')
    fig4.set_xlabel('t (s)')
    fig1.set_ylabel('Receiver Q (V)')
    fig1.set_title('Receiver Q (V)')
    fig2.set_ylabel('Receiver I (V)')
    fig2.set_title('Receiver I (V)')
    fig3.set_ylabel('Detector (V)')
    fig3.set_title('Detector (V)')
    fig4.set_ylabel('H (T)')
    fig4.set_title('Field')
    plott = 0
    #lock.lockdata = np.zeros((2,numberofsteps+5),dtype=float)
    starttime = time.time()
    for i in range(99999999):
    #for i in range(9):
        #print(time.time()-starttime)
        for g in range(10000000):
            if i*timestep > (time.time()-starttime):
                time.sleep(0.001)
            else:
                break
        oscilloscope.write(':MEASUrement:IMMed:SOUrce CH1')
        oscilloscope.write(':MEASUrement:IMMed:TYPe AMPlitude')
        tempQ = float(str(oscilloscope.query(':MEASU:IMM:VAL?')))
        #osci.Q[i] = float(str(osci.query(':MEASU:IMM:VAL?')))
        oscilloscope.Q = np.append(oscilloscope.Q,[tempQ])
        oscilloscope.write(':MEASUrement:IMMed:SOUrce CH2')
        oscilloscope.write(':MEASUrement:IMMed:TYPe AMPlitude') # this mean integrates over the entire waveform, so the integration time is controlled by the horizontal scale on the oscilloscope.  
        tempI = float(str(oscilloscope.query(':MEASU:IMM:VAL?')))
        #osci.I[i] = float(str(osci.query(':MEASU:IMM:VAL?')))
        oscilloscope.I = np.append(oscilloscope.I,[tempI])
        oscilloscope.write(':MEASUrement:IMMed:SOUrce CH3')
        oscilloscope.write(':MEASUrement:IMMed:TYPe AMPlitude')
        tempD = float(str(oscilloscope.query(':MEASU:IMM:VAL?')))
        #osci.Det[i] = float(str(osci.query(':MEASU:IMM:VAL?')))
        oscilloscope.D = np.append(oscilloscope.D,[tempD])
        tempH = float(str(keithley.query('sens:data?')))
        tempH = tempH/(0.00200098*11.078)
        magnet.H = np.append(magnet.H,[tempH])
        tempt = time.time()-starttime
        magnet.t = np.append(magnet.t,[tempt])
        ls.readsampleT()
        tempT = ls.T
        ls.temp = np.append(ls.temp,[tempT])
        if tempt>plott:   
            np.save(savefile +'_H_'+time.strftime("%m%d%Y"),magnet.H)   #field acquired throught the keithly
            #np.save(prefix+'_B_'+time.strftime("%H%M_%m%d%Y"),mag.H2)  #field acquired through the labview program
            np.save(savefile +'_t_'+time.strftime("%m%d%Y"),magnet.t)   #time using time.time in program
            np.save(savefile +'_Q_'+time.strftime("%m%d%Y"),oscilloscope.Q)  #quadrature component from receiver signal using the oscilloscope
            np.save(savefile +'_I_'+time.strftime("%m%d%Y"),oscilloscope.I)  #quadrature component from receiver signal using the oscilloscope
            np.save(savefile +'_D_'+time.strftime("%m%d%Y"),oscilloscope.D)#Detector signal using the oscilloscope
            np.save(savefile +'_K_'+time.strftime("%m%d%Y"),ls.temp)
            fig1.plot(tempt,tempQ,marker='o',color='blue')
            fig1.figure.canvas.draw()
            fig2.plot(tempt,tempI,marker='o',color='red')
            fig2.figure.canvas.draw()  
            #fig3.plot(tempt,tempD,marker='o',color='black')
            #fig3.figure.canvas.draw()
            fig3.plot(tempH,np.sqrt((tempI-qoff)**2+(tempQ-qoff)**2),marker='o',color='black')   #offsets currently hardcoded in -_-
            fig3.figure.canvas.draw()
            fig4.plot(tempt,tempH,marker='o',color='green')
            fig4.figure.canvas.draw()
            plott = plott+0
        magnet.getfield()
        if np.abs(magnet.fld - endfield)<0.01:
            break
        
def scanuca(oscilloscope,magnet,keithley,uca,timestep,startuca,enduca,stepuca,savefile):  #timestep must be a value that is 
    ucarray = np.arange(startuca,enduca+stepuca,stepuca)
    oscilloscope.SetHorizontalScale(timestep/10)
    magnet.H = np.zeros(0)
    magnet.H2 = np.zeros(0)
    magnet.t = np.zeros(0)
    oscilloscope.I = np.zeros(0)
    oscilloscope.Q = np.zeros(0)
    oscilloscope.D = np.zeros(0)
    fig, (fig1,fig2,fig3,fig4) = plt.subplots(4,1,figsize=(6,14))
    fig1.set_xlabel('t (s)')
    fig2.set_xlabel('t (s)')
    fig3.set_xlabel('t (s)')
    fig4.set_xlabel('t (s)')
    fig1.set_ylabel('Receiver Q (V)')
    fig1.set_title('Receiver Q (V)')
    fig2.set_ylabel('Receiver I (V)')
    fig2.set_title('Receiver I (V)')
    fig3.set_ylabel('Detector (V)')
    fig3.set_title('Detector (V)')
    fig4.set_ylabel('H (T)')
    fig4.set_title('Field')
    plott = 0
    #lock.lockdata = np.zeros((2,numberofsteps+5),dtype=float)
    starttime = time.time()
    for i in range(len(ucarray)):
        uca.setvoltage(ucarray[i])
        time.sleep(timestep)
        oscilloscope.write(':MEASUrement:IMMed:SOUrce CH1')
        oscilloscope.write(':MEASUrement:IMMed:TYPe MEan')
        tempQ = float(str(oscilloscope.query(':MEASU:IMM:VAL?')))
        #osci.Q[i] = float(str(osci.query(':MEASU:IMM:VAL?')))
        oscilloscope.Q = np.append(oscilloscope.Q,[tempQ])
        oscilloscope.write(':MEASUrement:IMMed:SOUrce CH2')
        oscilloscope.write(':MEASUrement:IMMed:TYPe MEan') # this mean integrates over the entire waveform, so the integration time is controlled by the horizontal scale on the oscilloscope.  
        tempI = float(str(oscilloscope.query(':MEASU:IMM:VAL?')))
        #osci.I[i] = float(str(osci.query(':MEASU:IMM:VAL?')))
        oscilloscope.I = np.append(oscilloscope.I,[tempI])
        oscilloscope.write(':MEASUrement:IMMed:SOUrce CH3')
        oscilloscope.write(':MEASUrement:IMMed:TYPe MEan')
        tempD = float(str(oscilloscope.query(':MEASU:IMM:VAL?')))
        #osci.Det[i] = float(str(osci.query(':MEASU:IMM:VAL?')))
        oscilloscope.D = np.append(oscilloscope.D,[tempD])
        tempH = float(str(keithley.query('sens:data?')))
        tempH = tempH/(0.00200098*11.078)
        magnet.H = np.append(magnet.H,[tempH])
        tempt = time.time()-starttime
        magnet.t = np.append(magnet.t,[tempt])
        np.save(savefile +'_P_'+time.strftime("%m%d%Y"),ucarray[:i+1])  
        np.save(savefile +'_H_'+time.strftime("%m%d%Y"),magnet.H)   #field acquired throught the keithly
        #np.save(prefix+'_B_'+time.strftime("%H%M_%m%d%Y"),mag.H2)  #field acquired through the labview program
        np.save(savefile +'_t_'+time.strftime("%m%d%Y"),magnet.t)   #time using time.time in program
        np.save(savefile +'_Q_'+time.strftime("%m%d%Y"),oscilloscope.Q)  #quadrature component from receiver signal using the oscilloscope
        np.save(savefile +'_I_'+time.strftime("%m%d%Y"),oscilloscope.I)  #quadrature component from receiver signal using the oscilloscope
        np.save(savefile +'_D_'+time.strftime("%m%d%Y"),oscilloscope.D)#Detector signal using the oscilloscope
        fig1.plot(tempt,tempQ,marker='o',color='blue')
        fig1.figure.canvas.draw()
        fig2.plot(tempt,tempI,marker='o',color='red')
        fig2.figure.canvas.draw()  
        fig3.plot(tempt,tempD,marker='o',color='black')
        fig3.figure.canvas.draw()
        fig4.plot(tempt,tempH,marker='o',color='green')
        fig4.figure.canvas.draw()
    uca.setvoltage(5)

def savedata(oscilloscope,magnet,prefix):
    #np.save(prefix+'lockdata'+time.strftime("%m%d%Y"),lock.lockdata)
    np.save(prefix + time.strftime("_%H%M")+'_H_'+time.strftime("%m%d%Y"),magnet.H)   #field acquired throught the keithly
    #np.save(prefix+'_B_'+time.strftime("%H%M_%m%d%Y"),mag.H2)  #field acquired through the labview program
    np.save(prefix + time.strftime("_%H%M")+'_t_'+time.strftime("%m%d%Y"),magnet.t)   #time using time.time in program
    np.save(prefix + time.strftime("_%H%M")+'_Q_'+time.strftime("%m%d%Y"),oscilloscope.Q)  #quadrature component from receiver signal using the oscilloscope
    np.save(prefix + time.strftime("_%H%M")+'_I_'+time.strftime("%m%d%Y"),oscilloscope.I)  #quadrature component from receiver signal using the oscilloscope
    np.save(prefix + time.strftime("_%H%M")+'_D_'+time.strftime("%m%d%Y"),oscilloscope.D)#Detector signal using the oscilloscope
# set field setpoint # in tesla
# get field sample
# get field setpoint
# get field status 1 ramp, 0 stop
# ramp rate: 0.27 T/min =0.0045 T/s

def pihalf(oscilloscope,pulsegen,start,end,step,period,filename):
    oscilloscope.write('trig:A:mod normal')
    starttime = time.time()
    #tau = np.logspace(-6,-2,50)+0.1
    pw = np.arange(start,end,step)
    polarity = 'inv'
    #holdoff = 0.15
    #osci.write('trig:A:holdoff:tim %f' % holdoff)
    pulsegen.setperiod(period)
    pulsegen.setchannel(1)
    pulsegen.setpulse(0.1)
    pulsegen.setpolarity(polarity)
    pulsegen.setamp(5.0)
    pulsegen.runchannel()
    print('predicted time for scan = ' + str((2*period+0.2)*datapoints))
    for i in pw:
        bnc.setpulse(i)
        time.sleep(period*2)
        oscilloscope.measureall()
        oscilloscope.savedataI(filename + time.strftime("_%H%M")+'_Q_'+time.strftime("%m%d%Y") + 'delayeq%s_microseconds' % pw)
        oscilloscope.savedataQ(filename + time.strftime("_%H%M")+'_I_'+time.strftime("%m%d%Y") + 'delayeq%s_microseconds' % pw)
        oscilloscope.savedataD(filename + time.strftime("_%H%M")+'_D_'+time.strftime("%m%d%Y") + 'delayeq%s_microseconds' % pw)
        oscilloscope.savedata4(filename + time.strftime("_%H%M")+'_4_'+time.strftime("%m%d%Y") + 'delayeq%s_microseconds' % pw)
    
    timecollect = time.time()-starttime
    print('time to collect = ' + timecollect + ' for %f data points' % datapoints)


def saturationrecovery(osc,bnc,uca,probew,proben,satw,probeperiod,satuca,probeuca,savefile):
    #probew = 0.02
    #probeperiod = 0.1
    #ponn = 9
    #poffn = 11
    osc.write('ACQ:STATE RUN')
    osc.triggermodeauto()
    bnc.stop()
    pol = 'norm'
    
    #satw = probeperiod*(ponn+poffn)/2
    
    bnc.setperiod(probeperiod)
    bnc.setchannel(1)
    bnc.setmux(1)
    bnc.bncmode('norm')
    #bnc.dutycycle(ponn,poffn,poffn)
    bnc.setpulse(probeperiod-probew) #this is because you pulse to turn ~off~ the microwaves. So probew still corresponds to the microwave probe pulse width.
    bnc.setpolarity(pol)
    bnc.setamp(5.0)
    bnc.runchannel()
    
    bnc.setchannel(2)
    bnc.setmux(2)
    bnc.bncmode('norm')
    #bnc.dutycycle(ponn,poffn,poffn)
    bnc.setpulse(probeperiod-probew) #this is because you pulse to turn ~off~ the microwaves. So probew still corresponds to the microwave probe pulse width.
    bnc.setpolarity(pol)
    bnc.setamp(5.0)
    bnc.runchannel()
    
    uca.setvoltage(satuca)
    starttime=time.time()
    
    for i in range(1000000):
        if time.time()-starttime > satw:
            osc.SetHorizontalScale(proben*probeperiod/10)
            time.sleep(0.001) # may not be necessary
            uca.setvoltage(probeuca)
            probetime = time.time()
            time.sleep(0.045)
            bnc.run()
            for i in range(1000000):
                if time.time()-probetime>((proben+2)*probeperiod):
                    osc.write('ACQ:STATE STOP')
                    osc.measureall()
                    osc.savedataI(savefile)
                    osc.savedataQ(savefile)
                    osc.savedataD(savefile)
                    osc.savedata4(savefile)
                    #time.sleep(offtime-(proben+2)*probeperiod)
                    break
                else:
                    time.sleep(0.001)
            break
        else:
            time.sleep(0.001)
    bnc.stop()
    uca.setvoltage(5)
    
def saturationrecovery2(osc,bnc,satw,period,amp,offset,avgnumber,yfactor,savefile): # should turn this into a class
    #pol = 'inv'
    #bnc.setchannel(1)
    #bnc.setpolarity(pol)
    #bnc.setchannel(2)
    #bnc.setpolarity(pol)
    
    osc.write('ACQ:MOD AVE')
    osc.write('ACQ:NUMAV %f' % avgnumber)
    osc.write('ACQ:STATE RUN')
    osc.triggermodenormal()
    #osc.SetHorizontalScale(satw)
    
    #osc.AFG_pulse(period-satw,period,offset,amp)
    #osc.write('AFG:OUTPut:STATE ON')
    
    #pol = 'norm'
    #bnc.setchannel(1)
    #bnc.setpolarity(pol)
    #bnc.setchannel(2)
    #bnc.setpolarity(pol)
    
    time.sleep(avgnumber*period+30)
    
    osc.measureall(yfactor)
    osc.savedataI(savefile)
    osc.savedataQ(savefile)
    osc.savedataD(savefile)
    osc.savedata4(savefile)
#####################################################################
#    bnc.setchannel(2)
#    bnc.setmux(2)
#    bnc.dutycycle(1,ponn,0)
#    bnc.setpulse(satw)
#    bnc.setpolarity(pol)
#    bnc.setamp(5.0)
#    bnc.runchannel()
#    
#    oscval
#    for j in range(100000000):
#        if j == 0:
#            bnc.run()
#        if oscval > 3:
#            uca.setvoltage(probeP)
#        if oscval < 3:
#            uca.setvoltage(ucaP)
#        time.sleep(0.001)
#    
#    # global bnc commands and variables
#    bnc.setperiod(period)
#    polarity = 'inv'
#    
#    bnc.setchannel(1)
#    bnc.setmux(3)
#    bnc.setpolarity(polarity)
#    bnc.setpulse(satw)
#    bnc.setamp(5.0)
#    bnc.runchannel()
#
#    

#    
#
#osci.write('trig:A:mod normal')
#
#starttime = time.time()
#datapoints = 50
#longpulsewidth = 1
#tau = np.logspace(-6,-2,50)+0.1
#
#
#
#
#holdoff = 0.15
#osci.write('trig:A:holdoff:tim %f' % holdoff)
#bnc.setperiod(period)
#bnc.setchannel(1)
#bnc.setpulse(0.1)
#bnc.setpolarity(polarity)
#bnc.setamp(5.0)
#bnc.runchannel()
#bnc.setchannel(2)
#bnc.setpolarity(polarity)
#bnc.setamp(5.0)
#bnc.runchannel()
#
#print('predicted time for scan = ' + str((2*period+0.2)*datapoints))
#
#for i in tau:
#    bnc.setpulse(probepulsewidth,i)
#    
#    time.sleep(0.4)
#    delay = str((i-0.1)*1000000.0)
#    osci.measureall()
#    osci.savedataI('Data/EPR_test_06042018/saturation_recovery/06112018_NENP/test/delayeq%s_microseconds' % delay)
#    osci.savedataQ('Data/EPR_test_06042018/saturation_recovery/06112018_NENP/test/delayeq%s_microseconds' % delay)
#    osci.savedataD('Data/EPR_test_06042018/saturation_recovery/06112018_NENP/test/delayeq%s_microseconds' % delay)
#    osci.savedata4('Data/EPR_test_06042018/saturation_recovery/06112018_NENP/test/delayeq%s_microseconds' % delay)
#
#timecollect = time.time()-starttime
#print('time to collect = ' + timecollect + ' for %f data points' % datapoints)
#    
#    bnc.setchannel(1)
#    bnc.setmux(1)
#    bnc.setchannel(2)
#    bnc.setmux(2)
#
#def inversionrecovery
#
#def spinecho