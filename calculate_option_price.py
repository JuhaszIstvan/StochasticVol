def MCWiener(PathNum,IterNum,Start,sigma,Annualrfree,T):
    #expects iternumber as number of days
    import numpy as np
    import pandas as pd
    IterNum=int(IterNum)
    resultset=pd.Series([None]*PathNum)
    #rPerStepp=int(Annualrfree/IterNum)
    dt=T/IterNum
    print('Iteration step:{:.3f}'.format(dt))
    for k in range(PathNum):
        X=None
        X=Start
        for l in range(1,IterNum+1):
            X=X+X*Annualrfree*dt+X*np.sqrt(dt)*np.random.normal(loc=0.0, scale=sigma)*sigma
            #z=z+np.random.normal(loc=0.0, scale=sigma)
        resultset[k]=X
        resultset=pd.to_numeric(resultset, errors='coerce') #otherwise it will be an object time that fails for the math functions
        #print('Final for ' + str(k) + ' is ' +str(resultset[k]) )       
    return resultset
def BSM(spot,strike,T,Vol,r,OType):
    import numpy as np
    import numpy as np
    import pandas as pd
    from scipy.stats import norm
    #d1=cdf(x, loc=0, scale=1)
    #d2=
    #T=((ExpDate-today).days)/365
    PVK=strike * np.exp(-1*r*T)
    d1=(np.log((spot/strike))+(r+(Vol**2)/2)*(T))/(Vol*np.sqrt(T))
    d2= d1 - Vol*np.sqrt(T)
    #popoptcall['d1']= (np.log(popoptcall['spot']/popoptcall['strike'])+(r+(Vol**2)/2)*(popoptcall['deltaT']))/(Vol*np.sqrt(popoptcall['deltaT']))
    Tag1=norm.cdf(d1, loc=0, scale=1)
    Tag2=norm.cdf(d2, loc=0, scale=1)  
    CVal= Tag1 * spot - Tag2*PVK
    PVal=PVK-spot+CVal
    if OType == "cl":
        Val=CVal
    elif OType == "pl":
        Val=PVal
    elif OType == "ps":
        Val=PVal
    elif OType == "cs":
        Val=PVal
    return Val
def getYahoo(QUOTE):
    import sys
    import yfinance as yf
    import datetime as datetime
    import numpy as np
    import pandas as pd
    from scipy.stats import norm
    TCKR = yf.Ticker(QUOTE)
    Underlying=TCKR.history(period="1d").iloc[0,]
    spot=Underlying['Close']
    #get the available option chain for the last expiration date
    calldf=None
    for optiondate in TCKR.options:
        tmpdf=TCKR.option_chain(optiondate).calls
        tmpdf['optiondate']=datetime.datetime.strptime(optiondate, '%Y-%m-%d').date()
        if calldf is None:
            calldf=tmpdf
        else:
            calldf=pd.concat([calldf, tmpdf], axis=0)
    calldf
    calldf['spot']=spot
    #ExpDate=TCKR.options[-1]
    
    #opt = TCKR.option_chain(ExpDate)
    #ExpDate=datetime.datetime.strptime(ExpDate, '%Y-%m-%d').date()
    #print('today is:{}'.format(ExpDate))
    #optcalls=opt.calls
    #print (optcalls.shape)
    #print('exptoday:',type(ExpDate))
    #Calls with the highest open interest rates
    #popoptcall=optcalls.sort_values('openInterest',ascending=False).iloc[0,]
    #popoptcall['ExpDate']=ExpDate
    
    calldf['spot']=spot
    Vol=0.365
    today = datetime.date.today()
    print('today:',type(today))
    r=0.13
    calldf['currt']=today
    calldf['deltaT']=(calldf['optiondate']-calldf['currt'])
    calldf['deltaT']=calldf['deltaT'].dt.days
    #popoptcall['PVK']=popoptcall['strike'] * np.exp(-1*r*popoptcall['deltaT'])
    #popoptcall['d1']= (np.log(popoptcall['spot']/popoptcall['strike'])+(r+(Vol**2)/2)*(popoptcall['deltaT']))/(Vol*np.sqrt(popoptcall['deltaT']))
    #popoptcall['d2']= popoptcall['d1'] - Vol*np.sqrt(popoptcall['deltaT'])
    #popoptcall['Tag1']=norm.cdf(popoptcall['d1'], loc=0, scale=1)
    #popoptcall['Tag2']=norm.cdf(popoptcall['d2'], loc=0, scale=1)
    #popoptcall['c']=popoptcall['Tag1']* popoptcall['strike']- popoptcall['Tag2']*popoptcall['PVK']
    return calldf

def fullwiener(spot,Vol,r,Numberofpaths,IterNum,T):
    #Wiener process
    import numpy as np
    import pandas as pd
    from scipy.stats import norm
    import sys
    DriftR=1*np.exp(-1*r*T)
    SPt=MCWiener(PathNum=Numberofpaths,IterNum=IterNum,Start=spot,sigma=Vol,Annualrfree=r,T=T)
    P0Set=pd.Series([spot]*Numberofpaths)
    RetSet=SPt-P0Set
    PVRetSet=RetSet*np.exp(-1*r*T)
    Cset=SPt-strike
    #charting
    import matplotlib.pyplot as plt
    import matplotlib.mlab as mlab
    fig, axs = plt.subplots(3,figsize=(10,10))
    plt.subplots_adjust( hspace=0.4)
    fig.suptitle('MC Asset price at T, discounted Pt, returns')
    (calcmu, calcsigma) = norm.fit(SPt)
    count, bins, ignored = axs[0].hist(SPt, 40, density=True)
    axs[0].hist(SPt, 40, density=True,color='lightseagreen')
    y = norm.pdf( bins, calcmu, calcsigma)
    l = axs[0].plot(bins, y, 'green', linewidth=2)
    axs[0].set_title(r'$\mathrm{Histogram\ of\ Pt:}\ \mu=%.3f,\ \sigma=%.3f$' %(calcmu, calcsigma))
    

    (PVRetmu, PVRetsigma) = norm.fit(PVRetSet)
    print(norm.fit(PVRetSet))
    PVRetcount, PVRetbins, ignored = axs[2].hist(PVRetSet, 40, density=True)
    axs[2].hist(PVRetSet, 40, density=True,color='lightseagreen')
    PVRetSety = norm.pdf( PVRetbins, PVRetmu, PVRetsigma)
    PVRetSetl = axs[2].plot(PVRetbins, PVRetSety, 'black', linewidth=2)
    axs[2].set_title(r'$\mathrm{Histogram\ of\ PVReturns:}\ \mu=%.3f,\ \sigma=%.3f$' %( PVRetmu, PVRetsigma))
    #plt.show()
    #print('MC price:', Cset.mean())
    #strike
    #optcalls('strike')
    print('Wiener Completed')
    return PVRetSet.mean()
def GetIV(c,r,Spot,Strike,deltaT):
    return IV
if __name__ == '__main__':
    import datetime as dt
    import numpy as np
    import sys
    strike = 100
    spot = 100
    r=0.03
    Vol = 0.3
    startdate = dt.date( 2014, 1, 1 )
    enddate = dt.date( 2020, 12, 31)
    days = np.busday_count( startdate, enddate) 
    T= 2.5
    OType = 'cl'
    print('Doing BSM call long:','spot:',spot,'strike',strike,'T-t0;',T,'VolPercent:',Vol)
    print ('BSM is call long is: ', BSM(spot=spot,strike=strike,T=T,Vol=Vol,r=r,OType=OType))
    OType = 'pl'
    #print ('BSM is put long is: ', BSM(spot=spot,strike=strike,T=T,Vol=Vol,r=r,OType=OType))
    print('Doing Wiener for the distribution of St spot prices:','spot:',spot,'strike',strike,'T-t0;',T,'VolPercent:',Vol, 'NumberofPaths:',1000,'IterNum:',360) 

    ReturnMean=fullwiener(r=r,Vol=Vol,spot=spot,Numberofpaths=1000,IterNum=1000,T=T)
    ExactPrice=spot*np.exp(1*r*T)
    print('Exactprice is:{:.3f},Simulated Price is:{:.3f},the difference being:{:.3f}'.format(ExactPrice,ReturnMean,ReturnMean-ExactPrice))    

    #it is time to do the implied volatility hack
    optiondata=getYahoo('AAPL')
    optiondata.head
    print (optiondata['deltaT'])
    GetIV
    sys.exit(0)
