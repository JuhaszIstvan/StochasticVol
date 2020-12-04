def MCWiener(PathNum,IterNum,Start,sigma,Annualrfree):
    import numpy as np
    import pandas as pd

    resultset=pd.Series([None]*PathNum)
    dt=Annualrfree/360
    print('rfree:{:.10f} dT:{:.4f}'.format(Annualrfree/360,dt))
    for k in range(PathNum):
        z=None
        z=Start

        for l in range(1,IterNum+1):
            z=z+z*Annualrfree*dt+z*np.sqrt(dt)*np.random.normal(loc=0.0, scale=sigma)*sigma
            #z=z+np.random.normal(loc=0.0, scale=sigma)
        resultset[k]=z
        
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
    print(spot/strike)
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

def fullweiner():
    import sys
    import yfinance as yf
    import datetime as datetime
    import numpy as np
    import pandas as pd
    from scipy.stats import norm
    TCKRstr="AAPL"
    #get the financial data for Apple
    TCKR = yf.Ticker(TCKRstr)
    Underlying=TCKR.history(period="1d").iloc[0,]
    spot=Underlying['Close']
    #get the available option chain for the last expiration date
    ExpDate=TCKR.options[-1]
    opt = TCKR.option_chain(ExpDate)
    ExpDate=datetime.datetime.strptime(ExpDate, '%Y-%m-%d').date()
    print(ExpDate)
    optcalls=opt.calls
    print (optcalls.shape)
    print('exptoday:',type(ExpDate))
    popoptcall=optcalls.sort_values('openInterest',ascending=False).iloc[0,]
    popoptcall['spot']=spot
    Vol=0.365
    today = datetime.date.today()
    print('today:',type(today))
    r=0.13
    popoptcall['currt']=today
    popoptcall['ExpDate']=ExpDate
    popoptcall['deltaT']=((ExpDate-today).days)/360
    popoptcall['PVK']=popoptcall['strike'] * np.exp(-1*r*popoptcall['deltaT'])
    popoptcall['d1']= (np.log(popoptcall['spot']/popoptcall['strike'])+(r+(Vol**2)/2)*(popoptcall['deltaT']))/(Vol*np.sqrt(popoptcall['deltaT']))
    popoptcall['d2']= popoptcall['d1'] - Vol*np.sqrt(popoptcall['deltaT'])
    popoptcall['Tag1']=norm.cdf(popoptcall['d1'], loc=0, scale=1)
    popoptcall['Tag2']=norm.cdf(popoptcall['d2'], loc=0, scale=1)
    popoptcall['c']=popoptcall['Tag1']* popoptcall['strike']- popoptcall['Tag2']*popoptcall['PVK']
    print(popoptcall)
    #strike=popoptcall['strike']



#Wiener process
    IterNum= 250
    Numberofpaths=10000
    DriftR=1*np.exp(-1*r*T)

    SPt=MCWiener(PathNum=Numberofpaths,IterNum=IterNum,Start=spot,sigma=Vol,Annualrfree=r)
    #Cset=max((SPt-strike)*np.exp(-1*r*T),0)
    
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
    
    
    plt.show()

    #print('MC price:', Cset.mean())





    #strike
    #optcalls('strike')


    
#start = dt.date( 2014, 1, 1 )
#end = dt.date( 2014, 1, 16 )

#days = np.busday_count( start, end 
#    print(BSM(spot=spot,))

    print('Completed')
    sys.exit(0)
if __name__ == '__main__':
    strike = 100
    spot = 200
    r=0.03
    Vol = 0.3
    T= 2.5
    OType = 'cl'
    print ('BSM is call long is: ', BSM(spot=spot,strike=strike,T=T,Vol=Vol,r=r,OType=OType))
    OType = 'pl'
    print ('BSM is put long is: ', BSM(spot=spot,strike=strike,T=T,Vol=Vol,r=r,OType=OType))