#https://pythonforfinance.net/2016/11/28/monte-carlo-simulation-in-python/
def MCWiener(PathNum,IterNum,Start,sigma,Annualrfree):
    import numpy as np
    import pandas as pd

    resultset=pd.Series([None]*PathNum)
    dt=1/360
    print('rfree:{:.10f} dT:{:.4f}'.format(Annualrfree/360,dt))
    for k in range(PathNum):
        z=None
        z=Start

        for l  in range(1,IterNum+1):
            z=z+z*Annualrfree*dt+z*np.sqrt(dt)*np.random.normal(loc=0.0, scale=sigma)*sigma
            #z=z+np.random.normal(loc=0.0, scale=sigma)
        resultset[k]=z
        
        resultset=pd.to_numeric(resultset, errors='coerce') #otherwise it will be an object time that fails for the math functions
        #print('Final for ' + str(k) + ' is ' +str(resultset[k]) )       
    return resultset

if __name__ == '__main__':
    import sys
    from scipy.stats import norm
    from scipy.stats import lognorm
    import os, os.path
    import matplotlib.pyplot as plt
    import matplotlib.mlab as mlab
    import numpy as np
    import pandas_datareader as dr
    import pandas as pd
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
    #import functions2

    sigma=0.1
    P0=10
    Annualrfree = 0.02
    IterNum = 360
    Numberofpaths=3000
    #print(BSM(mu,1.2,180,sigma))


    SPt=MCWiener(PathNum=Numberofpaths,IterNum=IterNum,Start=P0,sigma=sigma,Annualrfree=Annualrfree)

    print('Received series data length: {}'.format(len(SPt)))
    print('Received series data mean: {:.2f}'.format(SPt.mean()))
    print('Received series data std : {:.2f}'.format(SPt.std()))

    #charting
    fig, axs = plt.subplots(3,figsize=(10,10))

    plt.subplots_adjust( hspace=0.4)
    fig.suptitle('MC Asset price at T, discounted Pt, returns')


    (calcmu, calcsigma) = norm.fit(SPt)
    count, bins, ignored = axs[0].hist(SPt, 40, density=True)
    axs[0].hist(SPt, 40, density=True,color='lightseagreen')
    y = norm.pdf( bins, calcmu, calcsigma)
    l = axs[0].plot(bins, y, 'green', linewidth=2)
    axs[0].set_title(r'$\mathrm{Histogram\ of\ Pt:}\ \mu=%.3f,\ \sigma=%.3f$' %(calcmu, calcsigma))
    #plt.title(r'$\mathrm{Histogram\ of\ MC\ St\ No_paths=%\ \mu=%.3f\ \sigma=%.3f$' %(Numberofpaths,calcmu, calcsigma))
    
    #SPtnv
    SPtnv=SPt*np.exp(-1*Annualrfree*(IterNum/360))
    (SPtnvmu, SPtnvsigma) = norm.fit(SPtnv)
    SPtnvcount, SPtnvbins, ignored = axs[1].hist(SPtnv, 40, density=True)
    axs[1].hist(SPtnv, 40, density=True,color='lightseagreen')
    SPtnvy = norm.pdf( SPtnvbins, SPtnvmu, SPtnvsigma)
    SPtnvl = axs[1].plot(SPtnvbins, SPtnvy, 'black', linewidth=2)
    axs[1].set_title(r'$\mathrm{Histogram\ of\ Ptnv:}\ \mu=%.3f,\ \sigma=%.3f$' %( SPtnvmu, SPtnvsigma))
  
    
    P0Set=pd.Series([P0]*Numberofpaths)
    RetSet=SPtnv-P0Set
    (Retmu, Retsigma) = norm.fit(RetSet)
    print(norm.fit(RetSet))
    Retcount, Retbins, ignored = axs[2].hist(RetSet, 40, density=True)
    axs[2].hist(RetSet, 40, density=True,color='lightseagreen')
    RetSety = norm.pdf( Retbins, Retmu, Retsigma)
    RetSetl = axs[2].plot(Retbins, RetSety, 'black', linewidth=2)
    axs[2].set_title(r'$\mathrm{Histogram\ of\ Returns:}\ \mu=%.3f,\ \sigma=%.3f$' %( Retmu, Retsigma))
    
    K=10.1 #strike price
    Kset=pd.Series([K]*Numberofpaths)
    cset=np.max(SPtnv-Kset,0)
    print(cset.mean())


    #Call option price
    plt.show()

    #plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
    #plt.show()

    sys.exit(0)
