
def BSM(spot,strike,ttm,vol):
    import numpy as np
    import numpy as np
    import pandas as pd
    #d1=cdf(x, loc=0, scale=1)
    #d2=
    return np.log(spot/strike)

if __name__ == '__main__':
    import sys
    import yfinance as yf
    import datetime as datetime
    import numpy as np
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

    #strike
    #optcalls('strike')


    
#start = dt.date( 2014, 1, 1 )
#end = dt.date( 2014, 1, 16 )

#days = np.busday_count( start, end 
#    print(BSM(spot=spot,))

    print('Completed')
    sys.exit(0)
