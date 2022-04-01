

import warnings
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt


def Print_Results_TS_Model(Dataset, order_arima , steps = 5, Nome_Dataset = 'Dados', n = 10, interval = True):

    """
    Função Para se Ajustar um modelo ARIMA de Ordem (p,d,q) em um DataFrame de Pandas,
    dispor o sumário dos resultados e visualizar a imediata previsão de resultados. 

    Dataset: DataFrame de Pandas Contendo os Dados desejados (Index em DateTime e Coluna com Valores)
    
    order_arima: (p,d,q) = (Ordem de AR, Ordem de Diff, Ordem de MA) = (AR,I,MA)

    steps: Número de Períodos para Estimar

    Nome_Dataset: Nome que deseja-se dar aos Dados (Padrão é 'Dados')

    n: Número deperíodos da série original que serão plotados (Padrão n = 10)

    interval: Colocar ou não os intervalos de confiança de 95% no gráfico (Padrão = True)
    """
    
    model = SARIMAX(Dataset, order = order_arima)
    results = model.fit()

    print(results.summary())

    forecast = results.get_forecast(steps = steps)
    Mean = forecast.predicted_mean
    Conf = forecast.conf_int()
    Lower = Conf.iloc[:,0]
    Upper = Conf.iloc[:,1]

    plt.plot(Dataset.iloc[len(Dataset.index) - n :len(Dataset.index)])
    plt.plot(Mean, color = 'black')
    plt.grid()
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.xlabel('Datas',fontdict={'size':12})
    if interval == True: 
        plt.fill_between(Mean.index,Lower,Upper, color = 'grey', alpha = .25)
    plt.title('Forecast de ' + str(steps) + ' períodos de ' + Nome_Dataset, fontdict = {'size':17})
    plt.show()



def Avaliar_Ordem_ARIMA(Dataset, AR_Max = 5, MA_Max = 5, Diff_Max = 5):

    Avaliar_ARIMA = []

    warnings.filterwarnings('ignore')

    # Ordem de AR 
    for p in range(AR_Max+1):
        # Ordem de Diff
        for i in range(Diff_Max+1):
            # Ordem de MA
            for q in range(MA_Max+1):
                try:
                    model = SARIMAX(Dataset,order = (p,i,q), method_kwargs={"UserWarning": False})
                    result = model.fit()

                    Avaliar_ARIMA.append((p,i,q, result.aic, result.bic))
                
                except:
                        Avaliar_ARIMA.append((p,i,q,None,None))

    Avaliar_ARIMA_df = pd.DataFrame(Avaliar_ARIMA, columns = ['P','I','Q','AIC','BIC'])

    warnings.resetwarnings()

    return Avaliar_ARIMA_df



def Print_Results_SARIMAX_Model(Dataset, order_arima, order_sarima , steps = 5, Nome_Dataset = 'Dados', n = 10, interval = True):

    """
    Função Para se Ajustar um modelo ARIMA de Ordem (p,d,q) em um DataFrame de Pandas,
    dispor o sumário dos resultados e visualizar a imediata previsão de resultados. 

    Dataset: DataFrame de Pandas Contendo os Dados desejados (Index em DateTime e Coluna com Valores)
    
    order_arima: (p,d,q) = (Ordem de AR, Ordem de Diff, Ordem de MA) = (AR,I,MA)

    steps: Número de Períodos para Estimar

    Nome_Dataset: Nome que deseja-se dar aos Dados (Padrão é 'Dados')

    n: Número deperíodos da série original que serão plotados (Padrão n = 10)

    interval: Colocar ou não os intervalos de confiança de 95% no gráfico (Padrão = True)
    """
    
    model = SARIMAX(Dataset, order = order_arima, seasonal_order=order_sarima)
    results = model.fit()

    print(results.summary())

    forecast = results.get_forecast(steps = steps)
    Mean = forecast.predicted_mean
    Conf = forecast.conf_int()
    Lower = Conf.iloc[:,0]
    Upper = Conf.iloc[:,1]

    plt.plot(Dataset.iloc[len(Dataset.index) - n :len(Dataset.index)])
    plt.plot(Mean, color = 'black')
    plt.grid()
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.xlabel('Datas',fontdict={'size':12})
    if interval == True: 
        plt.fill_between(Mean.index,Lower,Upper, color = 'grey', alpha = .25)
    plt.title('Forecast de ' + str(steps) + ' períodos de ' + Nome_Dataset, fontdict = {'size':17})
    plt.show()

    return results

