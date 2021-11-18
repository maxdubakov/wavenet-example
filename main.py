import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

aapl_raw = pd.read_csv('./data/AAPL.csv')
msft_raw = pd.read_csv('./data/MSFT.csv')
sp500_raw = pd.read_csv('./data/^GSPC.csv')
nsdq_raw = pd.read_csv('./data/^IXIC.csv')

print(aapl_raw.info())
print(aapl_raw.head())

plt.show()