# NormEnhancer


## Result 1, comparison of ensemble-style and boosting-style with simple ensemble and boosting only
|     |       |  ETTh1  |  ETTh2  |  ETTm1  |  ETTm2  |  Exchange  |  WTH  |  Wins  | Norm Wins |
|:---:|:-----:|:-------:|:-------:|:-------:|:-------:|:----------:|:-----:|:------:|:--------:|
| GRU        | +Ensemble     |   13.937   |   27.020   |   10.900   |   15.032   |   1.55e-2   |   895.4   |   0   | **3** |
|            | +NormEnsemble |   15.076   |   26.966   |   10.921   |   13.378   |   1.61e-3   |   869.3   |   0   | **3** |
|            | +Boosting     |   14.843   |   24.646   |   11.190   |   15.115   |   1.51e-2   |   735.6   |   0   |   0   |
|            | +NormBoosting | **11.383** | **22.058** | **8.367**  | **9.083**  | **1.27e-3** | **722.1** | **6** | **6** |
| TCN        | +Ensemble     |   38.076   |   83.111   |   37.410   |   83.092   |   3.00e-2   |   985.2   |   0   |   0   |
|            | +NormEnsemble |   14.811   |   59.808   |   13.522   |   15.775   |   2.58e-2   |   829.9   |   0   | **6** |
|            | +Boosting     |   37.731   |   73.283   |   22.088   |   46.229   |   1.82e-2   |   934.8   |   0   |   0   |
|            | +NormBoosting | **11.582** | **21.615** | **8.214**  | **10.477** | **9.38e-3** | **754.4** | **6** | **6** |
| Autoformer | +Ensemble     | **12.995** |   15.403   |   16.521   | **11.264** | **9.40e-4** | **768.8** | **4** | **4** |
|            | +NormEnsemble |   13.179   | **15.364** |   13.553   |   11.580   |   1.07e-3   |   776.5   |   1   |   2   |
|            | +Boosting     |   13.450   |   17.404   |   19.289   |   12.210   |   9.99e-4   |   788.7   |   0   |   1   |
|            | +NormBoosting |   13.494   |   16.120   | **11.525** |   11.896   |   9.56e-4   |   784.5   |   1   | **5** |
| LightTS    | +Ensemble     |   8.819    |   12.268   |   7.110    |   7.546    |   4.90e-4   |   744.5   |   0   |   2   |
|            | +NormEnsemble |   9.143    | **11.820** | **6.941**  |   7.517    | **4.83e-4** |   784.7   | **3** | **4** |
|            | +Boosting     |   8.922    |   12.463   |   7.280    | **7.448**  |   8.28e-4   |   738.3   |   1   |   2   |
|            | +NormBoosting | **8.763**  |   12.164   |   7.306    |   7.660    |   5.65e-4   | **736.9** |   2   | **4** |
| SCINet     | +Ensemble     |   9.934    |   14.147   |   7.420    |   7.967    |   1.18e-3   |   729.7   |   0   |   2   |
|            | +NormEnsemble |   9.921    |   12.687   |   7.346    |   7.994    | **9.02e-4** |   765.6   |   1   | **4** |
|            | +Boosting     |   9.984    |   14.232   |   7.550    |   8.041    |   1.20e-3   |   725.1   |   0   |   0   |
|            | +NormBoosting | **9.287**  | **12.042** | **7.047**  | **7.532**  |   9.34e-4   | **724.1** | **5** | **6** |
| DLinear    | +Ensemble     |   9.078    |   12.216   |   7.080    |   7.904    |   4.56e-4   |   748.3   |   1   | **5** |
|            | +NormEnsemble |   9.240    |   11.998   |   7.103    |   7.925    |   4.66e-4   |   795.9   |   0   |   1   |
|            | +Boosting     | **9.077**  |   12.143   |   7.063    |   7.813    | **4.52e-4** |   748.2   |   2   |   2   |
|            | +NormBoosting |   9.092    | **11.662** | **7.040**  | **7.775**  |   5.11e-4   | **747.5** | **4** | **4** |



## Time series data properties.
| Dataset| ETTh1 [link](https://github.com/zhouhaoyi/ETDataset)| ETTh2 [link](https://github.com/zhouhaoyi/ETDataset) | ETTm1 [link](https://github.com/zhouhaoyi/ETDataset) | ETTm2 [link](https://github.com/zhouhaoyi/ETDataset) | Exchange [link](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) | WTH [link](https://drive.google.com/drive/folders/1ohGYWWohJlOlb2gsGTeEq3Wii2egnEPR) | ECL [link](https://drive.google.com/drive/folders/1ohGYWWohJlOlb2gsGTeEq3Wii2egnEPR)  | Traffic [link](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) | FA [link](https://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_arff.zip) | PE [link](https://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_arff.zip)  | HA [link](https://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_arff.zip) | PH [link](https://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_arff.zip)  | IN [link](https://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_arff.zip) |
| ------ | ----- | ----- | ----- | ------| -------  | ---  | -----| -------  | --  | --  | --  | ---  | --  | 
|sequence| 17420 | 17420 | 69680 | 69680 | 7588     | 35064| 26304| 17544    | 9414| 440 | 1000| 6668 | 50000|
|timesteps | -     | -     | -     | -     | -        | -     | -     | -        | 62   | 144  | 152  | 217  | 22 |
|channels  | 7     | 7     | 7     | 7     | 8        | 12    | 321   | 862      | 144  | 963  | 3    | 11   | 200|
|classes   | -     | -     | -     | -     | -        | -     | -     | -        | 2    | 7    | 26   | 39   | 10 |   
|metric    | MSE   | MSE   | MSE   | MSE   | MSE      | MSE   | MSE   | MSE      | AUC  | MAUC | MAUC | MAUC | MAUC|




## Tabular data properties. All the tabular data can be down load at [link](https://www.dropbox.com/s/dl/o53umyg6mn3zhxy/data.tar.gz).
|Dataset| YE     | MI  | YA     | CA    | AD    | HI    | EP     | HE    | CO     | JA   |
|------ |------ |------ |------ |------ |------ |------ |------ |------ |------ |------ |
|objects| 515345 | 1200192 | 709877 | 20640 | 48842 | 98050 | 500000 | 65196 | 581012 | 83733 |
|num_features | 90     | 136     |699    | 8    | 6     |28    | 2000   | 27    | 54     | 54    |
|cat_features | 0      |0        | 0     | 0    | 8     | 0    | 0      | 0     | 0      | 0     |
|classes      | -      | -       | -     | -    | 2     | 2    | 2      | 100   | 7      | 4     |
| metric      | MSE    | MSE     | MSE   | MSE   | AUC  | AUC  | AUC    | MAUC  | MAUC   | MAUC  |



## NormEnhancer's performance
![image](insight1.png)
