Ridge
# model = Ridge(fit_intercept=False).fit(XX[train], yy[train])~
Offset: 1 1855631.1639536477
Offset: 7  3169821.667225515
Offset: 14 4576981.248680272
Offset: 21 6044197.956032258
Offset: 28 7799311.529896245
Offset: 35 8693059.509034557
Offset: 42 9346034.603931392
Offset: 49 9285860.099139735

MLPR
# model = MLPRegressor(hidden_layer_sizes=(5), alpha=1.0/5).fit(XX[train], yy[train])
Offset: 1 3368955.1283652848
Offset: 7 4073554.406192709
Offset: 14 5258908.354695859
Offset: 21 7678167.149930634
Offset: 28 9701236.311778653
Offset: 35 11607746.610820813
Offset: 42 17233388.365235668
Offset: 49 31983340.387041796

linearscv
#model = LinearSVC(C=1000).fit(XX[train], yy[train])
Offset: 1 23124589.014084507
Offset: 7 23616296.09352518
Offset: 14 17259225.686346862
Offset: 21 16949902.821969695
Offset: 28 23412295.813229572
Offset: 35 19585744.176
Offset: 42 18187122.18106996
Offset: 49 15040460.038135594

logistic regression for the meme
Offset: 1 19776695.58802817
Offset: 7 12938700.377697842
Offset: 14 15453376.944649447
Offset: 21 13845169.969696969
Offset: 28 7985784.116731518
Offset: 35 15577474.644
Offset: 42 9732237.946502058
Offset: 49 8617072.288135594

Offset:
0.001
3150997.787254474
Offset:
0.01
3152245.033730522
Offset:
0.1
3154330.29284908
Offset:
0
3151249.108453179
Offset:
1
3145805.265975508
Offset:
10
3151406.5190982106
Offset:
100
3149227.6626955075



ridge 7 days
Alpha: 0.001
0.005860489075921309
Alpha: 0.01
0.005815355089631113
Alpha: 0.1
0.005636648929093105
Alpha: 0
0.005872833625395104
Alpha: 1
0.0048067479284997
Alpha: 10
0.005510777186571849
Alpha: 100
0.02315397226462382
mean: 0.04230894845900914
median: 0.0448658311838819
yesterday 0.007774357397225851

ridge offsets
Time Offset: 0
0.004839419942867931
Time Offset: 1
0.003455834806255273
Time Offset: 7
0.005872833625395104
Time Offset: 14
0.006375169860948685
Time Offset: 21
0.007982531534814735
Time Offset: 28
0.012580712007936479

ridge 1 day
Alpha: 0.001
0.0034548949478184236
Alpha: 0.01
0.0034503882295983898
Alpha: 0.1
0.003423339122031852
Alpha: 0
0.003455834806255273
Alpha: 1
0.0033128212738462682
Alpha: 10
0.004620993689149205
Alpha: 100
0.023533682919918184

mlpr m 5 c 1/5
Time Offset: 0
0.014011023769305702
Time Offset: 1
0.011734589847525614
Time Offset: 7
0.0110069875838079
Time Offset: 14
0.014442425403847513
Time Offset: 21
0.021228867905959403
Time Offset: 28
0.04017046438547362

mlpr c 1/5 o 7
Hidden Layer Size: 5
0.018777334987281496
/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/sklearn/utils/validation.py:72: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  return f(**kwargs)
Hidden Layer Size: 6
0.007205343595017356
/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/sklearn/utils/validation.py:72: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  return f(**kwargs)
Hidden Layer Size: 7
0.01064775763733516
/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/sklearn/utils/validation.py:72: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  return f(**kwargs)
/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/sklearn/neural_network/_multilayer_perceptron.py:585: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
  % self.max_iter, ConvergenceWarning)
Hidden Layer Size: 8
0.02658296725377667
/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/sklearn/utils/validation.py:72: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  return f(**kwargs)
Hidden Layer Size: 9
0.014806352725988715
/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/sklearn/utils/validation.py:72: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  return f(**kwargs)
Hidden Layer Size: 10
0.006227143275365649
/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/sklearn/utils/validation.py:72: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  return f(**kwargs)
Hidden Layer Size: 11
0.004640698804599842
/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/sklearn/utils/validation.py:72: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  return f(**kwargs)
Hidden Layer Size: 12
0.014831079467742335
/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/sklearn/utils/validation.py:72: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  return f(**kwargs)
Hidden Layer Size: 13
0.006605537011452484
/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/sklearn/utils/validation.py:72: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  return f(**kwargs)
Hidden Layer Size: 14
0.009400760829595938


mlpr o = 7 hl = 11
Alpha: 0.001
0.008855302461531224
Alpha: 0.01
0.009968317939197922
Alpha: 0.1
0.008001827977908693
Alpha: 0
0.02831715292046459
Alpha: 1
0.004958129618793328
Alpha: 10
0.023122621057643317
Alpha: 100
0.042051316247638054

lasso alpha = 0
Time Offset: 0
0.004845985664557441
Time Offset: 1
0.0034674231831959
Time Offset: 7
0.005977173445314658
Time Offset: 14
0.006866820246611171
Time Offset: 21
0.009445555368540613
Time Offset: 28
0.015728533620663482  
 lasso o = 7
Alpha: 0.001
0.003541120847583207
Alpha: 0.01
0.006701323411257358
Alpha: 0.1
0.038974626197811356
Alpha: 0
0.0034674231831959
Alpha: 1
0.038974626197811356