# import matplotlib.pyplot
# import numpy
# from sklearn.metrics import confusion_matrix
# from optimal_bayes_decisions import compute_minimum_DCF, compute_dummy_system, optimal_bayes_decisions, compute_confusion_matrix, bayes_risk, fnr_fpr


# Cfn = 1
# Cfp = 1


# infpar_llr = numpy.load("Data/commedia_llr_infpar.npy")
# labels = numpy.load("Data/commedia_labels_infpar.npy")


# # Consider values of p˜ ranging, for example, from -3 to +3
# # 21 is the number of points we evaluate the DCF at in the example
# effPriorLogOdds = numpy.linspace(-3, 3, 21)

# DCF_n = []
# min_DCF_n = []
# # for each value of p˜ compute the corresponding effective prior
# for p_tilde in effPriorLogOdds:  # p_tilde is a log-odds
#     π_tilde = 1 / (1 + numpy.e ** p_tilde)

#     # Compute optimal Bayes decisions starting from priors, costs and binary log-likelihood ratios
#     predictions = optimal_bayes_decisions(infpar_llr, π_tilde, Cfn, Cfp)
#     # Compute confusion matrix
#     confusion_matrix = compute_confusion_matrix(predictions, labels)
#     # Compute False Negative and False Positive ratios
#     FNR, FPR = fnr_fpr(confusion_matrix)
#     # Compute bayes risk for the model
#     # Un-normalized Detection Cost Function
#     DCF_u = bayes_risk(π_tilde, Cfn, Cfp, FNR, FPR)

#     # Compute bayes risk for dummy models that do not use test data at all
#     B_dummy = compute_dummy_system(π_tilde, Cfn, Cfp)

#     # Compute the normalized DCF
#     DCF_n.append(DCF_u / B_dummy)

#     thresholds = [-numpy.Infinity, *infpar_llr, numpy.Infinity]
#     min_DCF_n.append(compute_minimum_DCF(thresholds, infpar_llr, labels, π_tilde, Cfn, Cfp))


# matplotlib.pyplot.plot(effPriorLogOdds, DCF_n, label='DCF', color='r')
# matplotlib.pyplot.plot(effPriorLogOdds, min_DCF_n, label='min DCF', color='b')
# matplotlib.pyplot.ylim([0, 1.1])
# matplotlib.pyplot.xlim([-3, 3])

# matplotlib.pyplot.xlabel('prior log-odds')
# matplotlib.pyplot.ylabel('DCF value')
# matplotlib.pyplot.show()