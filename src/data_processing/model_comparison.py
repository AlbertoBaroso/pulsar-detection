# import matplotlib.pyplot
# import numpy
# import scipy.special
# from confusion_matrix import print_confusion_matrix
# from optimal_bayes_decisions import compute_minimum_DCF, compute_dummy_system, optimal_bayes_decisions, compute_confusion_matrix, bayes_risk, fnr_fpr

# configurations = [
#     {"π1": 0.5, "Cfn": 1, "Cfp": 1},
#     {"π1": 0.8, "Cfn": 1, "Cfp": 1},
#     {"π1": 0.5, "Cfn": 10, "Cfp": 1},
#     {"π1": 0.8, "Cfn": 1, "Cfp": 10}
# ]

# llCond = numpy.load("Data/commedia_llr_infpar_eps1.npy")
# cl = numpy.load("Data/commedia_labels_infpar_eps1.npy")
# def evaluate_model(π1, Cfn, Cfp):

#     # llJoint = llCond + numpy.log(numpy.array([[1.0/3.0], [1.0/3.0], [1.0/3.0]]))
#     # llMarginal = scipy.special.logsumexp(llJoint, axis=0)
#     # Post = numpy.exp(llJoint - llMarginal)
#     # predictions = numpy.argmax(Post, axis=0)
    
#     predictions = optimal_bayes_decisions(llCond, π1, Cfn, Cfp)
#     confusion_matrix = compute_confusion_matrix(predictions, cl)
#     print_confusion_matrix(confusion_matrix)
#     # Compute False Negative and False Positive ratios
#     FNR, FPR = fnr_fpr(confusion_matrix)
#     # Compute bayes risk for the model
#     # Un-normalized Detection Cost Function
#     DCF_u = bayes_risk(π1, Cfn, Cfp, FNR, FPR)

#     # Compute bayes risk for dummy models that do not use test data at all
#     B_dummy = compute_dummy_system(π1, Cfn, Cfp)

#     # Compute the normalized DCF
#     DCF_n = DCF_u / B_dummy

#     """
#         We can observe that only in two cases the DCF is lower than 1, in the remaining cases our system is actually harmful.
#     """

#     # Minimum DCF
#     thresholds = [-numpy.Infinity, *llCond, numpy.Infinity]
#     min_DCF = compute_minimum_DCF(thresholds, llCond, cl, π1, Cfn, Cfp)

#     print("(π1, Cfn, Cfp) \t DCF\t\t\tMin DCF")
#     print("({}, {}, {}) \t {}\t {}".format(π1, Cfn, Cfp, DCF_n, min_DCF))


#     # # BAYES ERROR PLOTS #


#     # Consider values of p˜ ranging, for example, from -3 to +3
#     # 21 is the number of points we evaluate the DCF at in the example
#     effPriorLogOdds = numpy.linspace(-3, 3, 21)

#     DCF_n = []
#     min_DCF_n = []

#     for p_tilde in effPriorLogOdds:  # p_tilde is a log-odds
#         π_tilde = 1 / (1 + numpy.e ** p_tilde)

#         # Compute optimal Bayes decisions starting from priors, costs and binary log-likelihood ratios
#         predictions = optimal_bayes_decisions(llCond, π_tilde, Cfn, Cfp)
#         # Compute confusion matrix
#         confusion_matrix = compute_confusion_matrix(predictions, cl)
#         # Compute False Negative and False Positive ratios
#         FNR, FPR = fnr_fpr(confusion_matrix)
#         # Compute bayes risk for the model
#         # Un-normalized Detection Cost Function
#         DCF_u = bayes_risk(π_tilde, Cfn, Cfp, FNR, FPR)

#         # Compute bayes risk for dummy models that do not use test data at all
#         B_dummy = compute_dummy_system(π_tilde, Cfn, Cfp)

#         # Compute the normalized DCF
#         DCF_n.append(DCF_u / B_dummy)

#         # thresholds = [-numpy.Infinity, *infpar_llr, numpy.Infinity]
#         thresholds = llCond
#         min_DCF_n.append(compute_minimum_DCF(thresholds, llCond, cl, π_tilde, Cfn, Cfp))


#     matplotlib.pyplot.plot(effPriorLogOdds, DCF_n, label='DCF', color='r')
#     matplotlib.pyplot.plot(effPriorLogOdds, min_DCF_n, label='min DCF', color='b')

# matplotlib.pyplot.ylim([0, 1.1])
# matplotlib.pyplot.xlim([-3, 3])

# matplotlib.pyplot.xlabel('prior log-odds')
# matplotlib.pyplot.ylabel('DCF value')
# matplotlib.pyplot.show()























# import numpy
# from sklearn.metrics import confusion_matrix
# from confusion_matrix import compute_confusion_matrix, print_confusion_matrix


# def fnr_fpr(confusion_matrix):
#     fn = confusion_matrix[0][1]
#     tp = confusion_matrix[1][1]
#     fp = confusion_matrix[1][0]
#     tn = confusion_matrix[0][0]
#     fpr = fp / (fp + tn)
#     fnr = fn / (fn + tp)
#     return fnr, fpr

# def optimal_bayes_decisions(llrs, π1, Cfn, Cfp):
#     # HT is inferno and HF is paradiso
#     # C = [[0, Cfn], [Cfp, 0]]  # Cost matrix
#     # class_priors = (1 - π1, π1)

#     threshold = - numpy.log(π1 * Cfn / (1 - π1) * Cfp)
#     return (llrs > threshold).astype(int)

# def optimal_bayes_decisions_threshold(llrs, threshold):
#     return (llrs > threshold).astype(int)

# def bayes_risk(π1, Cfn, Cfp, FNR, FPR):
#     return π1 * Cfn * FNR + (1 - π1) * Cfp * FPR

# def compute_dummy_system(π1, Cfn, Cfp):
#     return min(π1 * Cfn, (1 - π1) * Cfp)

# def compute_minimum_DCF(thresholds, llrs, labels, π1, Cfn, Cfp):
#     min_DCF = None
#     thresholds = [-numpy.Infinity, *thresholds, numpy.Infinity]
#     for threshold in thresholds:
#         predictions = optimal_bayes_decisions_threshold(llrs, threshold)
#         confusion_matrix = compute_confusion_matrix(predictions, labels)
#         FNR, FPR = fnr_fpr(confusion_matrix)
#         DCF_u = bayes_risk(π1, Cfn, Cfp, FNR, FPR)
#         B_dummy = compute_dummy_system(π1, Cfn, Cfp)
#         DCF_n = DCF_u / B_dummy
#         min_DCF = DCF_n if min_DCF is None else min(DCF_n, min_DCF)
#     return min_DCF

# if __name__ == '__main__':
    
#     configurations = [
#         { "π1" : 0.5, "Cfn" : 1, "Cfp" : 1 },
#         { "π1" : 0.8, "Cfn" : 1, "Cfp" : 1 },
#         { "π1" : 0.5, "Cfn" : 10, "Cfp" : 1 },
#         { "π1" : 0.8, "Cfn" : 1, "Cfp" : 10 }
#     ]

        
#     infpar_llr = numpy.load("Data/commedia_llr_infpar.npy")
#     labels = numpy.load("Data/commedia_labels_infpar.npy")

#     for configuration in configurations:
        
#         π1 = configuration["π1"]
#         Cfn = configuration["Cfn"]
#         Cfp = configuration["Cfp"]
        
#         # Compute optimal Bayes decisions starting from priors, costs and binary log-likelihood ratios
#         predictions = optimal_bayes_decisions(infpar_llr, π1, Cfn, Cfp)
#         # Compute confusion matrix
#         confusion_matrix = compute_confusion_matrix(predictions, labels)
#         # Compute False Negative and False Positive ratios
#         FNR, FPR = fnr_fpr(confusion_matrix)
#         # Compute bayes risk for the model
#         DCF_u = bayes_risk(π1, Cfn, Cfp, FNR, FPR) # Un-normalized Detection Cost Function
        
#         print_confusion_matrix(confusion_matrix)
#         print("DCF_u: {}".format(DCF_u))

#         # Compute bayes risk for dummy models that do not use test data at all
#         B_dummy = compute_dummy_system(π1, Cfn, Cfp)

#         # Compute the normalized DCF
#         DCF_n = DCF_u / B_dummy
#         print("(π1, Cfn, Cfp) \t DCF")
#         print("({}, {}, {}) \t {}".format(π1, Cfn, Cfp, DCF_n))
        
#         """
#         We can observe that only in two cases the DCF is lower than 1, in the remaining cases our system is actually harmful.
#         """

#         # Minimum DCF 
#         thresholds = [-numpy.Infinity, *infpar_llr, numpy.Infinity]
#         min_DCF = compute_minimum_DCF(thresholds, infpar_llr, labels, π1, Cfn, Cfp)
#         print("Minimum DCF_n: {}".format(min_DCF))

#         """
#         With the except of the first application, we can observe a significant loss due to poor calibration. This
#         loss is even more significant for the two applications which had a normalized DCF larger than 1. In
#         these two scenarios, our classifer is able to provide discriminant scores, but we were not able to employ
#         the scores to make better decisions than those that we would make from the prior alone
#         """
