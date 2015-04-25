"""
Bar chart demo with pairs of bars grouped for easy comparison.
"""
import numpy as np
import matplotlib.pyplot as plt


n_groups = 5

means_men = (84.14, 88.42, 75.95, 86.90, 76.76)
std_men = (2, 3, 4, 1, 2)

fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.25

opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = plt.bar(index, means_men, bar_width,alpha=opacity,color='r')

plt.xlabel('Classifiers',fontsize=18)
plt.ylabel('Accuracy(%)',fontsize=18)
#plt.title('Accuracies of Different Classifiers with Average Word Vectors(IMDB)')
plt.xticks(index + bar_width/2, ('Random\nForest', 'SVM', 'Naive\nBayes', 'Logistic\nRegression', 'k-NN'))
plt.legend()
ax = fig.add_subplot(111)
ax.set_ylim(0,100)
for i,j in zip(index,means_men):
    ax.annotate(str(j),xy=(i,j+0.5))

plt.tight_layout()
plt.show()
