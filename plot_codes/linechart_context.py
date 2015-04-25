"""
Bar chart demo with pairs of bars grouped for easy comparison.
"""
import numpy as np
import matplotlib.pyplot as plt


x = (5,6,7,8,9,10)
y1 = (88.64,88.69,89.02,88.74,89.02,88.98)
y2 = (90.90,90.96,91.19,90.99,91.17,91.15)

fig, ax = plt.subplots()
#index = np.arange(n_groups)
bar_width = 0.25

opacity = 0.4
error_config = {'ecolor': '0.3'}

plt.plot(x, y1, marker='o', linestyle='--', color='r', label='MP3')
plt.plot(x, y2, marker='x', linestyle='-', color='b', label='Watches')

plt.xlabel('Context Size',fontsize=18)
plt.ylabel('Accuracy(%)',fontsize=18)
#plt.title('Accuracies of Different Classifiers with Average Word Vectors(IMDB)')
#plt.xticks(index + bar_width/2, ('Random\nForest', 'SVM', 'Naive\nBayes', 'Logistic\nRegression', 'k-NN'))
plt.legend()
ax = fig.add_subplot(111)
ax.set_ylim(85,95)
#for i,j in zip(index,means_men):
#    ax.annotate(str(j),xy=(i,j+0.5))

plt.tight_layout()
plt.show()
