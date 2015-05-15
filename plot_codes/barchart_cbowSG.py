"""
Bar chart demo with pairs of bars grouped for easy comparison.
"""
import numpy as np
import matplotlib.pyplot as plt


n_groups = 2

sg = (88.98,91.15)
cbow = (88.39,90.69)

fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.25

opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = plt.bar(index, sg, bar_width,alpha=opacity,color='r',label="SkipGram")
rects2 = plt.bar(index+bar_width, cbow, bar_width,alpha=opacity,color='b',label="CBOW")

plt.xlabel('Distributed Semantic Models',fontsize=18)
plt.ylabel('Accuracy(%)',fontsize=18)
#plt.title('Accuracies of Different Classifiers with Average Word Vectors(IMDB)')
plt.xticks(index + bar_width, ('MP3', 'Watches', ))
plt.legend()
ax = fig.add_subplot(111)
ax.set_ylim(85,95)
for i,j in zip(index,sg):
    ax.annotate(str(j),xy=(i+0.05,j+0.3))
for i,j in zip(index,cbow):
    ax.annotate(str(j),xy=(i+0.35,j+0.3))

plt.tight_layout()
plt.show()
