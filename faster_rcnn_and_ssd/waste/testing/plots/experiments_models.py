import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.cm as cm

exp1 = pd.read_csv(r'C:\Users\Zohra\MScProject\faster_rcnn_and_ssd\waste\testing\plots\experiment1.csv')
exp2 = pd.read_csv(r'C:\Users\Zohra\MScProject\faster_rcnn_and_ssd\waste\testing\plots\experiment2.csv')
collage = pd.read_csv(r'C:\Users\Zohra\MScProject\faster_rcnn_and_ssd\waste\testing\plots\collage.csv')

models = ['Faster_RCNN', 'SSD']
predictions = ['Bad Predictions', 'Good Predictions', 'Excellent Predictions']
df = pd.DataFrame(collage)

fig = plt.figure(figsize=(6,5), dpi=200)
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax = fig.add_axes([left, bottom, width, height])
width = 0.35

# bar_width = 0.9/ len(predictions)
ticks = np.arange(len(models))

ax.bar(ticks, df['Bad Predictions'], width, label= 'Bad Predictions', bottom= df['Good Predictions'] + df['Excellent Predictions'])
ax.bar(ticks, df['Good Predictions'], width, align='center', label= 'Good Predictions', bottom = df['Excellent Predictions'])
ax.bar(ticks, df['Excellent Predictions'], width, align='center', label= 'Excellent Predictions')
ax.set_ylabel('Number of Predictions')
ax.set_title('Stacked bar plot for Faster RCNN and SSD using 100% labelled records')
ax.set_xticks(ticks)
ax.set_xticklabels(models)
ax.legend(loc = 'best')

plt.savefig('Stacked bar plot for Faster RCNN and SSD using 100% labelled records - experiment 2.jpg')

plt.show()
