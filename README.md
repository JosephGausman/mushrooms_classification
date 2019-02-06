## Mushrooms - Multi-Class Classification

The aim of this classification problem is to predict the odor of the mushrooms based on 22 features and 8124 instances.

Odor is the target variable which contains the next 9 classes:

odor: almond=a, anise=l, creosote=c, fishy=y, foul=f, musty=m, none=n, pungent=p, spicy=s

## Features

| |
| --- |
| class: edible=e, poisonous=p |
| cap-shape: bell=b, conical=c, convex=x, flat=f, knobbed=k, sunken=s |
| cap-surface: fibrous=f, grooves=g, scaly=y, smooth=s |
| cap-color: brown=n, buff=b, cinnamon=c, gray=g, green=r, pink=p, purple=u, red=e, white=w, yellow=y |
| bruises: bruises=t, no=f |
| gill-attachment: attached=a, descending=d, free=f, notched=n |
| gill-spacing: close=c, crowded=w, distant=d |
| gill-size: broad=b, narrow=n |
| gill-color: black=k, brown=n, buff=b, chocolate=h, gray=g, green=r, orange=o, pink=p, purple=u, red=e, white=w, yellow=y |
| stalk-shape |
| stalk-root |
| stalk-surface-above-ring |
| stalk-surface-below-ring |
| stalk-color-above-ring |
| stalk-color-below-ring |
| veil-type |
| veil-color |
| ring-number |
| ring-type |
| spore-print-color |
| population |
| habitat |

## Model and hyper-parameters comparison

|   | Score %Default Parameters | Score %Grid Search | Best Parameters |
| --- | --- | --- | --- |
| ﻿Logistic Regression | ﻿79.01 | ﻿80.26 | ﻿C=0.1, penalty=l1 |
| ﻿KNN | ﻿72.43 | 76.68 | ﻿﻿Algorithm=auto, n\_neighbors: 30, weights: uniform |
| ﻿Random Forests | ﻿69.78 | ﻿80.34 | ﻿Criterion=gini, max\_depth=5, n\_estimators=60 |
| ﻿SVM | ﻿79.14 | 81.34 | C=0.1, decision\_function\_shape=ovr, &#39;kernel=rbf |

## Conclusions

- SVM was found to be the best model to predict mushroom odor, although Logistic Regression and Random Forests yielded almost the same results.
- Random Forest model was found to be the most sensitive to parameter changes, the model improved from 69.78 to 80.34% due to grid search. The number of estimators and max\_depth was responsible for the model improvement.
- In my opinion, all the models didn&#39;t do a great job since there are 9 classes that needs to be predicted. On average its less than 1000 instances for each class. Therefore, more data is needed in order to improve the prediction power of the models.
