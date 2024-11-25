import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
from PIL import Image

data = pd.read_csv('clean_reviews.csv')
clean_reviews = data['Clean_Reviews']
clean_reviews = ' '.join(data['Clean_Reviews'])


mask = np.array(Image.open(r"C:\Users\Maik und Luisa\Desktop\portfolio.PNG"))
mask[mask == 1] = 255

wordcloud = WordCloud(
    background_color="white",
    mask=mask,
    contour_color='#FED8B1',
    contour_width=1,
    relative_scaling=0.5,
    colormap="inferno").generate(clean_reviews)

plt.figure(figsize=(10, 10), dpi=300)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
