# Cheat sheet

# Add markdown
st.title ("this is the app title")
st.header("this is the markdown")
st.markdown("this is the header")
st.subheader("this is the subheader")
st.caption("this is the caption")
st.code("x=2021")st.latex(r''' a+a r^1+a r^2+a r^3 ''')


# Display image
st.image("kid.jpg")

# Display dataframe
# pandas

import pandas as pd

df = pd.read_csv('cc_rfm.csv')
st.dataframe(df) 

# Display plot (on-call)

# add these lines in requirements.in
# matplotlib
# numpy

import matplotlib.pyplot as plt
import numpy as np

rand=np.random.normal(1, 2, size=20)
fig, ax = plt.subplots()ax.hist(rand, bins=15)st.pyplot(fig)

 # See https://www.datacamp.com/tutorial/streamlit to learn more about how to implement more streamlit features in-depth