# Step detection workshop

A tutorial on the step detection task, using convolutional sparse coding.

The tutorial is in the notebook `step-detection-workshop.ipynb`.

To run the notebook, you will need to install the following packages:

- pandas
- scikit-learn
- matplotlib
- loadmydata
- seaborn

They are available through `pip` or `conda`.


Downloading the data can take a few minutes.
In order to download them beforehand, please run the following commands in a Python terminal:

```python
from loadmydata.load_human_locomotion import load_human_locomotion_dataset
_ = load_human_locomotion_dataset("1-1")
```
