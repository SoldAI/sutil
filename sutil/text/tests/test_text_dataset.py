from sutil.text.TextDataset import TextDataset

# Load the data
filename = "./sutil/datasets/tweets.csv"
t = TextDataset.standard(filename, ",")
print(t.texts)
print(t.X)
print(t.shape)
