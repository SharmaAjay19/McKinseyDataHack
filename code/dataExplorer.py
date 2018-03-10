import pandas as pd

challenge_data = pd.read_csv('../data/train/challenge_data.csv')
train_data = pd.read_csv('../data/train/train.csv')
test_data = pd.read_csv('../data/test/test.csv')

print(challenge_data.head())
print(train_data.head())
print(test_data.head())

##HERE GOES THE CODE FOR SEQUENCE PREDICTION##