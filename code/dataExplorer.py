import pandas as pd
from sklearn.preprocessing import LabelEncoder

encoderDict = {}
encoderDict['challenge_ID'] = LabelEncoder()
encoderDict['challenge_series_ID'] = LabelEncoder()
encoderDict['author_ID'] = LabelEncoder()
encoderDict['author_org_ID'] = LabelEncoder()
encoderDict['author_gender'] = LabelEncoder()
encoderDict['category_id'] = LabelEncoder()

challenge_data = pd.read_csv('../data/train/challenge_data.csv')
train_data = pd.read_csv('../data/train/train.csv')
test_data = pd.read_csv('../data/test/test.csv')

challenge_data = challenge_data.apply(lambda x: encoderDict[x.name].fit_transform(x))

print(challenge_data.head())
print(train_data.head())
print(test_data.head())

#challenge_data['challenge_ID'] = challenge_data['challenge_ID'].apply(lambda x: int(x[2:]))
##HERE GOES THE CODE FOR SEQUENCE PREDICTION##

