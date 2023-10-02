import pandas as pd
from pathlib import Path
import os


path = Path(r"C:\Users\hp\Downloads\ml-100k\ml-100k\u.data")
userIdData, movieIdData, ratingData = [], [], []
with open(path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        line = line.split()
        userID = int(line[0])
        movieID = int(line[1])
        rating = int(line[2])


        userIdData.append(userID)
        movieIdData.append(movieID)
        ratingData.append(rating)
        # print(f'userID {userID} movieID {movieID} rating {rating}')

df = pd.DataFrame(list(zip(userIdData, movieIdData, ratingData)),
                  columns=['userID','movieID', 'rating'])

output_path = os.getcwd()
df.to_csv(os.path.join(output_path, 'Movielens.csv'), index=False)