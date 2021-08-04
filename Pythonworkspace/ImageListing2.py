import os
import random

def partitionRankings(rawRatings, testPercent):
    howManyNumbers = int(round(testPercent*len(rawRatings)))
    shuffled = rawRatings[:]
    random.shuffle(shuffled)
    return shuffled[howManyNumbers:], shuffled[:howManyNumbers]

trainfile = 'train.txt'
testfile = 'test.txt'
fext = ('.jpg', '.JPG', '.jpeg', '.png')
root = 'images/'                                                       # image directory
os.chdir(root)  # change working directory to root

if os.path.exists(testfile):
    os.remove(testfile)
if os.path.exists(trainfile):
    os.remove(trainfile)

files = [f for f in os.listdir('.') if f.endswith(fext)]     # all file names in directory
test, training = partitionRankings(files, 80/100)

with open(trainfile, 'a+') as f:
    for file in training:
        f.write('custom_data/images/' + file + '\n')

with open(testfile, 'a+') as f:
    for file in test:
        f.write('custom_data/images/' + file + '\n')