import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import random
print("Start prepare the data: ")
print("========================")

trainMyImagesFolder = "D:/UNA-PUNO/MACHINE LEARNING/monkeys/training/training"
testMyImagesFolder = "D:/UNA-PUNO/MACHINE LEARNING/monkeys/validation/validation"

def chekMyDir(dir):
    folders = len(glob.glob(dir + '/*'))
    image_files = len(glob.glob(dir + '/*/*.jpg' ))

    print("--->>> The Data Folder: {} contains {} foldeer and {} images.".format(dir,folders,image_files))

print(chekMyDir(trainMyImagesFolder))
print(chekMyDir(testMyImagesFolder))

columns = ["Label","Common Name","Train Images","Validation Images"]


df = pd.read_csv("D:/UNA-PUNO/MACHINE LEARNING/monkeys/monkey_labels.txt",names=columns,skiprows=1)

df['Label'] = df['Label'].str.strip()
df['Common Name'] = df['Common Name'].str.strip()

df = df.set_index("Label")

print(df)


monkeyDic = df['Common Name']
print(monkeyDic)

print(monkeyDic['n0'])

#lets show some images randomly
#each columns will hold 6 images


def displayDirectory(dir):
    folderList = os.listdir(dir)
    folderList.sort()

    numOfClasses = len(folderList)
    columnForDisplay = 6

    fig , ax = plt.subplots(numOfClasses, columnForDisplay, figsize=(3*columnForDisplay, 3*numOfClasses))#space for images

    for countRow , folderClassItem in enumerate(folderList):
        path = os.path.join(dir,folderClassItem)
        subDirList = os.listdir(path)
        #print(subDirList)
        #now, lets road 6 random images in each category

        for i in range(columnForDisplay):
            randomImageFile = random.choice(subDirList)
            imageFilePath = os.path.join(path,randomImageFile)
            #print(imageFilePath)
            img = plt.imread(imageFilePath)
            monkeyLabel = monkeyDic[folderClassItem]
            monkeyLabel = monkeyLabel[:10] # get only first 10 characters

            ax[countRow,i].set_title(monkeyLabel)
            ax[countRow,i].imshow(img)
            ax[countRow,i].axis('off')

    plt.show()

displayDirectory(trainMyImagesFolder)
