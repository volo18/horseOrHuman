import os
import glob
import time

imFrmt = '.png'

def getFileNames(folderPth, classNames):

    fileList = []

    for curClass in classNames:

        curPth = folderPth + curClass + '/'

        fileNames   = glob.glob(curPth + '*')

        fileList.extend(fileNames)

    return fileList

def writeFileNames(names,fName):
    
    with open(fName, 'w') as f:
        for item in names:
            f.write("%s\n" % item)


if __name__ == "__main__":
    
    startTime = time.time()

    trainingDir = './data/train/'
    validDir    = './data/validation/'

    classes     = [ 'horses', 'humans' ]

    trainingFileNames   = getFileNames(trainingDir, classes)
    validationFileNames = getFileNames(validDir, classes)

    writeFileNames(trainingFileNames,'trainingFileNames.txt')
    writeFileNames(validationFileNames,'validationFileNames.txt')
    
    endTime = time.time()

    timeTaken = endTime - startTime
    doneStr = '[Done! Time Taken: %.2f]' % (timeTaken)
    print(doneStr)
