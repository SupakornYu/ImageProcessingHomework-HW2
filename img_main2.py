import numpy as np
import matplotlib.pyplot as plt
import math

def padwithzeros(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 255
    vector[-pad_width[1]:] = 255
    return vector

class ImageLibFourier:
    def readPGMImage(self,path):
        file = open(path, "rb")
        pgmVer = file.readline().split()
        pgmComment = []
        while True:
            pgmComment_eachline = file.readline()
            if(pgmComment_eachline[0]=="#"):
                pgmComment.append(pgmComment_eachline)
            else:
                break
        pgmSize = pgmComment_eachline.split()
        pgmGreyscale = file.readline().split()
        pgmDataList = []
        htg = np.zeros((256),dtype=np.int32)
        np.set_printoptions(suppress=True)
        for j in range(int(pgmSize[1])):
            pgmDataX = []
            for i in range(int(pgmSize[0])):
                byte = file.read(1)
                chrToInt = ord(byte)
                pgmDataX.append(chrToInt)
                htg[chrToInt] = htg[chrToInt]+1
            pgmDataList.append(pgmDataX)
        file.close()
        pgmData = np.asarray(pgmDataList,dtype=np.int32)
        return pgmVer,pgmComment,pgmSize,pgmGreyscale,pgmData,htg

    def buildPGMFile(self,fileName,width,height,greyLevel,pgmData):
        f = open(str(fileName)+".pgm","wb")
        f.write("P5\n");
        f.write("# "+str(fileName)+"\n");
        f.write(str(width)+" "+str(height)+"\n"+str(greyLevel[0])+"\n");
        for i in range(int(height)):
            for j in range(int(width)):
                if pgmData[i][j]<0:
                    pgmData[i][j] = 0
                elif pgmData[i][j]>int(greyLevel[0]):
                    pgmData[i][j] = int(greyLevel[0])
                f.write(chr(pgmData[i][j]));
        f.close()



    def padImage(self,path,padWidthEachSide):
        pgmVer,pgmComment,pgmSize,pgmGreyscale,pgmData,htg = myLib.readPGMImage(path)
        pgmData = np.lib.pad(pgmData, padWidthEachSide, padwithzeros)
        pgmSize[0] = int(pgmSize[0]) + (int(padWidthEachSide)*2) #x axis
        pgmSize[1] = int(pgmSize[1]) + (int(padWidthEachSide)*2)
        path = path.replace(".pgm", "");
        self.buildPGMFile(path+"Padding",pgmSize[0],pgmSize[1],pgmGreyscale,pgmData)

    def convertToFourier(self,pgmData):
        return np.fft.fft2(pgmData)

    def scalepgmData(self,pgmData):
        pgmData = np.log(pgmData)
        return pgmData

    def moveAxispgmDataBeforeFourier(self,pgmData,pgmSize):
        for j in range(int(pgmSize[1])):
            for i in range(int(pgmSize[0])):
                x = math.pow(-1,i+j)
                pgmData[i][j] = pgmData[i][j]*x
        return pgmData

    def pgmDataToPhasePicWithScale(self,filename,pgmDataFourier,pgmSize,pgmGreyscale):
        #pgmDataFourier = np.log(np.angle(pgmDataFourier))
        pgmDataFourier = np.angle(pgmDataFourier)
        pgmDataFourier = np.around(pgmDataFourier*(float(pgmGreyscale[0])/(np.amax(pgmDataFourier)-np.amin(pgmDataFourier))),0).astype(int)
        #pgmDataFourierPhase = np.around(np.angle(pgmDataFourier)*(16),0).astype(int)
        self.buildPGMFile(filename+"phase",pgmSize[0],pgmSize[1],pgmGreyscale,pgmDataFourier)
        print pgmDataFourier
        return True


#np.set_printoptions(threshold=np.nan)
myLib = ImageLibFourier()
myLib.padImage("Cross.pgm",28)
pgmVer,pgmComment,pgmSize,pgmGreyscale,pgmData,htg = myLib.readPGMImage("CrossPadding.pgm")
pgmData = myLib.moveAxispgmDataBeforeFourier(pgmData,pgmSize)
pgmDataFourier = myLib.convertToFourier(pgmData)
print pgmDataFourier
myLib.pgmDataToPhasePicWithScale("CrossPadding",pgmDataFourier,pgmSize,pgmGreyscale)

pgmDataFourierDivideSampleAmp =  np.around(np.abs(pgmDataFourier)/(256),0).astype(int)
print pgmDataFourierDivideSampleAmp
print pgmDataFourierPhase

myLib.buildPGMFile("amplitude",pgmSize[0],pgmSize[1],pgmGreyscale,pgmDataFourierDivideSampleAmp)
print htg