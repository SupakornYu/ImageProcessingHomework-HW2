import numpy as np
import matplotlib.pyplot as plt
import math
import cmath

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
        #pgmDataFourier = 1+np.log2(1+pgmDataFourier)
        pgmDataFourier = np.around(pgmDataFourier*(float(pgmGreyscale[0])/(np.amax(pgmDataFourier)-np.amin(pgmDataFourier))),0).astype(int)
        #pgmDataFourierPhase = np.around(np.angle(pgmDataFourier)*(16),0).astype(int)
        self.buildPGMFile(filename+"phase",pgmSize[0],pgmSize[1],pgmGreyscale,pgmDataFourier)

    def pgmDataToAmplitudePicWithScale(self,filename,pgmDataFourier,pgmSize,pgmGreyscale):
        pgmDataFourier = np.around(np.abs(pgmDataFourier/float(pgmGreyscale[0])),0).astype(int)
        #pgmDataFourier = np.around(1+np.log2(1+np.abs(pgmDataFourier)),0).astype(int)
        self.buildPGMFile(filename+"Amplitude",pgmSize[0],pgmSize[1],pgmGreyscale,pgmDataFourier)
        #np.set_printoptions(threshold=np.nan)

    def shiftAxisInFourier(self,filename,a,b):
        pgmVer,pgmComment,pgmSize,pgmGreyscale,pgmData,htg = self.readPGMImage(filename)
        #pgmData = myLib.moveAxispgmDataBeforeFourier(pgmData,pgmSize)
        pgmDataFourier = self.convertToFourier(pgmData)
        #print pgmDataFourier
        for j in range(int(pgmSize[1])):
            for i in range(int(pgmSize[0])):
                #realSim = np.cos(-2*np.pi*(((a*i)/int(pgmSize[0]))+((b*j)/int(pgmSize[1]))))
                #complexSim = np.sin(-2*np.pi*(((a*i)/int(pgmSize[0]))+((b*j)/int(pgmSize[1]))))*1j
                #pgmDataFourier[i][j] = (realSim+complexSim)* pgmDataFourier[i][j]
                pgmDataFourier[i][j] = pgmDataFourier[i][j] * cmath.exp(-2j*np.pi*((a*i/float(pgmSize[0]))+(b*j/float(pgmSize[1]))))
        pgmDataInverse = np.around(np.abs(np.fft.ifft2(pgmDataFourier)),0).astype(int)
        #print pgmDataInverse
        filename = filename.replace(".pgm", "");
        self.buildPGMFile(filename+"ShiftAxis",pgmSize[0],pgmSize[1],pgmGreyscale,pgmDataInverse)

    def rotatePic(self,filename,angle):
        angle = np.deg2rad(angle)
        pgmVer,pgmComment,pgmSize,pgmGreyscale,pgmData,htg = self.readPGMImage(filename)
        pgmDataNew  = np.full((int(pgmSize[0]), int(pgmSize[0])), 255, dtype=np.int)

        for j in range(int(pgmSize[1])):
            for i in range(int(pgmSize[0])):
                xAxis = (i-100)*np.cos(angle)-(j-100)*np.sin(angle)
                yAxis = (i-100)*np.sin(angle)+(j-100)*np.cos(angle)
                xAxis = xAxis + 100
                yAxis = yAxis + 100
                #print xAxis
                #print yAxis
                if xAxis >= 200 or yAxis >= 200 or xAxis < 0 or yAxis < 0:
                    pgmDataNew[i][j] = 255
                else:
                    pgmDataNew[i][j] = pgmData[xAxis][yAxis]
        filename = filename.replace(".pgm", "");
        self.buildPGMFile(filename+"RotatePic",pgmSize[0],pgmSize[1],pgmGreyscale,pgmDataNew)
        #print pgmDataNew

if __name__ == "__main__":

    """
    #1.1
    myLib = ImageLibFourier()
    myLib.padImage("Cross.pgm",28)
    pgmVer,pgmComment,pgmSize,pgmGreyscale,pgmData,htg = myLib.readPGMImage("CrossPadding.pgm")
    pgmData = myLib.moveAxispgmDataBeforeFourier(pgmData,pgmSize)
    pgmDataFourier = myLib.convertToFourier(pgmData)
    myLib.pgmDataToPhasePicWithScale("CrossPadding",pgmDataFourier,pgmSize,pgmGreyscale)
    myLib.pgmDataToAmplitudePicWithScale("CrossPadding",pgmDataFourier,pgmSize,pgmGreyscale)
    """

    """
    #1.2
    myLib = ImageLibFourier()
    myLib.padImage("Cross.pgm",28)
    myLib.shiftAxisInFourier("CrossPadding.pgm",20,30)
    """

    #1.3
    myLib = ImageLibFourier()
    myLib.rotatePic("Cross.pgm",-30)
