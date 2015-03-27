import numpy as np
import matplotlib.pyplot as plt
import math
import cmath

def padwithzeros(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 255
    vector[-pad_width[1]:] = 255
    return vector

def padwithzerooos(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
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
                x = math.pow(-1.0,float(i+j))
                pgmData[i][j] = float(pgmData[i][j])*x
        return pgmData

    def pgmDataToPhasePicWithScale(self,filename,pgmDataFourier,pgmSize,pgmGreyscale):
        #pgmDataFourier = np.log(np.angle(pgmDataFourier))
        pgmDataFourier = np.angle(pgmDataFourier)
        #pgmDataFourier = 1+np.log2(1+pgmDataFourier)
        pgmDataFourier = np.round(pgmDataFourier*(float(pgmGreyscale[0])/(np.amax(pgmDataFourier)-np.amin(pgmDataFourier))),0).astype(int)
        #pgmDataFourierPhase = np.around(np.angle(pgmDataFourier)*(16),0).astype(int)
        self.buildPGMFile(filename+"phase",pgmSize[0],pgmSize[1],pgmGreyscale,pgmDataFourier)

    def pgmDataToAmplitudePicWithScale(self,filename,pgmDataFourier,pgmSize,pgmGreyscale):
        pgmDataFourier = np.round(np.abs(pgmDataFourier/float(pgmGreyscale[0])),0).astype(int)
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
        pgmDataInverse = np.round(np.abs(np.fft.ifft2(pgmDataFourier)),0).astype(int)
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

    def downSample(self,filename,ratio):
        pgmVer,pgmComment,pgmSize,pgmGreyscale,pgmData,htg = self.readPGMImage(filename)
        newPgmSize = int(int(pgmSize[0])*ratio)
        pgmDataNew  = np.full((newPgmSize, newPgmSize), 255, dtype=np.int)
        for j in range(newPgmSize):
            for i in range(newPgmSize):
                pgmDataNew[i][j] = pgmData[i*(1/ratio)][j*(1/ratio)]
        filename = filename.replace(".pgm", "");
        self.buildPGMFile(filename+"DownSamplePic",newPgmSize,newPgmSize,pgmGreyscale,pgmDataNew)

    def inverseFourierPgmWithOutAmplitude(self,filename):
        pgmVer,pgmComment,pgmSize,pgmGreyscale,pgmData,htg = myLib.readPGMImage(filename)
        pgmData = self.moveAxispgmDataBeforeFourier(pgmData,pgmSize)
        #print pgmData
        pgmDataFourier = self.convertToFourier(pgmData)
        pgmDataFourier = np.angle(pgmDataFourier)

        #pgmDataFourier = np.around(np.abs(np.fft.ifft2(pgmDataFourier)),0).astype(float)
        pgmDataFourier = np.abs(np.fft.ifft2(pgmDataFourier))
        pgmDataFourier = np.round(pgmDataFourier*(float(pgmGreyscale[0])/(np.amax(pgmDataFourier)-np.amin(pgmDataFourier))),0).astype(int)
        #print np.min(pgmDataFourier)
        pgmDataFourier = self.moveAxispgmDataBeforeFourier(pgmDataFourier,pgmSize)
        filename = filename.replace(".pgm", "");
        self.buildPGMFile(filename+"PgmWithOutAmplitude",pgmSize[0],pgmSize[1],pgmGreyscale,pgmDataFourier)

    def inverseFourierPgmWithOutPhase(self,filename):
        pgmVer,pgmComment,pgmSize,pgmGreyscale,pgmData,htg = myLib.readPGMImage(filename)
        pgmData = self.moveAxispgmDataBeforeFourier(pgmData,pgmSize)
        pgmDataFourier = self.convertToFourier(pgmData)
        pgmDataFourier = np.abs(pgmDataFourier)

        #pgmDataFourier = np.around(np.abs(np.fft.ifft2(pgmDataFourier)),0).astype(float)
        pgmDataFourier = np.log2(np.abs(np.fft.ifft2(pgmDataFourier)))
        pgmDataFourier = np.round(pgmDataFourier*(float(pgmGreyscale[0])/(np.amax(pgmDataFourier)-np.amin(pgmDataFourier))),0).astype(int)
        #pgmDataFourier = np.around(np.abs(pgmDataFourier/float(pgmGreyscale[0])),0)
        #pgmDataFourier = np.around(pgmDataFourier*(float(pgmGreyscale[0])/(np.amax(pgmDataFourier)-np.amin(pgmDataFourier))),0).astype(int)
        print np.min(pgmDataFourier)
        pgmDataFourier = self.moveAxispgmDataBeforeFourier(pgmDataFourier,pgmSize)
        filename = filename.replace(".pgm", "");
        self.buildPGMFile(filename+"PgmWithOutPhase",pgmSize[0],pgmSize[1],pgmGreyscale,pgmDataFourier)

    def convolutionWithKernel(self,inputFileName,kernelName,kernel,extendPadding):
        pgmVer,pgmComment,pgmSize,pgmGreyscale,pgmData,htg = self.readPGMImage(str(inputFileName)+".pgm")
        pgmData = np.lib.pad(pgmData, extendPadding, padwithzeros)
        pgmDataCon = np.zeros((int(pgmSize[1]),int(pgmSize[0])),dtype=np.float)
        pgmDataCon = np.lib.pad(pgmDataCon, extendPadding, padwithzeros)
        pgmDataCon.fill(255)
        #print pgmDataCon.shape
        for i in range(1,int(pgmSize[1])+1):
            for j in range(1,int(pgmSize[0])+1):
                temp = 0
                #XYY
                #YYY
                #YYY
                temp += pgmData[i][j]*kernel[1][1]
                temp += pgmData[i-1][j-1]*kernel[0][0]
                temp += pgmData[i-1][j]*kernel[0][1]
                temp += pgmData[i+1][j+1]*kernel[2][2]
                temp += pgmData[i][j-1]*kernel[1][0]
                temp += pgmData[i][j+1]*kernel[1][2]
                temp += pgmData[i+1][j-1]*kernel[2][0]
                temp += pgmData[i+1][j]*kernel[2][1]
                temp += pgmData[i+1][j+1]*kernel[2][2]
                pgmDataCon[i][j] = temp
        pgmDataCon = np.delete(pgmDataCon, int(pgmSize[1])+1, 0)
        #print pgmDataCon.shape
        pgmDataCon = np.delete(pgmDataCon, 0, 0)
        #print pgmDataCon.shape
        pgmDataCon = np.delete(pgmDataCon, int(pgmSize[0])+1, 1)
        #print pgmDataCon.shape
        pgmDataCon = np.delete(pgmDataCon, 0, 1)
        #print pgmDataCon.shape
        pgmDataCon = np.round(pgmDataCon,0).astype(int)
        self.buildPGMFile(str(inputFileName)+"Con"+str(kernelName),pgmSize[0],pgmSize[1],pgmGreyscale,pgmDataCon)

    def convolutionWithKernelFrequencyDomain(self,inputFileName,kernelName,kernel):
        pgmVer,pgmComment,pgmSize,pgmGreyscale,pgmData,htg = self.readPGMImage(str(inputFileName)+".pgm")
        pgmData = self.moveAxispgmDataBeforeFourier(pgmData,pgmSize)
        pgmDataFourier = self.convertToFourier(pgmData)

        kernel = np.lib.pad(kernel, 127, padwithzerooos)
        kernel = np.delete(kernel, 0, 1)
        kernel = np.delete(kernel, 0, 0)
        kernel = self.moveAxispgmDataBeforeFourier(kernel,pgmSize)
        kernelFourier = self.convertToFourier(kernel)
        pgmResult = kernelFourier*pgmDataFourier
        pgmResult = np.abs(np.fft.ifft2(pgmResult))
        pgmResult = np.round(self.moveAxispgmDataBeforeFourier(pgmResult,pgmSize),0).astype(int)
        self.buildPGMFile(str(inputFileName)+"ConFre"+str(kernelName),pgmSize[0],pgmSize[1],pgmGreyscale,pgmResult)

    #def idealLowPassFilter(self,filename,):

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
    pgmVer,pgmComment,pgmSize,pgmGreyscale,pgmData,htg = myLib.readPGMImage("CrossPaddingShiftAxis.pgm")
    pgmData = myLib.moveAxispgmDataBeforeFourier(pgmData,pgmSize)
    pgmDataFourier = myLib.convertToFourier(pgmData)
    myLib.pgmDataToPhasePicWithScale("CrossPaddingShiftAxis",pgmDataFourier,pgmSize,pgmGreyscale)
    myLib.pgmDataToAmplitudePicWithScale("CrossPaddingShiftAxis",pgmDataFourier,pgmSize,pgmGreyscale)
    """

    """
    #1.3
    myLib = ImageLibFourier()
    myLib.rotatePic("Cross.pgm",-30)
    pgmVer,pgmComment,pgmSize,pgmGreyscale,pgmData,htg = myLib.readPGMImage("CrossRotatePic.pgm")
    pgmData = myLib.moveAxispgmDataBeforeFourier(pgmData,pgmSize)
    pgmDataFourier = myLib.convertToFourier(pgmData)
    myLib.pgmDataToPhasePicWithScale("CrossRotatePic",pgmDataFourier,pgmSize,pgmGreyscale)
    myLib.pgmDataToAmplitudePicWithScale("CrossRotatePic",pgmDataFourier,pgmSize,pgmGreyscale)
    """

    """
    #1.4
    myLib = ImageLibFourier()
    myLib.downSample("Cross.pgm",0.5)
    pgmVer,pgmComment,pgmSize,pgmGreyscale,pgmData,htg = myLib.readPGMImage("CrossDownSamplePic.pgm")
    pgmData = myLib.moveAxispgmDataBeforeFourier(pgmData,pgmSize)
    pgmDataFourier = myLib.convertToFourier(pgmData)
    myLib.pgmDataToPhasePicWithScale("CrossDownSamplePic",pgmDataFourier,pgmSize,pgmGreyscale)
    myLib.pgmDataToAmplitudePicWithScale("CrossDownSamplePic",pgmDataFourier,pgmSize,pgmGreyscale)
    """

    """
    #1.5
    myLib = ImageLibFourier()
    myLib.inverseFourierPgmWithOutAmplitude("CrossPadding.pgm")
    myLib.inverseFourierPgmWithOutPhase("CrossPadding.pgm")
    """

    """
    #1.6
    myLib = ImageLibFourier()
    myLib.inverseFourierPgmWithOutAmplitude("Lenna.pgm")
    myLib.inverseFourierPgmWithOutPhase("Lenna.pgm")
    """

    """
    #1.7
    kernel = np.array([[1,2,1],[2,4,2],[1,2,1]],dtype=np.float)
    kernel = (1.0/16.0)*kernel
    myLib = ImageLibFourier()
    myLib.convolutionWithKernel("Chess","Blur",kernel,1)
    myLib.convolutionWithKernelFrequencyDomain("Chess","BlurInFourier",kernel)
    """