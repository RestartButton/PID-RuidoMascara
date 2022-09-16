import cv2
import numpy as np
from matplotlib import pyplot as plt


def normaliza(img) :
    _min = np.amin(img)
    _max = np.amax(img)
    result = (img - _min) * 255.0 / (_max - _min)
    return np.uint8(result)


def ruido(img) :
    valormedio = 0.0
    desviopadrao = 25.0
    ruido = np.random.normal(loc=valormedio, scale=desviopadrao, size=[img.shape[0],img.shape[1]]).astype('uint8')
    saida = cv2.add(img,ruido)
    return normaliza(saida)


def calculaMSE(iOrig, iRuid) :
    somaDif = 0
    for i in range(iOrig.shape[0]):
        for j in range(iRuid.shape[1]):
            somaDif += float((float(iOrig[i,j]) - float(iRuid[i,j])) ** 2)
    return somaDif/(iOrig.shape[0] * iOrig.shape[1])


def calculaPSNR(iOrig, iRuid) :
    return (np.max(np.log10(((np.max(iOrig) ** 2) / calculaMSE(iOrig,iRuid)))) * 10)


def aplicaMascara(img) :
    masc = np.ones((3,3),dtype="float") * float(1.0 / (3 ** 2))
    return cv2.filter2D(img, -1, masc)
    

def lena() :
    orig = cv2.imread("lena.png",0)
    result = ruido(orig)
    cMascara = aplicaMascara(result)

    plt.hist(orig.ravel(), bins = 25, range = [0,256], label="original", alpha=.8, edgecolor='red')
    plt.hist(result.ravel(),bins=25,range=[0,256], label="ruido", alpha=.7, edgecolor='yellow')
    plt.hist(cMascara.ravel(),bins=25,range=[0,256], label="mascara", alpha=.6, edgecolor='green')

    print("PSNR lena: " + str(calculaPSNR(orig,result)))
    cv2.imshow("Original",orig)
    cv2.imshow("Com Ruido",result)
    cv2.imshow("Com Mascara",cMascara)

    plt.legend()
    plt.show()


def jato() :
    orig = cv2.imread("jetplane.png",0)
    result = ruido(orig)
    cMascara = aplicaMascara(result)

    plt.hist(orig.ravel(), bins = 25, range = [0,256], label="original", alpha=.8, edgecolor='red')
    plt.hist(result.ravel(),bins=25,range=[0,256], label="ruido", alpha=.7, edgecolor='yellow')
    plt.hist(cMascara.ravel(),bins=25,range=[0,256], label="mascara", alpha=.6, edgecolor='green')

    print("PSNR jato: " + str(calculaPSNR(orig,result)))
    cv2.imshow("Original",orig)
    cv2.imshow("Com Ruido",result)
    cv2.imshow("Com Mascara",cMascara)

    plt.legend()
    plt.show()

def linux() :
    orig = cv2.imread("linux.png",0)
    result = ruido(orig)
    cMascara = aplicaMascara(result)

    plt.hist(orig.ravel(), bins = 25, range = [0,256], label="original", alpha=.8, edgecolor='red')
    plt.hist(result.ravel(),bins=25,range=[0,256], label="ruido", alpha=.7, edgecolor='yellow')
    plt.hist(cMascara.ravel(),bins=25,range=[0,256], label="mascara", alpha=.6, edgecolor='green')

    print("PSNR linux: " + str(calculaPSNR(orig,result)))
    cv2.imshow("Original",orig)
    cv2.imshow("Com Ruido",result)
    cv2.imshow("Com Mascara",cMascara)

    plt.legend()
    plt.show()


def main() :
    lena()
    jato()
    linux()


if __name__ == "__main__":
    main()