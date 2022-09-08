from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QApplication, QFileDialog, QShortcut
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import QDate, QTime, QDateTime, Qt, QSize
from PyQt5.QtGui import QPainter, QKeySequence
from omegaconf import OmegaConf
import torch.cuda
from torch.backends import cudnn
from itertools import count
from subprocess import call, PIPE, run
from easygui import fileopenbox, diropenbox
import os
import random
from math import floor, ceil
from PIL import Image
import cv2 as cv
from qdarkstyle import load_stylesheet
from threading import Thread
import sys
from resizeimage import resizeimage
from BSRGAN_main.main_test_bsrgan import main as bsrgan
from predict_sr import predict
from ldm.util import instantiate_from_config
from skimage import exposure
import cv2
# import numpy
import numpy as np


ac = False
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True) #enable highdpi scaling
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True) #use highdpi icons

## set up gui elements

outputPath = ""
uiDir = '.\\UI\\'
modelDir = '.\\models\\ldm\\stable-diffusion-v1\\'
iconDir = '.\\UI\\icons\\'

dialogues = {'dlg': 'StableDiffusion.ui', 'grd': 'gridPreview.ui', 'upr': 'upres.ui'}

app = QtWidgets.QApplication([])

for keys, values in dialogues.items():
    globals()[keys] = uic.loadUi(uiDir + values)

dlg.setWindowIcon(QtGui.QIcon(iconDir + 'StableDifusion.ico'))
dlg.scaleSlider.setMaximum(100)
dlg.strSlider.setMinimum(1)
dlg.strSlider.setMaximum(99)
grd.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
img = QPixmap(iconDir + 'splash.png')
grd.gridPreview.setPixmap(img)
grd.show()

from scripts.txt2imgModule import main as txt2img
from scripts.txt2imgModule import load_model_from_config
from scripts.img2imgModule import main as img2img

## initial setup
defaultWidth = 934
defaultHeight = 700
fixedWidth = 934
fixedHeight = 700
previewWidth = 0
previewHeight = 0
dlg.sizeHint()
checkpoints = next(os.walk(modelDir))[-1]
checkpoints.sort(reverse = True)
dlg.sampleEntry.setText('1')
dlg.iterationEntry.setText('1')
dlg.widthSlider.setValue(512)
dlg.heightSlider.setValue(512)
dlg.strSlider.setValue(66)
dlg.strValue.setAlignment(Qt.AlignLeft)

for i in checkpoints:
    dlg.checkDrop.addItem(i)

def setSeed():
    seed = random.randint(42,429496729)
    dlg.seedEntry.setText(str(seed))

def setSliders():

    if dlg.initCheck.isChecked() == True:
        stepMax = (int(ceil(1/float(dlg.strValue.text())*500)))
        stepShow = (int(dlg.stepSlider.value() * float(dlg.strValue.text())))
        dlg.stepSlider.setMaximum(stepMax)
        dlg.stepValue.setText(str(stepShow))
    else:
        dlg.stepValue.setText(str(dlg.stepSlider.value()))

    dlg.scaleValue.setText(str(dlg.scaleSlider.value()))
    dlg.widthValue.setText(str(dlg.widthSlider.value()))
    dlg.heightValue.setText(str(dlg.heightSlider.value()))
    dlg.strValue.setText(str(float(dlg.strSlider.value() / 100)))
    dlg.zoomValue.setText(str(float(dlg.zoomSlider.value() / 100)))
    dlg.xtransValue.setText(str(dlg.xtransSlider.value()))
    dlg.ytransValue.setText(str(dlg.ytransSlider.value()))
    dlg.rotValue.setText(str(dlg.rotSlider.value()))
    dlg.xturnValue.setText(str(dlg.xturnSlider.value()))
    dlg.yturnValue.setText(str(dlg.yturnSlider.value()))
    upr.upresValue.setText(str(upr.upresSlider.value()))


setSliders()
setSeed()

def setWidthIntervals():
    result = next(
    number
    for distance in count()
    for number in (dlg.widthSlider.value() + distance, dlg.widthSlider.value() - distance)
    if number % 64 == 0
    )
    dlg.widthSlider.setValue(result)
    initImage()

def setHeightIntervals():
    result = next(
    number
    for distance in count()
    for number in (dlg.heightSlider.value() + distance, dlg.heightSlider.value() - distance)
    if number % 64 == 0
    )
    dlg.heightSlider.setValue(result)
    initImage()

def setOutput():
    try:
        outputPath = diropenbox()
        dlg.outputEntry.setText(outputPath)
    except:
        pass

## Action functions

def initCheck():
    global initImg
    global outputPath
    global previewPath
    global ac
    setSliders()
    dlg.precisionDrop.setCurrentIndex(0)
    global defaultHeight
    global defaultWidth
    try:
        im = Image.open(outputPath)
        width, height = im.size
    except:
        width = 0
        height = 0
    if dlg.initCheck.isChecked() == True:
        try:
            flexWindow(outputPath)
        except Exception as e: print(e)


        dlg.initButton.setEnabled(True)
        dlg.strSlider.setEnabled(True)
        dlg.strValue.setEnabled(True)
        dlg.strLabel.setEnabled(True)
        dlg.imgTypeDrop.setEnabled(True)
        defaultHeight = defaultHeight + 50
        torch.cuda.empty_cache()

    else:
        dlg.precisionDrop.setCurrentIndex(1)
        dlg.initButton.setEnabled(False)
        dlg.widthValue.setEnabled(True)
        dlg.widthSlider.setEnabled(True)
        dlg.heightValue.setEnabled(True)
        dlg.heightSlider.setEnabled(True)
        # dlg.strSlider.setEnabled(False)
        # dlg.strValue.setEnabled(False)
        # dlg.strLabel.setEnabled(False)
        dlg.imgTypeDrop.setEnabled(False)
        dlg.initPreview.clear()
        defaultHeight = 600
        defaultWidth = 934
        initImg = ""
        outputPath = ""
        previewPath = ""
        ac = False
        torch.cuda.empty_cache()

initCheck()

def initImage():
    global defaultWidth
    global defaultHeight
    global width
    global height
    global initImg
    global outputPath
    global ac

    if dlg.initCheck.isChecked() == True:
        try:
            if outputPath == "":
                initImg = fileopenbox(title='Please select an Initial Image')
                initImg = initImg.replace('\\', '//')
            else:
                if ac == True:
                    initImg = outputPath
                    initImg = initImg.replace('\\', '//')
            print("init Image is " + initImg)
            with open(initImg, 'r+b') as f:
                with Image.open(f) as image:
                    cover = resizeimage.resize_cover(image, [dlg.widthSlider.value(), dlg.heightSlider.value()])
                    cover.save('./resize.png')
            outputPath = "./resize.png"
            flexWindow(outputPath)

        except Exception as e: print(e)

def initClicked():
    global outputPath
    global ac
    ac = False
    outputPath = ""
    initImage()

def flexWindow(outputPath):
    global defaultWidth
    global defaultHeight
    global width
    global height

    im = Image.open(outputPath)
    width, height = im.size
    dlg.widthValue.setText(str(width))
    dlg.heightValue.setText(str(height))
    preview = QPixmap(outputPath)
    dlg.initPreview.setPixmap(preview)
    dlg.sizeHint()
    if defaultWidth + width >= width + fixedWidth:
        defaultWidth = defaultWidth + int(width) + 40
    if defaultWidth + width >= (width + previewWidth) + fixedWidth:
        defaultWidth = width + fixedWidth + 40

def darkTheme():
    if dlg.darkCheck.isChecked() == True:
        dlg.setStyleSheet(load_stylesheet())
    else:
        dlg.setStyleSheet('Windows')

def generate(animCheck,test):
    global previewWidth
    global previewHeight
    global previewPath

    torch.cuda.empty_cache()
    ## set variables
    if dlg.seedCheck.isChecked() == False:
        setSeed()
    else:
        pass
    prompt = str(dlg.promptEntry.text())
    if prompt == '':
        prompt = 'photo still of danny trejo looking into the camera very disapointedly, 8k, 85mm f1.8'
    scale = int(dlg.scaleSlider.value())
    steps = int(dlg.stepSlider.value())
    width = int(dlg.widthValue.text())
    height = int(dlg.heightValue.text())
    checkpoint = dlg.checkDrop.currentText()
    formatNumber = lambda n: n if n%1 else int(n)
    rows = abs((int(dlg.sampleEntry.text()) * int(dlg.iterationEntry.text())/2))
    rows = int(formatNumber(rows))
    strength = float(dlg.strValue.text())

    if float(rows)//1 != float(rows)/1:
        if float(rows) == .5:
            rows = str(ceil(float(rows)))
        else:
            rows = str(floor(float(rows)))

    if dlg.modelDrop.currentText() == 'plms':
        sampler = True
    else:
        sampler = False
    precision = dlg.precisionDrop.currentText()
    seed = int(dlg.seedEntry.text())
    samples = int(dlg.sampleEntry.text())
    iterations = 1

    if outputPath != "":
        initImage = str(outputPath)
        initImage = initImage.replace('\\', '/')
    if dlg.outputEntry.text() == "":
        if dlg.initCheck.isChecked() == False:
            outputDir = './/outputs//txt2img-samples//'
        else:
            outputDir = './/outputs//img2img-samples//'
    else:
        outputDir = str(dlg.outputEntry.text()) + '//'
        if dlg.initCheck.isChecked() == False:
            if os.path.exists(outputDir + '//txt2img-samples//'):
                outputDir = outputDir + '//txt2img-samples//'
            else:
                os.mkdir(outputDir + '//txt2img-samples//')
                outputDir = outputDir + '//txt2img-samples//'
        else:
            if os.path.exists(outputDir + '//img2img-samples//'):
                outputDir = outputDir + '//img2img-samples//'
            else:
                os.mkdir(outputDir + '//img2img-samples//')
                outputDir = outputDir + '//img2img-samples//'

    ## run generation
    try:

        if dlg.initCheck.isChecked() == False:
            txt2img(model = model, plms = sampler, prompt = prompt, seed = seed, ckpt = './models/ldm/stable-diffusion-v1/' + checkpoint, scale = scale,
                    ddim_steps = steps, n_iter = iterations, n_samples = samples, W = width, H = height, precision = precision,
                    outdir = outputDir, n_rows = rows)

        else:
            try:
                im = Image.open(outputPath)
            except:
                initImage()
            width, height = im.size
            img2img(animCheck, test, device = device, model = model, prompt = prompt, seed = seed, init_img = initImage, ckpt = './models/ldm/stable-diffusion-v1/' + checkpoint, scale = scale,
                    ddim_steps = steps, n_iter = iterations, n_samples = samples, precision = precision, outdir = outputDir, n_rows = rows, strength = strength)

        previewFile = next(os.walk(outputDir + '//samples//'))[-1]
        previewFile.sort(reverse = True)
        previewFile = previewFile[0]
        preview = QPixmap(outputDir + '//samples//' + previewFile)
        previewPath = outputDir + '//samples//' + previewFile

        QApplication.processEvents()
        dlg.imgPreview.setPixmap(preview)

        if int(dlg.sampleEntry.text()) > 1:
            grd.setMinimumWidth(int(width)*int(rows))
            grd.setMinimumHeight(int(height) * ceil((abs((int(dlg.sampleEntry.text()) * int(dlg.iterationEntry.text()))) / int(rows))))
            previewFile = next(os.walk(outputDir))[-1]
            previewFile.sort(reverse = True)
            previewFile = previewFile[0]
            preview = QPixmap(outputDir + previewFile)
            grd.gridPreview.setPixmap(preview)
            grd.setWindowTitle(prompt)
            grd.activateWindow()
            grd.show()

        dlg.sizeHint()
        dlg.activateWindow()
        torch.cuda.empty_cache()
        previewWidth = width
        previewHeight = height

    except Exception as e: print(e)


    torch.cuda.empty_cache()

dlg.widthValue.setText('512')
dlg.heightValue.setText('512')

## Load model into memory
def loadModel():
    global model
    global device
    torch.cuda.empty_cache()
    config ="configs/stable-diffusion/v1-inference.yaml"
    config = OmegaConf.load(f"{config}")

    ckpt = './models/ldm/stable-diffusion-v1/' + dlg.checkDrop.currentText()
    model = load_model_from_config(config, f"{ckpt}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    if dlg.precisionDrop.currentText() == 'autocast':
        model.half()
    torch.cuda.empty_cache()

def loadBSRModel():
    global model
    torch.cuda.empty_cache()
    ckpt = "models/ldm/bsr_sr/model.ckpt"
    # subprocess.call(["pip", "install", "-e", "."])
    global config, model, global_step, device
    device = torch.device("cuda")
    conf = "models/ldm/bsr_sr/config.yaml"
    # config = OmegaConf.load("/src/configs/latent-diffusion/superres.yaml")
    config = OmegaConf.load(conf)
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cuda")
    global_step = pl_sd["global_step"]
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    torch.cuda.empty_cache()

def loadPrompt():
    global previewPath
    global initImg
    global outputPath
    outputPath = ""
    initImg = ""
    try:
        prompt = fileopenbox(title="Please select an image to load")
        previewPath = prompt
        im = Image.open(prompt)
        preview = prompt.replace('\\', '//')
        flexWindow(outputPath = preview)
        width, height = im.size
    except:
        pass
    try:
        dlg.heightSlider.setValue(height)
        dlg.widthSlider.setValue(width)
        dlg.seedCheck.setChecked(True)
        dlg.scaleSlider.setValue(int(im.text['scale']))
        dlg.stepSlider.setValue(int(im.text['steps']))
        dlg.seedEntry.setText(str(im.text['seed']))
        dlg.promptEntry.setText(im.text['prompt'])
        try:
            dlg.strSlider.setValue(int(im.text['strength']))
        except:
            pass
        try:
            index = dlg.checkDrop.findText(im.text['ckpt'].split('/')[-1], QtCore.Qt.MatchFixedString)
            dlg.checkDrop.setCurrentIndex(index)
        except:
            pass
    except:
        pass

def imgCheck():
    dlg.initButton.setText('Init Image')
    dlg.initButton.setIcon(QIcon(iconDir + 'img.png'))

def imgLoop():
    global outputPath
    if dlg.imgTypeDrop.currentText() == 'still':
        if dlg.iterationEntry.text() == 1:
            if dlg.initCheck.isChecked() == False:
                generate(False,"")
            else:
                image=Image.open(outputPath.replace('\\', '/'))
                test=cv2.cvtColor(np.asarray(image.copy()), cv2.COLOR_RGB2LAB)
                generate(True,test)
        else:
            for i in range(int(dlg.iterationEntry.text())):
                if dlg.initCheck.isChecked() == False:
                    generate(False,"")
                else:
                    try:
                        image=Image.open(outputPath.replace('\\', '/'))
                        test=cv2.cvtColor(np.asarray(image.copy()), cv2.COLOR_RGB2LAB)
                        generate(True,test)
                    except:
                        initImage()
    else:
        try:
            img = outputPath.replace('\\', '/')
            imageList = next(os.walk('/'.join(img.split('/')[0:-1])))[-1]
            imageDir = next(os.walk('/'.join(img.split('/')[0:-1])))[0]
            # print(imageList)
            # print(imageDir)
            for i in imageList:
                if '.png' not in i:
                    imageList.remove(i)

            for i in imageList:
                outputPath = imageDir + '\\' + i
                initImage()
                QApplication.processEvents()
                image=Image.open(outputPath.replace('\\', '/'))
                test=cv2.cvtColor(np.asarray(image.copy()), cv2.COLOR_RGB2LAB)
                initImg = QPixmap(outputPath.replace('\\', '/'))
                dlg.initPreview.setPixmap(initImg)
                generate(True,test)

        except:
            initImage()

def clearPreview():
    global outputPath
    global previewPath
    global initImg

    outputPath = ""
    previewPath = ""
    initImg = ""

    dlg.initPreview.clear()
    dlg.imgPreview.clear()
    if os.path.exists('./resize'):
        os.remove('./resize')

def upres():
    torch.cuda.empty_cache()
    global E_file
    global img

    try:
        try:
            img = upresImage
        except:
            img = fileopenbox(title='Please select an Image to upres')
        if dlg.outputEntry.text() == "":
            if os.path.exists('.//outputs//upres//'):
                outputDir = './/outputs//upres//'
            else:
                os.mkdir('.//outputs//upres//')
                outputDir = './/outputs//upres//'
        else:
            outPath = dlg.outputEntry.text().replace('\\', '//')
            if not os.path.exists(outPath + '//upres//'):
                os.mkdir(outPath + '//upres//')
                outputDir = outPath + '//upres//'
            else:
                outputDir = outPath + '//upres//'

        E_file = outputDir
        upr.setWindowTitle("upres " + os.path.basename(img))
        upr.show()
        upr.upresSlider.setValue(100)
    except Exception as e: print(e)

def runUpres():
    if upr.upresDrop.currentText() == "BSRGaN":
        bsrgan(img,E_file)
    elif upr.upresDrop.currentText() == "Latent_SR":
        loadBSRModel()
        predict(BSRmodel = model, image = img, up_f = 4, steps = upr.upresSlider.value(), outPath = E_file)
        torch.cuda.empty_cache()
    upr.close()
    loadModel()
    torch.cuda.empty_cache()

def upresCombo():
    if upr.upresDrop.currentText() == 'BSRGaN':
        upr.upresSlider.setVisible(False)
    else:
        upr.upresSlider.setVisible(True)

def upresLast():
    global upresImage
    try:
        upresImage = previewPath
        upres()
    except:
        upres()

def upresAction():
    global upresImage
    try:
        del upresImage
        upres()
    except:
        upres()

def transform(img, x, y, zoom):
    width, height = img.size
    scale = zoom
    x = int(width * scale)
    y = int(height * scale)
    sx = int((width - x)/2) - int(dlg.xtransValue.text())
    sy = int((height - y)/2) + int(dlg.ytransValue.text())

    xTurnUp = dlg.xturnSlider.value()*5
    xTurnDown = dlg.xturnSlider.value()*2.5
    yTurnUp = dlg.yturnSlider.value()*5
    yTurnDown = dlg.yturnSlider.value()*2.5

    if dlg.xturnSlider.value() <= 0:
        # print('negx')
        xturnCoeffs = find_coeffs(
            [(0, 0), (width, 0), (width, height), (0, height)],
            [(xTurnUp, xTurnDown), (width, -xTurnDown), (width, height+xTurnDown), (xTurnUp, height-xTurnDown)])
    else:
        # print('posx')
        xturnCoeffs = find_coeffs(
            [(0, 0), (width, 0), (width, height), (0, height)],
            [(0, xTurnDown), (width+xTurnUp, -xTurnDown), (width+xTurnUp, height+xTurnDown), (0, height-xTurnDown)])

    if dlg.yturnSlider.value() >= 0:
        # print('posy')
        yturnCoeffs = find_coeffs(
            [(0, 0), (width, 0), (width, height), (0, height)],
            [(-yTurnUp, 0), (width+yTurnUp, 0), (width-yTurnUp, height-yTurnDown), (yTurnUp, height-yTurnDown)])
    else:
        # print('negy')
        yturnCoeffs = find_coeffs(
            [(0, 0), (width, 0), (width, height), (0, height)],
            [(-yTurnUp, -yTurnDown), (width+yTurnUp, -yTurnDown), (width-yTurnUp, height), (+yTurnUp, height)])


    image = img.transform((width, height), Image.Transform.PERSPECTIVE, xturnCoeffs,
                  Image.Resampling.BICUBIC)

    image = image.transform((width, height), Image.Transform.PERSPECTIVE, yturnCoeffs,
                  Image.Resampling.BICUBIC)

    image = image.resize((x,y))
    image = image.rotate(dlg.rotSlider.value(), Image.NEAREST)
    img.paste(image, (sx,sy), mask=image)
    return img

def find_coeffs(source_coords, target_coords):
    matrix = []
    for s, t in zip(source_coords, target_coords):
        matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0]*t[0], -s[0]*t[1]])
        matrix.append([0, 0, 0, t[0], t[1], 1, -s[1]*t[0], -s[1]*t[1]])
    A = np.matrix(matrix, dtype=np.float)
    B = np.array(source_coords).reshape(8)
    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

def animate():
    global outputPath
    global outputDir
    global initImg
    global ac
    dlg.seedCheck.setChecked(False)
    ac = True

    if dlg.outputEntry.text().split('/')[-1] != dlg.projEntry.text():
        dlg.outputEntry.clear()

    try:
        image=Image.open(initImg)
        outputPath = initImg
        initImage()
        dlg.imgPreview.clear()
    except:
        image=Image.open(previewPath)
        outputPath = previewPath
        initImage()
        dlg.imgPreview.clear()
    test=cv2.cvtColor(np.asarray(image.copy()), cv2.COLOR_RGB2LAB)

    if dlg.initCheck.isChecked() == False:
        dlg.stepSlider.setValue(
                                dlg.stepSlider.value()
                                /
                                float(dlg.strValue.text())
                                )

    dlg.initCheck.setChecked(True)

    if dlg.projEntry.text() == "":
        dlg.projEntry.setText("untitled_" + dlg.seedEntry.text())

    if dlg.outputEntry.text() == "":
        if os.path.exists('./outputs/animate/' + dlg.projEntry.text()):
            outputDir = './outputs/animate/' + dlg.projEntry.text()
        else:
            os.makedirs('./outputs/animate/' + dlg.projEntry.text())
            outputDir = './outputs/animate/' + dlg.projEntry.text()
        dlg.outputEntry.setText(outputDir)
    else:
        outPath = dlg.outputEntry.text().replace('\\', '//')

        if not os.path.exists(outPath):
            os.makedirs(outPath)
            outputDir = outPath
        else:
            outputDir = outPath


    steps = dlg.stepSlider.value()
    strength = dlg.strSlider.value()

    if dlg.lenEntry.text() == "":
        animLength = 60
    else:
        animLength = int(dlg.lenEntry.text())

    for i in range(0, int(float(animLength))):

            img = Image.open(outputPath).convert('RGBA')
            width, height = img.size
            print('frame: ' + str(i+1))

            if i % 2:
                dlg.strSlider.setValue(strength)
                dlg.stepSlider.setValue(steps)
            else:
                dlg.strSlider.setValue(int(strength/2))
                dlg.stepSlider.setValue(steps*2)

            scale = float(dlg.zoomValue.text())

            image = transform(img, width/2, height/2, scale)

            image = image.save('./resize.png')

            QApplication.processEvents()
            dlg.imgPreview.clear()
            initImage()
            dlg.imgPreview.clear()
            try:
                generate(True, test)
            except:
                initImage()
            dlg.imgPreview.clear()
            outputPath = previewPath
            # initImg = previewPath
            initImage()
            dlg.imgPreview.clear()

def outputVideo():
    try:
        outputDir = dlg.outputEntry.text().replace('\\', '/') + '/'
        dlg.projEntry.setText(
                            outputDir.split('/')[-2]
                            )
        pngPath = dlg.outputEntry.text() + '/img2img-samples/samples/'
        outVidName = dlg.projEntry.text() + '.mp4'
        pngFiles = os.listdir(pngPath)
        fullPath = []
        for i in pngFiles:
            i = pngPath + i
            fullPath.append(i)

        fourCC = cv2.VideoWriter_fourcc(*'avc1')

        img = Image.open(fullPath[0])
        size = list(img.size)
        print('creating ' + outVidName)

        video = cv2.VideoWriter(outputDir + outVidName, fourCC, 12, size)

        for i in range(len(fullPath)):
            video.write(cv2.imread(fullPath[i]))

        print('All done! You can find your video at ' + outputDir)
        video.release
    except:
        print('please select an output folder with an animation sequence')



imgCheck()
loadModel()

## Actions

dlg.scaleSlider.valueChanged.connect(setSliders)
dlg.stepSlider.valueChanged.connect(setSliders)
dlg.widthSlider.valueChanged.connect(setSliders)
dlg.strSlider.valueChanged.connect(setSliders)
dlg.zoomSlider.valueChanged.connect(setSliders)
dlg.xtransSlider.valueChanged.connect(setSliders)
dlg.ytransSlider.valueChanged.connect(setSliders)
dlg.rotSlider.valueChanged.connect(setSliders)
dlg.xturnSlider.valueChanged.connect(setSliders)
dlg.yturnSlider.valueChanged.connect(setSliders)
dlg.widthSlider.sliderReleased.connect(setWidthIntervals)
dlg.heightSlider.valueChanged.connect(setSliders)
dlg.heightSlider.sliderReleased.connect(setHeightIntervals)
upr.upresSlider.valueChanged.connect(setSliders)
dlg.outputButton.clicked.connect(setOutput)
upr.upresButton.clicked.connect(runUpres)
dlg.initCheck.stateChanged.connect(initCheck)
dlg.initButton.clicked.connect(initClicked)
dlg.genButton.setShortcut('Return')
dlg.updateKey = QShortcut(QKeySequence(Qt.Key_Enter),dlg)
dlg.darkCheck.stateChanged.connect(darkTheme)
dlg.promptEntry.setFocus()
dlg.checkDrop.currentIndexChanged.connect(loadModel)
dlg.precisionDrop.currentIndexChanged.connect(loadModel)
upr.upresDrop.currentIndexChanged.connect(upresCombo)
dlg.actionLoad_Prompt_From_Image.triggered.connect(loadPrompt)
dlg.actionLoad_Prompt_From_Image.setShortcut('Ctrl+O')
dlg.actionMakeVid.triggered.connect(outputVideo)
dlg.actionMakeVid.setShortcut('Ctrl+M')
dlg.actionClear_Viewer.setShortcut('Alt+C')
dlg.actionClear_Viewer.triggered.connect(clearPreview)
dlg.actionUpres_Image.setShortcut('Alt+S')
dlg.actionUpres_Image.triggered.connect(upresAction)
dlg.UpresLastButton.clicked.connect(upresLast)
dlg.imgTypeDrop.currentIndexChanged.connect(imgCheck)
dlg.genButton.clicked.connect(imgLoop)
dlg.animateButton.clicked.connect(animate)

grd.setWindowFlags(Qt.WindowTitleHint | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
grd.close()

dlg.show()
app.exec()
