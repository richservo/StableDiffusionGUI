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
# import random
from math import floor, ceil
from PIL import Image
from qdarkstyle import load_stylesheet
from threading import Thread
from numpy import random
import sys
from resizeimage import resizeimage

QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True) #enable highdpi scaling
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True) #use highdpi icons

## set up gui elements

outputPath = ""
uiDir = '.\\UI\\'
modelDir = '.\\models\\ldm\\stable-diffusion-v1\\'
iconDir = '.\\UI\\icons\\'

dialogues = {'dlg': 'StableDiffusion.ui', 'grd': 'gridPreview.ui'}

app = QtWidgets.QApplication([])

for keys, values in dialogues.items():
    globals()[keys] = uic.loadUi(uiDir + values)

dlg.setWindowIcon(QtGui.QIcon(iconDir + 'StableDifusion.ico'))
dlg.scaleSlider.setMaximum(100)

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
# dlg.setMinimumWidth(defaultWidth)
# dlg.setMinimumHeight(defaultHeight)
dlg.sizeHint()
checkpoints = next(os.walk(modelDir))[-1]
checkpoints.sort(reverse = True)
dlg.sampleEntry.setText('1')
dlg.iterationEntry.setText('1')
dlg.widthSlider.setValue(512)
dlg.heightSlider.setValue(512)
dlg.strSlider.setValue(66)

for i in checkpoints:
    dlg.checkDrop.addItem(i)

def setSeed():
    seed = random.randint(42,429496729)
    dlg.seedEntry.setText(str(seed))

def setSliders():
    dlg.scaleValue.setText(str(dlg.scaleSlider.value()))
    dlg.stepValue.setText(str(dlg.stepSlider.value()))
    dlg.widthValue.setText(str(dlg.widthSlider.value()))
    dlg.heightValue.setText(str(dlg.heightSlider.value()))
    dlg.strValue.setText(str(float(dlg.strSlider.value() / 100)))

# def setEntry():
#     dlg.scaleSlider.setValue(int(dlg.scaleValue.text()))
#     dlg.stepSlider.setValue(int(dlg.stepValue.text()))
#     dlg.widthSlider.setValue(int(dlg.widthValue.text()))
#     dlg.heightSlider.setValue(int(dlg.heightValue.text()))
#     dlg.strSlider.setValue(int(dlg.strValue.text()))

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
            # outputPath = dlg.initEntry.text()
            flexWindow(outputPath)
        except Exception as e: print(e)


        dlg.initButton.setEnabled(True)
        # dlg.initEntry.setEnabled(True)
        # dlg.widthValue.setEnabled(False)
        # dlg.widthSlider.setEnabled(False)
        # dlg.heightValue.setEnabled(False)
        # dlg.heightSlider.setEnabled(False)
        dlg.strSlider.setVisible(True)
        dlg.strValue.setVisible(True)
        dlg.strLabel.setVisible(True)
        dlg.imgTypeDrop.setEnabled(True)
        defaultHeight = defaultHeight + 50
        # dlg.setMinimumHeight(defaultHeight)
        torch.cuda.empty_cache()

    else:
        dlg.precisionDrop.setCurrentIndex(1)
        dlg.initButton.setEnabled(False)
        # dlg.initEntry.setEnabled(False)
        dlg.widthValue.setEnabled(True)
        dlg.widthSlider.setEnabled(True)
        dlg.heightValue.setEnabled(True)
        dlg.heightSlider.setEnabled(True)
        dlg.strSlider.setVisible(False)
        dlg.strValue.setVisible(False)
        dlg.strLabel.setVisible(False)
        dlg.imgTypeDrop.setEnabled(False)
        dlg.initPreview.clear()
        defaultHeight = 600
        defaultWidth = 934
        # dlg.setMinimumWidth(defaultWidth + previewWidth)
        # if int(height) > defaultHeight:
        #     defaultHeight = abs((int(height) - defaultHeight) + int(height))-40
        #     dlg.setMinimumHeight(defaultHeight)
        # else:
        #     dlg.setMinimumHeight(defaultHeight)
        torch.cuda.empty_cache()

initCheck()

def initImage():
    global defaultWidth
    global defaultHeight
    global width
    global height
    global initImg
    global outputPath

    if dlg.initCheck.isChecked() == True:
        try:
            if outputPath == "":
                initImg = fileopenbox()
                initImg = initImg.replace('\\', '//')
            with open(initImg, 'r+b') as f:
                with Image.open(f) as image:
                    cover = resizeimage.resize_cover(image, [dlg.widthSlider.value(), dlg.heightSlider.value()])
                    cover.save('./resize.png')
            # init_image = Image.open(outputPath)
            # init_image = init_image.resize_contain((dlg.widthSlider.value(), dlg.heightSlider.value()))
            # init_image.save('resize.png')
            outputPath = "./resize.png"
            flexWindow(outputPath)



        except Exception as e: print(e)

        # setEntry()

def initClicked():
    global outputPath
    outputPath = ""
    initImage()

def flexWindow(outputPath):
    global defaultWidth
    global defaultHeight
    global width
    global height

    # dlg.initEntry.setText(outputPath)
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

    # dlg.setMinimumWidth(defaultWidth)

    # if int(height) > defaultHeight:
    #     defaultHeight = abs((int(height) - defaultHeight) + int(height)) + 40
    #     dlg.setMinimumHeight(height)
    # else:
    #     dlg.setMinimumHeight(defaultHeight)

def darkTheme():
    if dlg.darkCheck.isChecked() == True:
        dlg.setStyleSheet(load_stylesheet())
    else:
        dlg.setStyleSheet('Windows')

def generate():
    global previewWidth
    global previewHeight

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
            im = Image.open(outputPath)
            width, height = im.size
            img2img(device = device, model = model, prompt = prompt, seed = seed, init_img = initImage, ckpt = './models/ldm/stable-diffusion-v1/' + checkpoint, scale = scale,
                    ddim_steps = steps, n_iter = iterations, n_samples = samples, precision = precision, outdir = outputDir, n_rows = rows, strength = strength)

        previewFile = next(os.walk(outputDir + '//samples//'))[-1]
        previewFile.sort(reverse = True)
        previewFile = previewFile[0]
        preview = QPixmap(outputDir + '//samples//' + previewFile)

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

        # dlg.setMinimumWidth(defaultWidth + int(width)-40)
        dlg.sizeHint()
        # if int(height) > defaultHeight:
        #     dlg.setMinimumHeight(abs((int(height) - defaultHeight) + int(height))-40)
        # else:
        #     dlg.setMinimumHeight(defaultHeight)
        dlg.activateWindow()
        torch.cuda.empty_cache()
        previewWidth = width
        previewHeight = height

    except Exception as e: print(e)


    torch.cuda.empty_cache()

dlg.widthValue.setText('512')
dlg.heightValue.setText('512')
# setEntry()

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

def loadPrompt():
    try:
        prompt = fileopenbox()
        im = Image.open(prompt)
        preview = prompt.replace('\\', '//')
        flexWindow(outputPath = preview)
        # preview = QPixmap(preview)
        # dlg.imgPreview.setPixmap(preview)
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
            index = dlg.checkDrop.findText(im.text['ckpt'].split('/')[-1], QtCore.Qt.MatchFixedString)
            dlg.checkDrop.setCurrentIndex(index)
        except:
            pass
    except:
        pass


def imgCheck():
    # if dlg.imgTypeDrop.currentText() == 'still':
    dlg.initButton.setText('Init Image')
    dlg.initButton.setIcon(QIcon(iconDir + 'img.png'))
    # else:
    #     dlg.initButton.setText('Init Directory')
    #     dlg.initButton.setIcon(QIcon(iconDir + 'folder.png'))

def imgLoop():
    global outputPath
    if dlg.imgTypeDrop.currentText() == 'still':
        if dlg.iterationEntry.text() == 1:
            generate()
        else:
            for i in range(int(dlg.iterationEntry.text())):
                generate()
    else:
        # try:
        img = outputPath.replace('\\', '/')
        imageList = next(os.walk('/'.join(img.split('/')[0:-1])))[-1]
        imageDir = next(os.walk('/'.join(img.split('/')[0:-1])))[0]
        print(imageList)
        print(imageDir)
        for i in imageList:
            if '.png' not in i:
                imageList.remove(i)

        for i in imageList:
            outputPath = imageDir + '\\' + i
            initImage()
            # dlg.initEntry.setText(outputPath)
            QApplication.processEvents()
            initImg = QPixmap(outputPath.replace('\\', '/'))
            dlg.initPreview.setPixmap(initImg)
            generate()

        # except Exception as e: print(e)

def clearPreview():
    dlg.initPreview.clear()
    dlg.imgPreview.clear()
    # dlg.setMinimumWidth(fixedWidth)
    # dlg.setMaximumWidth(defaultWidth)
    # dlg.setMinimumHeight(fixedHeight)
    # dlg.setMaximumHeight(defaultWidth)
    if os.path.exists('./resize'):
        os.remove('./resize')

imgCheck()
loadModel()
## Actions

dlg.scaleSlider.valueChanged.connect(setSliders)
dlg.stepSlider.valueChanged.connect(setSliders)
dlg.widthSlider.valueChanged.connect(setSliders)
dlg.strSlider.valueChanged.connect(setSliders)
dlg.widthSlider.sliderReleased.connect(setWidthIntervals)
dlg.heightSlider.valueChanged.connect(setSliders)
dlg.heightSlider.sliderReleased.connect(setHeightIntervals)
dlg.outputButton.clicked.connect(setOutput)
dlg.initCheck.stateChanged.connect(initCheck)
dlg.initButton.clicked.connect(initClicked)
dlg.genButton.setShortcut('Return')
dlg.updateKey = QShortcut(QKeySequence(Qt.Key_Enter),dlg)
# dlg.updateKey.activated.connect(setEntry)
dlg.darkCheck.stateChanged.connect(darkTheme)
dlg.promptEntry.setFocus()
dlg.checkDrop.currentIndexChanged.connect(loadModel)
dlg.precisionDrop.currentIndexChanged.connect(loadModel)
dlg.actionLoad_Prompt_From_Image.triggered.connect(loadPrompt)
dlg.actionLoad_Prompt_From_Image.setShortcut('Ctrl+O')
dlg.actionClear_Viewer.setShortcut('Alt+C')
dlg.actionClear_Viewer.triggered.connect(clearPreview)
dlg.imgTypeDrop.currentIndexChanged.connect(imgCheck)
dlg.genButton.clicked.connect(imgLoop)

grd.setWindowFlags(Qt.WindowTitleHint | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
grd.close()



dlg.show()
app.exec()
