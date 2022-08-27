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
from qdarkstyle import load_stylesheet
from threading import Thread


## set up gui elements

uiDir = '.\\UI\\'
modelDir = '.\\models\\ldm\\stable-diffusion-v1\\'
iconDir = '.\\UI\\icons\\'

dialogues = {'dlg': 'StableDiffusion.ui', 'grd': 'gridPreview.ui'}

app = QtWidgets.QApplication([])

for keys, values in dialogues.items():
    globals()[keys] = uic.loadUi(uiDir + values)

dlg.setWindowIcon(QtGui.QIcon(iconDir + 'StableDifusion.ico'))

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
dlg.setFixedWidth(defaultWidth)
dlg.setFixedHeight(defaultHeight)
checkpoints = next(os.walk(modelDir))[-1]
checkpoints.sort(reverse = True)
dlg.sampleEntry.setText('1')
dlg.iterationEntry.setText('1')


for i in checkpoints:
    dlg.checkDrop.addItem(i)

def setSeed():
    seed = random.randint(42,4294967295)
    dlg.seedEntry.setText(str(seed))

def setSliders():
    dlg.scaleValue.setText(str(dlg.scaleSlider.value()))
    dlg.stepValue.setText(str(dlg.stepSlider.value()))
    dlg.widthValue.setText(str(dlg.widthSlider.value()))
    dlg.heightValue.setText(str(dlg.heightSlider.value()))
    dlg.strValue.setText(str(dlg.strSlider.value()))

def setEntry():
    dlg.scaleSlider.setValue(int(dlg.scaleValue.text()))
    dlg.stepSlider.setValue(int(dlg.stepValue.text()))
    dlg.widthSlider.setValue(int(dlg.widthValue.text()))
    dlg.heightSlider.setValue(int(dlg.heightValue.text()))
    dlg.strSlider.setValue(int(dlg.strValue.text()))

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

def setHeightIntervals():
    result = next(
    number
    for distance in count()
    for number in (dlg.heightSlider.value() + distance, dlg.heightSlider.value() - distance)
    if number % 64 == 0
    )
    dlg.heightSlider.setValue(result)

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
    if dlg.initCheck.isChecked() == True:
        dlg.initButton.setEnabled(True)
        dlg.initEntry.setEnabled(True)
        dlg.widthValue.setEnabled(False)
        dlg.widthSlider.setEnabled(False)
        dlg.heightValue.setEnabled(False)
        dlg.heightSlider.setEnabled(False)
        dlg.strSlider.setVisible(True)
        dlg.strValue.setVisible(True)
        dlg.strLabel.setVisible(True)
        dlg.imgTypeDrop.setEnabled(True)
        defaultHeight = defaultHeight + 50
        dlg.setFixedHeight(defaultHeight)
        torch.cuda.empty_cache()


    else:
        dlg.precisionDrop.setCurrentIndex(1)
        dlg.initButton.setEnabled(False)
        dlg.initEntry.setEnabled(False)
        dlg.widthValue.setEnabled(True)
        dlg.widthSlider.setEnabled(True)
        dlg.heightValue.setEnabled(True)
        dlg.heightSlider.setEnabled(True)
        dlg.strSlider.setVisible(False)
        dlg.strValue.setVisible(False)
        dlg.strLabel.setVisible(False)
        defaultHeight = 600
        dlg.setFixedHeight(defaultHeight)
        torch.cuda.empty_cache()

initCheck()

def initImage():
    if dlg.imgTypeDrop.currentText() == 'still':
        try:
            outputPath = fileopenbox()
            dlg.initEntry.setText(outputPath)
            im = Image.open(dlg.initEntry.text())
            width, height = im.size
            dlg.widthValue.setText(str(width))
            dlg.heightValue.setText(str(height))
            #preview = QPixmap(outputPath)
            #dlg.initPreview.setPixmap(preview)
        except:
            pass
    if dlg.imgTypeDrop.currentText() == 'sequence':
        try:
            outputPath = diropenbox()
            dlg.initEntry.setText(outputPath)
            im = Image.open(dlg.initEntry.text())
            width, height = im.size
            dlg.widthValue.setText(str(width))
            dlg.heightValue.setText(str(height))
            #preview = QPixmap(outputPath)
            #dlg.initPreview.setPixmap(preview)
        except:
            pass

    setEntry()

def darkTheme():
    if dlg.darkCheck.isChecked() == True:
        dlg.setStyleSheet(load_stylesheet())
    else:
        dlg.setStyleSheet('Windows')

def generate():


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
    strength = float("{:.2}".format(float(10/int(dlg.strValue.text()))))

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

    if dlg.initEntry.text() == "":
        pass
    else:
        initImage = str(dlg.initEntry.text())
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
            im = Image.open(dlg.initEntry.text())
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
            grd.setFixedWidth(int(width)*int(rows))
            grd.setFixedHeight(int(height) * ceil((abs((int(dlg.sampleEntry.text()) * int(dlg.iterationEntry.text()))) / int(rows))))
            previewFile = next(os.walk(outputDir))[-1]
            previewFile.sort(reverse = True)
            previewFile = previewFile[0]
            preview = QPixmap(outputDir + previewFile)
            grd.gridPreview.setPixmap(preview)
            grd.setWindowTitle(prompt)
            grd.activateWindow()
            grd.show()

        dlg.setFixedWidth(defaultWidth + int(width)-40)

        if int(height) > defaultHeight:
            dlg.setFixedHeight(abs((int(height) - defaultHeight) + int(height))-40)
        else:
            dlg.setFixedHeight(defaultHeight)
        dlg.activateWindow()
        torch.cuda.empty_cache()

    except:
        torch.cuda.empty_cache()
        print("\n" + "Looks like you're doing too much there cowboy!" + "\n" + "Try lowering some stuff and not breaking things!")

    torch.cuda.empty_cache()

dlg.widthValue.setText('512')
dlg.heightValue.setText('512')
setEntry()

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
    prompt = fileopenbox()
    im = Image.open(prompt)
    width, height = im.size
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
    if dlg.imgTypeDrop.currentText() == 'still':
        dlg.initButton.setText('Init Image')
        dlg.initButton.setIcon(QIcon(iconDir + 'img.png'))
    else:
        dlg.initButton.setText('Init Directory')
        dlg.initButton.setIcon(QIcon(iconDir + 'folder.png'))

def imgLoop():
    if dlg.imgTypeDrop.currentText() == 'still':
        if dlg.iterationEntry.text() == 1:
            generate()
        else:
            for i in range(int(dlg.iterationEntry.text())):
                generate()
    else:
        try:
            imageList = next(os.walk(dlg.initEntry.text()))[-1]
            imageDir = dlg.initEntry.text()
            for i in imageList:
                if '.png' not in i:
                    imageList.remove(i)

            for i in imageList:
                dlg.initEntry.setText(imageDir + '\\' + i)
                generate()

        except:
            print('\n' + 'Sorry, try a directory with images')





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
dlg.initButton.clicked.connect(initImage)
dlg.genButton.setShortcut('Return')
dlg.updateKey = QShortcut(QKeySequence(Qt.Key_Enter),dlg)
dlg.updateKey.activated.connect(setEntry)
dlg.darkCheck.stateChanged.connect(darkTheme)
dlg.promptEntry.setFocus()
dlg.checkDrop.currentIndexChanged.connect(loadModel)
dlg.precisionDrop.currentIndexChanged.connect(loadModel)
dlg.actionLoad_Prompt_From_Image.triggered.connect(loadPrompt)
dlg.actionLoad_Prompt_From_Image.setShortcut('Ctrl+O')
dlg.imgTypeDrop.currentIndexChanged.connect(imgCheck)
dlg.genButton.clicked.connect(imgLoop)

grd.setWindowFlags(Qt.WindowTitleHint | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
grd.close()

dlg.show()
app.exec()
