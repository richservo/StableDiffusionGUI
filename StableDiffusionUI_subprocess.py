from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QApplication, QFileDialog, QShortcut
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import QDate, QTime, QDateTime, Qt, QSize
from PyQt5.QtGui import QPainter, QKeySequence
from itertools import count
from subprocess import call, PIPE, run
import easygui
import os
import random
from math import floor, ceil
from PIL import Image
from qdarkstyle import load_stylesheet


## set up gui elements

uiDir = '.\\UI\\'
modelDir = '.\\models\\ldm\\stable-diffusion-v1\\'
iconDir = '.\\UI\\icons\\'

dialogues = {'dlg': 'StableDiffusion.ui', 'grd': 'gridPreview.ui'}
    
app = QtWidgets.QApplication([])

for keys, values in dialogues.items():
    globals()[keys] = uic.loadUi(uiDir + values)

dlg.setWindowIcon(QtGui.QIcon(iconDir + 'StableDifusion.ico'))

## initial setup
defaultWidth = 934
defaultHeight = 600
dlg.setFixedWidth(defaultWidth)
dlg.setFixedHeight(defaultHeight)
checkpoints = next(os.walk(modelDir))[-1]
checkpoints.sort(reverse = True)
dlg.sampleEntry.setText('1')
dlg.iterationEntry.setText('1')
dlg.outputButton.setIcon(QIcon(iconDir + 'folder.png'))
dlg.initButton.setIcon(QIcon(iconDir + 'img.png'))
dlg.strSlider.setVisible(False)
dlg.strValue.setVisible(False)
dlg.strLabel.setVisible(False)



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
        outputPath = easygui.diropenbox()
        dlg.outputEntry.setText(outputPath)
    except:
        pass
    

## Action functions

def initCheck():
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

    else:
        dlg.initButton.setEnabled(False)
        dlg.initEntry.setEnabled(False)
        dlg.widthValue.setEnabled(True)
        dlg.widthSlider.setEnabled(True)
        dlg.heightValue.setEnabled(True)
        dlg.heightSlider.setEnabled(True)
        dlg.strSlider.setVisible(False)
        dlg.strValue.setVisible(False)
        dlg.strLabel.setVisible(False)

initCheck()

def initImage():
    try:
        outputPath = easygui.fileopenbox()
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
    ## set variables
    if dlg.seedCheck.isChecked() == False:
        setSeed()
    else:
        pass
    prompt = str(dlg.promptEntry.text())
    if prompt == '':
        prompt = 'photo still of danny trejo looking into the camera very disapointedly, 8k, 85mm f1.8'
    scale = str(dlg.scaleSlider.value())
    steps = str(dlg.stepSlider.value())
    width = str(dlg.widthValue.text())
    height = str(dlg.heightValue.text())
    checkpoint = str(dlg.checkDrop.currentText())
    formatNumber = lambda n: n if n%1 else int(n)
    rows = abs((int(dlg.sampleEntry.text()) * int(dlg.iterationEntry.text())/2))
    rows = str(formatNumber(rows))
    strength = str("{:.2}".format(float(10/int(dlg.strValue.text()))))
    
    if float(rows)//1 != float(rows)/1:
        if float(rows) == .5:
            rows = str(ceil(float(rows)))
        else:
            rows = str(floor(float(rows)))
    
    if dlg.modelDrop.currentText() == 'plms':
        model = '--plms'
    else:
        model = ''
    precision = str(dlg.precisionDrop.currentText())
    seed = str(dlg.seedEntry.text())
    samples = str(dlg.sampleEntry.text())
    iterations = str(dlg.iterationEntry.text())
    
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
    if dlg.initCheck.isChecked() == False:
        if dlg.modelDrop.currentText() == 'plms':
            call(['python', './scripts/txt2img.py', model, '--prompt', prompt, '--seed', seed, '--ckpt',
                  './models/ldm/stable-diffusion-v1/' + checkpoint, '--scale', scale, '--ddim_steps', steps,
                  '--n_iter', iterations, '--n_samples', samples, '--W', width, '--H', height, '--precision',
                  precision, '--outdir', outputDir, '--n_rows', rows]
                 )
        else:
            call(['python', './scripts/txt2img.py', '--prompt', prompt, '--seed', seed, '--ckpt',
                  './models/ldm/stable-diffusion-v1/' + checkpoint, '--scale', scale, '--ddim_steps', steps,
                  '--n_iter', iterations, '--n_samples', samples, '--W', width, '--H', height, '--precision',
                  precision, '--outdir', outputDir, '--n_rows', rows]
                 )

    else:
        im = Image.open(dlg.initEntry.text())
        width, height = im.size
        call(['python', './scripts/img2img.py', '--prompt', prompt, '--init-img', initImage, '--seed', seed, '--ckpt',
              './models/ldm/stable-diffusion-v1/' + checkpoint, '--scale', scale, '--ddim_steps', steps,
              '--n_iter', iterations, '--n_samples', samples, '--precision',
                precision, '--outdir', outputDir, '--n_rows', rows, '--strength', strength]
                 )
        

    previewFile = next(os.walk(outputDir + '//samples//'))[-1]
    previewFile.sort(reverse = True)
    previewFile = previewFile[0]
    preview = QPixmap(outputDir + '//samples//' + previewFile)
    
    dlg.imgPreview.setPixmap(preview)

    if abs((int(dlg.sampleEntry.text()) * int(dlg.iterationEntry.text()))) > 1:
        grd.setFixedWidth(int(width)*int(rows))
        grd.setFixedHeight(int(height) * ceil((abs((int(dlg.sampleEntry.text()) * int(dlg.iterationEntry.text()))) / int(rows)))                           )
        previewFile = next(os.walk(outputDir))[-1]
        previewFile.sort(reverse = True)
        previewFile = previewFile[0]
        preview = QPixmap(outputDir + previewFile)
        grd.gridPreview.setPixmap(preview)
        grd.setWindowTitle(prompt)
        grd.show()
        grd.activateWindow()

    dlg.setFixedWidth(defaultWidth + int(width)-40)

    if int(height) > defaultHeight:
        dlg.setFixedHeight(abs((int(height) - defaultHeight) + int(height))-40)
    else:
        dlg.setFixedHeight(defaultHeight)
    dlg.activateWindow()

dlg.widthValue.setText('512')
dlg.heightValue.setText('512')
setEntry()





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
dlg.genButton.clicked.connect(generate)
dlg.genButton.setShortcut('Return')
dlg.updateKey = QShortcut(QKeySequence(Qt.Key_Enter),dlg)
dlg.updateKey.activated.connect(setEntry)
dlg.darkCheck.stateChanged.connect(darkTheme)
dlg.promptEntry.setFocus()





dlg.show()
app.exec()
