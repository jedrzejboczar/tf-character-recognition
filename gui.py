import sys
import string

import numpy as np
import tensorflow as tf
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTextEdit, QSlider
from PyQt5.QtGui import QPainter, QImage, QPen, QFont, QFontDatabase, QColor
from PyQt5.QtCore import QSize, QRect, pyqtSignal, pyqtSlot
from PyQt5.Qt import Qt, QSizePolicy
import qimage2ndarray

import data


NET_IMAGE_SIZE = QSize(*data.Database.IMAGE_SIZE)
GUI_IMAGE_SIZE = QSize(350, 350)
PEN_WIDTH_RANGE = (1, 60)
IMAGE_RGB = False

class Gui(QWidget):
    """
    Main window of graphical user interface for letter recognition.
    """
    def __init__(self, classifier):
        super().__init__()
        self.setWindowTitle('Letter recognition')
        # self.setGeometry(QRect)
        self.drawing_box = DrawingBox(self)
        self.drawing_viewer = DrawingViewer(self)
        self.clear_button = QPushButton('Clear', self)
        self.invert_button = QPushButton('Invert', self)
        self.pen_width_slider = QSlider(Qt.Horizontal, self)
        self.pen_width_slider.setRange(*PEN_WIDTH_RANGE)
        self.pen_width_slider.setValue(self.drawing_box.pen.width())
        self.pen_width_value_diplay = QLabel(str(self.pen_width_slider.value()), self)
        self.pen_width_value_diplay.setMinimumWidth(30)
        self.letter_recognition_box = LetterRecognitionBox(classifier, self)
        self.connectSignals()
        self.createLayout()

    def connectSignals(self):
        self.clear_button.clicked.connect(self.drawing_box.clearImage)
        self.invert_button.clicked.connect(self.drawing_box.invert)
        self.pen_width_slider.valueChanged.connect(self.drawing_box.setPenWidth)
        self.pen_width_slider.valueChanged.connect(self.pen_width_value_diplay.setNum)
        self.drawing_box.newImage.connect(self.drawing_viewer.showImage)
        self.drawing_box.newImage.connect(self.letter_recognition_box.evaluateImage)

    def createLayout(self):
        drawing_layout = QHBoxLayout()
        drawing_layout.addWidget(self.drawing_box)
        drawing_layout.addWidget(self.drawing_viewer)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.clear_button)
        buttons_layout.addWidget(self.invert_button)
        buttons_layout.addWidget(self.pen_width_slider)
        buttons_layout.addWidget(self.pen_width_value_diplay)

        info_layout = QVBoxLayout()
        info_layout.addLayout(buttons_layout)
        info_layout.addWidget(self.letter_recognition_box)
        self.clear_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.invert_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.letter_recognition_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        main_layout = QVBoxLayout()
        main_layout.addLayout(drawing_layout)
        main_layout.addLayout(info_layout)

        self.setLayout(main_layout)
        self.show()


class LetterRecognitionBox(QTextEdit):
    """
    Connects to the Tensorflow model of letter recognition neural network,
    evaluates images drawn by user and displays results.
    """
    def __init__(self, classifier, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classifier = classifier
        self.tf_session = tf.Session()
        self.setReadOnly(True)
        self.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont)) # monospace
        self.setFontPointSize(10)
        self.setInfo()

    def sizeHint(self):
        return QSize(256, 128)

    def setInfo(self, n=5, logits=[], probabilities=[], letters=[]):
        logits_str = '   '.join('{:^5.1f}'.format(num) for num in logits[:n])
        probabilities_str = '   '.join('{:^5.1f}'.format(100 * num) for num in probabilities[:n])
        letters_str = '   '.join('{:^5}'.format(letter) for letter in letters[:n])
        prediction_str = ''
        if letters and probabilities:
            prediction_str = '{} ({:.2f} %)'.format(letters[0], 100 * probabilities[0])
        text = """
   Prediction:  {}

      Letters:  {}
Probabilities:  {}
       Logits:  {}
        """.strip('\n').format(prediction_str, letters_str, probabilities_str, logits_str)
        self.setText(text)

    def showPredictions(self, predictions):
        top_indices = predictions['top_indices']
        letters = [data.Database.LABELS[i] for i in top_indices]
        logits = [predictions['logits'][i] for i in top_indices]
        probabilities = [predictions['probabilities'][i] for i in top_indices]
        self.setInfo(5, logits, probabilities, letters)

    @pyqtSlot(QImage)
    def evaluateImage(self, image):
        image = qimage2ndarray.rgb_view(image).astype(np.float32)  # 3D array (RGB)
        if not IMAGE_RGB:
            # do not use this, just take one color, as they are the same in this array
            # image = 0.2126 * image['r'] + 0.7152 * image['g'] + 0.0722 * image['b']
            image = image[:, :, :1]
        images = np.array([image, ])
        input_fn = lambda: tf.data.Dataset.from_tensor_slices(images).batch(1)
        predictions, = self.classifier.predict(input_fn)
        self.showPredictions(predictions)


class DrawingBox(QWidget):
    """
    Allows to draw an image of a letter. After drawing, sends signal containing
    image rescaled to the input size of the neural network.
    """
    # signal emitted when user releases mouse after drawing
    newImage = pyqtSignal(QImage)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.imageSize = GUI_IMAGE_SIZE
        self.setFixedSize(self.imageSize)
        self.inverted = False
        self.pen = QPen(Qt.white if self.inverted else Qt.black,
            int(np.mean(PEN_WIDTH_RANGE)), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        self.image = QImage(self.imageSize, QImage.Format_Grayscale8)
        self.clearImage()
        self.last_pos = None
        self.drawing = False

    @pyqtSlot(int)
    def setPenWidth(self, pen_width):
        self.pen.setWidth(pen_width)

    @pyqtSlot()
    def invert(self):
        self.inverted = not self.inverted
        self.pen.setColor(Qt.white if self.inverted else Qt.black)
        self.image.invertPixels()
        self.update()
        self.pushImage()

    def pushImage(self):
        image = self.image.scaled(NET_IMAGE_SIZE, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.newImage.emit(image)

    @pyqtSlot()
    def clearImage(self):
        self.image.fill(0 if self.inverted else 255)
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_pos = event.pos()
            self.drawing = True

    def mouseMoveEvent(self, event):
        dist = (event.pos() - self.last_pos).manhattanLength()
        if (event.buttons() & Qt.LeftButton) and self.drawing and dist > 3:
            self.drawLineTo(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawLineTo(event.pos())
            self.drawing = False
            self.pushImage()

    def paintEvent(self, event):
        "Paints changes in the image on the widget."
        painter = QPainter(self)
        dirty_rect = event.rect()
        painter.drawImage(dirty_rect, self.image, dirty_rect)

    def drawLineTo(self, end_pos):
        "Draws line on the image and signals which part is dirty."
        painter = QPainter(self.image)
        painter.setPen(self.pen)
        painter.drawLine(self.last_pos, end_pos)
        radius = int((self.pen.widthF() / 2) + 2)
        self.update(QRect(self.last_pos, end_pos).normalized().adjusted(
            -radius, -radius, radius, radius))
        self.last_pos = end_pos


class DrawingViewer(QWidget):
    """
    Displays image drawn by user in the form that is passed to the neural network.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.imageSize = GUI_IMAGE_SIZE
        self.setFixedSize(self.imageSize)
        self.image = QImage(self.imageSize, QImage.Format_Grayscale8)
        self.image.fill(255)
        self.update()

    @pyqtSlot(QImage)
    def showImage(self, image):
        # self.image = image.scaled(GUI_IMAGE_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image = image.scaled(GUI_IMAGE_SIZE)
        self.update()

    def paintEvent(self, event):
        "Paints changes in the image on the widget."
        painter = QPainter(self)
        dirty_rect = event.rect()
        painter.drawImage(dirty_rect, self.image, dirty_rect)


def runApp(estimator):
    app = QApplication(sys.argv)
    gui = Gui(estimator)
    sys.exit(app.exec_())

if __name__ == '__main__':
    print('Estimator needed', file=sys.stderr)
