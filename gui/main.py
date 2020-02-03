import sys
from PyQt4 import QtCore, QtGui

opened = False


class QDataViewer(QtGui.QWidget):
    def __init__(self):
        QtGui.QWidget.__init__(self)
        self.setGeometry(650, 300, 600, 600)
        self.setWindowTitle('File Upload')

        self.quitButton = QtGui.QPushButton('QUIT', self)
        self.uploadButton = QtGui.QPushButton('UPLOAD', self)

        self.hBoxLayout = QtGui.QHBoxLayout()
        self.hBoxLayout.addWidget(self.quitButton)
        self.hBoxLayout.addWidget(self.uploadButton)
        self.setLayout(self.hBoxLayout)

        # Signal Init.
        self.connect(self.quitButton,   QtCore.SIGNAL('clicked()'), QtGui.qApp, QtCore.SLOT('quit()'))
        self.connect(self.uploadButton, QtCore.SIGNAL('clicked()'), self.open)

    def open(self):
        self.file = QtGui.QFileDialog.getOpenFileName(self, 'Open File', '.')


def main():
    app = QtGui.QApplication(sys.argv)
    mw = QDataViewer()
    mw.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
