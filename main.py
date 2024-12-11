import pickle
import re
import sys
from PyQt5.QtWidgets import QMessageBox, QApplication, QFileDialog, QWidget
import numpy as np
from GUI import Ui_StructualResponsePrediction
from functools import partial
import sys
import pickle
import re
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import matplotlib
import tensorflow as tf
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib import pyplot
 
pyplot.rcParams['font.sans-serif'] = ['Times New Roman']
pyplot.rcParams['axes.unicode_minus'] = False

class MyFigure(FigureCanvasQTAgg):
   def __init__(self,width=5,height=4,dpi = 600):
      self.fig = Figure(figsize=(width,height),dpi=dpi)
      super(MyFigure, self).__init__(self.fig)
 
   def plot(self,x,y):
      self.axes0 = self.fig.add_subplot(111)
      self.axes0.plot(x,y)

class MyUiComputer(Ui_StructualResponsePrediction, QWidget):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.select_1.clicked.connect(self.open_file_dialog1)
        self.select_2.clicked.connect(self.open_file_dialog2)
        self.PhyCNN.toggled.connect(self.update_plot)
        self.CNN.toggled.connect(self.update_plot)
        self.DNN.toggled.connect(self.update_plot)
        self.LSTM.toggled.connect(self.update_plot)
        self.Predict.clicked.connect(partial(self.FRSPredict))
        self.SaveR.clicked.connect(partial(self.Savetxt))
        self.SaveF.clicked.connect(partial(self.save_Output_image))

    def open_file_dialog1(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open File', '', 'All Files (*);;Text Files (*.txt)')
        if file_name:
            with open(file_name, 'r') as file:
                file_content = file.read()
                try:
                    dt = 0.02
                    GM = np.array(re.sub(r"\\s+", " ", file_content).split())
                    GM = GM.astype(float)
                    F1 = MyFigure(width=5, height=4, dpi=100)
                    npts = len(GM)
                    t = np.arange(0, dt*npts, dt)
                    F1.axes1 = F1.fig.add_subplot(111)
                    F1.axes1.plot(t, GM)
                    # F1.axes1.set_title("Result")
                    F1.axes1.set_ylim(np.min(GM)-0.5, np.max(GM)+0.5)
                    F1.axes1.set_xticks(np.arange(0, dt*npts, 10))
                    F1.axes1.set_yticks(np.linspace(np.ceil(np.min(GM)-0.5), np.ceil(np.max(GM)+0.5), 6))
                    F1.axes1.set_xlabel("t (s)")
                    F1.axes1.set_ylabel("Acc (m/s$^{2}$)")
                    width,height = self.graphicsView_G.width(),self.graphicsView_G.height()
                    F1.resize(width*0.9,height*0.8)
                    F1.fig.subplots_adjust(bottom=0.2, left=0.2)

                    self.scene = QGraphicsScene()
                    self.scene.addWidget(F1)
                    self.graphicsView_G.setScene(self.scene)
                    self.GM = GM
                except Exception as e:
                    QMessageBox.warning(self, 'Warning', f"Error:{e}.", QMessageBox.Ok)

    def open_file_dialog2(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open File', '', 'All Files (*);;Text Files (*.txt)')
        if file_name:
            with open(file_name, 'r') as file:
                file_content = file.read()
                try:
                    dt = 0.02
                    RM = np.array(re.sub(r"\\s+", " ", file_content).split())
                    RM = RM.astype(float)
                    F1 = MyFigure(width=5, height=4, dpi=100)
                    npts = len(RM)
                    t = np.arange(0, dt*npts, dt)
                    F1.axes1 = F1.fig.add_subplot(111)
                    F1.axes1.plot(t, RM)
                    # F1.axes1.set_title("Result")
                    F1.axes1.set_ylim(np.min(RM)-0.5, np.max(RM)+0.5)
                    F1.axes1.set_xticks(np.arange(0, dt*npts, 10))
                    F1.axes1.set_yticks(np.linspace(np.ceil(np.min(RM)-0.5), np.ceil(np.max(RM)+0.5), 6))
                    F1.axes1.set_xlabel("t (s)")
                    F1.axes1.set_ylabel("Acc (m/s$^{2}$)")
                    width,height = self.graphicsView_R.width(),self.graphicsView_R.height()
                    F1.resize(width*0.9,height*0.8)
                    F1.fig.subplots_adjust(bottom=0.2, left=0.2)

                    self.scene = QGraphicsScene()
                    self.scene.addWidget(F1)
                    self.graphicsView_R.setScene(self.scene)
                    self.RM = RM
                    self.t=t
                    self.npts=npts
                except Exception as e:
                    QMessageBox.warning(self, 'Warning', f"Error:{e}.", QMessageBox.Ok)


    def FRSPredict(self,window):

        def reshape_and_pad(array):
            arr = np.array(array)
            
            num_elements = arr.size
            remainder = num_elements % 100
            if remainder != 0:
                padding_size = 100 - remainder
                arr = np.pad(arr, (0, padding_size), mode='constant', constant_values=0)
            reshaped_array = arr.reshape(-1, 100)
        
            return reshaped_array

        try:
            N1 = self.N.text()
            Lt1 = self.Lt.text()
            Ll1 = self.Ll.text() 
            H1 = self.H.text()
            h1 = self.h.text()
            zh1 = self.zh.text()
            B1 = self.B.text()

            N1 = float(N1)
            Lt1 = float(Lt1)
            Ll1 = float(Ll1)
            H1 = float(H1)
            h1 = float(h1)

            textbox_values = [
                (N1 - 19.044) / 18.727,
                (Lt1 - 34.567) / 14.984,
                (Ll1 - 60.503) / 20.447,
                (H1 - 71.485) / 73.138,
                (h1 - 4.02) / 1.184,
                float(zh1),
                float(B1)
            ]
            # textbox_values = [(N1-19.044)/18.727, (Lt1-34.567)/14.984, (Ll1-60.503)/20.447, (H1-71.485)/73.138, (h1-4.02)/1.184, zh1, B1]
            I1 = self.I.currentText()
            FX1 = self.F.currentText()
            if any([not N1, not Lt1, not Ll1, not H1, not zh1, not B1]):
                QMessageBox.warning(self, 'Warning', 'Please enter a value for all fields.', QMessageBox.Ok)
                return
            textbox_values = [float(x) for x in textbox_values]
            if FX1 == "transverse":
                textbox_values.append(1)
            else:
                textbox_values.append(2)

            if I1 == "concrete":
                textbox_values.extend([1, 0, 0])
            elif I1 == "steel":
                textbox_values.extend([0, 1, 0])
            else:
                textbox_values.extend([0, 0, 1])

            loaded_modelPhy = tf.keras.models.load_model("PhyCNN.h5",compile=False)
            loaded_modelCNN = tf.keras.models.load_model("CNN.h5",compile=False)
            loaded_modelDNN = tf.keras.models.load_model("DNN.h5",compile=False)
            loaded_modelLSTM = tf.keras.models.load_model("LSTM.h5",compile=False)

            GM100 = reshape_and_pad(self.GM)
            RM100 = reshape_and_pad(self.RM)
            l=GM100.shape[0]           
            Stru = np.tile(textbox_values, (l, 1))

            Acc_Phy = loaded_modelPhy.predict([GM100,RM100,Stru])
            Acc_CNN = loaded_modelCNN.predict([GM100,RM100,Stru])
            Acc_DNN = loaded_modelDNN.predict([GM100,RM100,Stru])
            Acc_LSTM = loaded_modelLSTM.predict([GM100,RM100,Stru])
            pred_Phy=[]
            pred_CNN=[]
            pred_DNN=[]
            pred_LSTM=[]
            for i in range(0, l):
                pred_Phy.append(Acc_Phy[i, :100])
                pred_CNN.append(Acc_CNN[i, :100])
                pred_DNN.append(Acc_DNN[i, :100])
                pred_LSTM.append(Acc_LSTM[i, :100])
                y_pred_Phy = np.concatenate(pred_Phy)
                y_pred_CNN = np.concatenate(pred_CNN)
                y_pred_DNN = np.concatenate(pred_DNN)
                y_pred_LSTM = np.concatenate(pred_LSTM)
            y_pred_Phy = y_pred_Phy.reshape(-1)
            y_pred_CNN = y_pred_CNN.reshape(-1)
            # y_pred_Phy = y_pred_Phy[:,-pdG]
            # y_pred_CNN = y_pred_CNN[:,-pdG]
            # y_pred_DNN = y_pred_DNN[:,-pdG]
            # y_pred_LSTM = y_pred_LSTM[:,-pdG]
            # print(y_pred_Phy.shape)
            # print(y_pred_CNN.shape)
            # print(y_pred_DNN.shape)
            # print(y_pred_LSTM.shape)
            self.y_pred_Phy = y_pred_Phy
            self.y_pred_CNN = y_pred_CNN
            self.y_pred_DNN = y_pred_DNN
            self.y_pred_LSTM = y_pred_LSTM

            self.N1=N1
            self.Ll1=Ll1
            self.Lt1=Lt1
            self.H1=H1
            self.h1=h1
            self.zh1=zh1
            self.B1=B1
            self.I1=I1
            self.FX1=FX1

        except Exception as e:
            QMessageBox.warning(self, 'Warning', f"Error:{e}.", QMessageBox.Ok)

        Acc_Phy_list = list(y_pred_Phy)
        Acc_CNN_list = list(y_pred_CNN)
        Acc_DNN_list = list(y_pred_DNN)
        Acc_LSTM_list = list(y_pred_LSTM)

        cols_num = max(len(Acc_Phy_list), len(Acc_CNN_list), len(Acc_DNN_list), len(Acc_LSTM_list))

        self.tableWidget.setColumnCount(cols_num)

        for i in range(cols_num):
            Acc_Phy2 = "{:.3f}".format(Acc_Phy_list[i]) if i < len(Acc_Phy_list) else "0.000"
            Acc_CNN2 = "{:.3f}".format(Acc_CNN_list[i]) if i < len(Acc_CNN_list) else "0.000"
            Acc_DNN2 = "{:.3f}".format(Acc_DNN_list[i]) if i < len(Acc_DNN_list) else "0.000"
            Acc_LSTM2 = "{:.3f}".format(Acc_LSTM_list[i]) if i < len(Acc_LSTM_list) else "0.000"

            item_Acc_Phy = QTableWidgetItem(Acc_Phy2)
            item_Acc_CNN = QTableWidgetItem(Acc_CNN2)
            item_Acc_DNN = QTableWidgetItem(Acc_DNN2)
            item_Acc_LSTM = QTableWidgetItem(Acc_LSTM2)

            self.tableWidget.setItem(0, i, item_Acc_Phy)
            self.tableWidget.setItem(1, i, item_Acc_CNN)
            self.tableWidget.setItem(2, i, item_Acc_DNN)
            self.tableWidget.setItem(3, i, item_Acc_LSTM)

    def update_plot(self):
        F1 = MyFigure(width=5, height=4, dpi=100)
        self.F1=F1
        F1.axes1 = F1.fig.add_subplot(111)
        if self.PhyCNN.isChecked():
            F1.axes1.plot(self.t, self.y_pred_Phy, color='blue', linestyle='-', linewidth=2, label='PhyCNN')
        if self.CNN.isChecked():
            F1.axes1.plot(self.t, self.y_pred_CNN, color='red', linestyle='-.', linewidth=2, label='CNN')
        if self.DNN.isChecked():
            F1.axes1.plot(self.t, self.y_pred_DNN, color='green', linestyle='--', linewidth=2, label='DNN')
        if self.LSTM.isChecked():
            F1.axes1.plot(self.t, self.y_pred_LSTM, color='orange', linestyle='--', linewidth=2, label='LSTM')
        max_values = np.array([np.max(self.y_pred_Phy),np.max(self.y_pred_CNN),np.max(self.y_pred_DNN),np.max(self.y_pred_LSTM)])
        min_values = np.array([np.min(self.y_pred_Phy),np.min(self.y_pred_CNN),np.min(self.y_pred_DNN),np.min(self.y_pred_LSTM)])
        max_value = np.max(max_values)
        min_value = np.min(min_values)
        # print(max_value)
        # print(min_value)
        F1.axes1.set_ylim(np.ceil(min_value-1), np.ceil(max_value+1))
        F1.axes1.set_xticks(np.arange(0, 0.02*self.npts, 10))
        F1.axes1.set_yticks(np.linspace(np.ceil(min_value-1), np.ceil(max_value+1), 6))
        F1.axes1.set_xlabel("t (s)")
        F1.axes1.set_ylabel("Acc. (m/s$^{2}$)")
        F1.axes1.legend()
        F1.axes1.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=4)
        width,height = self.graphicsView_O.width(),self.graphicsView_O.height()
        F1.resize(width*0.9,height*0.8)
        F1.fig.subplots_adjust(bottom=0.2, left=0.2)
        self.scene = QGraphicsScene()
        self.scene.addWidget(F1)
        self.graphicsView_O.setScene(self.scene)

    def Savetxt(self):
        try:
            data = (
                    np.transpose([self.t]),
                    np.transpose([self.y_pred_Phy]),
                    np.transpose([self.y_pred_CNN]),
                    np.transpose([self.y_pred_DNN]),
                    np.transpose([self.y_pred_LSTM]),
                )
            data2 = np.concatenate(data, axis=1)
            options = QFileDialog.Options()
            filePath, _ = QFileDialog.getSaveFileName(self, "Save File", "", "Text Files (*.txt)", options=options)

            if filePath:
                line1 = 'Number of stories: {}'.format(self.N1)
                line2 = 'Lateral length (m): {}'.format(self.Lt1)
                line3 = 'Longitudinal length (m): {}'.format(self.Ll1)
                line4 = 'Total height (m): {}'.format(self.H1)
                line5 = 'Story height (m): {}'.format(self.h1)
                line6 = 'Relative height: {}'.format(self.zh1)
                line7 = 'Number of basement stories: {}'.format(self.B1)
                line8 = 'Structure types: {}'.format(self.I1)
                line9 = 'Direction: {}'.format(self.FX1)
                line10 = 'Units: m/s2'
                column_names = ['t', 'PhyCNN', 'CNN', 'DNN', 'LSTM']
                header_lines = [line1, line2, line3, line4, line6, line7, line8, line9, line10]
                header = '\n'.join(header_lines) + '\n' + '\t'.join(column_names)
                np.savetxt(filePath, data2, fmt='%0.3f', header=header, comments='')
        except Exception as e:
            QMessageBox.warning(self, 'Warning', f"Error:{e}.", QMessageBox.Ok)

    def save_Output_image(self):
        try:
            fig = self.F1.fig
            file_dialog = QFileDialog(self)
            file_dialog.setNameFilter("PNG files (*.png);;JPEG files (*.jpg *.jpeg)")
            if file_dialog.exec_():
                file_path = file_dialog.selectedFiles()[0]
                fig.savefig(file_path)
        except Exception as e:
            QMessageBox.warning(self, 'Warning', f"Error: {e}", QMessageBox.Ok)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    demo = MyUiComputer()
    demo.show()
    sys.exit(app.exec_())
