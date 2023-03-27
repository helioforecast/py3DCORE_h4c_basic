import argparse
import datetime as dt
import json
import os
import sys
from typing import List
import py3dcore_h4c.gui.guiold as go
import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QLabel, QComboBox, QSlider, QHBoxLayout, QVBoxLayout, QCheckBox
from PyQt5.QtCore import Qt, QTime, QDateTime, QDate, QSize
from PyQt5.QtGui import QFont

from astropy import units
from astropy.coordinates import concatenate
from matplotlib import colors
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from sunpy import log
from sunpy.coordinates import get_horizons_coord
from sunpy.map import Map
from sunpy.visualization import axis_labels_from_ctype

from py3dcore_h4c.gui.geometry import gcs_mesh_sunpy, apex_radius
from py3dcore_h4c.gui.utils.helioviewer import get_helioviewer_client
from py3dcore_h4c.gui.utils.widgets import SliderAndTextbox

matplotlib.use('Qt5Agg')

hv = get_helioviewer_client()

straight_vertices, front_vertices, circle_vertices = 10, 10, 20
filename = 'gcs_params.json'
draw_modes = ['off', 'point cloud', 'grid']
font_sizes = [10,12,14,16,18]
spacecrafts = ['STA','STB','SOHO']
instruments = ['SECCHI','SECCHI','LASCO']
dets = [['COR2', 'COR1'],['COR2', 'COR1'],['C2', 'C3']]
spatial_units = [None, None]

# disable sunpy warnings
log.setLevel('ERROR')

#################################################################################

def running_difference(a, b):
    return Map(b.data * 1.0 - a.data * 1.0, b.meta)

def load_image(spacecraft: str, detector: str, date: dt.datetime, runndiff: bool):
    if spacecraft == 'STA':
        observatory = 'STEREO_A'
        instrument = 'SECCHI'
        if detector not in ['COR1', 'COR2']:
            raise ValueError(f'unknown detector {detector} for spacecraft {spacecraft}.')
    elif spacecraft == 'STB':
        observatory = 'STEREO_B'
        instrument = 'SECCHI'
        if detector not in ['COR1', 'COR2']:
            raise ValueError(f'unknown detector {detector} for spacecraft {spacecraft}.')
    elif spacecraft == 'SOHO':
        observatory = 'SOHO'
        instrument = 'LASCO'
        if detector not in ['C2', 'C3']:
            raise ValueError(f'unknown detector {detector} for spacecraft {spacecraft}.')
    else:
        raise ValueError(f'unknown spacecraft: {spacecraft}')

    f = download_helioviewer(date, observatory, instrument, detector)

    if runndiff:
        f2 = download_helioviewer(date - dt.timedelta(hours=1), observatory, instrument, detector)
        return running_difference(f2, f)
    else:
        return f

def download_helioviewer(date, observatory, instrument, detector):
    file = hv.download_jp2(date, observatory=observatory, instrument=instrument, detector=detector)
    f = Map(file)

    if observatory == 'SOHO':
       # add observer location information:
       soho = get_horizons_coord('SOHO', f.date)
       f.meta['HGLN_OBS'] = soho.lon.to('deg').value
       f.meta['HGLT_OBS'] = soho.lat.to('deg').value
       f.meta['DSUN_OBS'] = soho.radius.to('m').value

    return f


###############################################################################################



class py3dcoreGUI(QtWidgets.QWidget): #.QMainWindow
    def __init__(self):
        super().__init__()
        
        # (left, top, width, height)
        self.setGeometry(100,100,2400,1000)
        self.setWindowTitle('3DCORE - GUI')
        
        #set up main window layout
        
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)
        
        ############################### creating sidebar ###############################
        

        sidebar_layout = QVBoxLayout()
        
        # CALENDAR
        
        date_label = QLabel()
        date_label.setText('Select Date:')
        sidebar_layout.addWidget(date_label)
        calendar = QtWidgets.QCalendarWidget()
        self.calendar = calendar
        sidebar_layout.addWidget(self.calendar)
        date = QDate(2020, 4, 15)
        self.calendar.setSelectedDate(date)
        self.calendar.clicked.connect(self.update_canvas)
        
        # TIME
        emptylabel = QLabel()
        emptylabel.setText('')
        sidebar_layout.addWidget(emptylabel)

        time_label = QLabel("Select Time:")
        self.time_combobox = QComboBox()
        for hour in range(24):
            time = QTime(hour, 0)
            self.time_combobox.addItem(time.toString("h:mm AP"))
        self.time_combobox.setCurrentIndex(6)
        self.time_combobox.currentIndexChanged.connect(self.update_canvas)
        sidebar_layout.addWidget(time_label)
        sidebar_layout.addWidget(self.time_combobox)
       
        # SPACECRAFTs
        
        
        sidebar_layout.addWidget(emptylabel)

        self.checkboxes = []
        
        spaecrafts_label = QLabel()
        spaecrafts_label.setText('Spacecraft and Instrument:')
        sidebar_layout.addWidget(spaecrafts_label)
        
        vlayout_sta = QHBoxLayout()
        checkbox_sta = QCheckBox("STEREO-A")
        checkbox_sta.setChecked(True)
        checkbox_sta.stateChanged.connect(self.update_canvas)
        self.checkboxes.append(checkbox_sta)
        vlayout_sta.addWidget(checkbox_sta)
        self.instr_combobox_sta = QComboBox()
        for i in dets[0]:
            self.instr_combobox_sta.addItem(i)
        self.instr_combobox_sta.currentIndexChanged.connect(self.update_canvas)
        vlayout_sta.addWidget(self.instr_combobox_sta)
        sidebar_layout.addLayout(vlayout_sta)
        
        vlayout_stb = QHBoxLayout()
        checkbox_stb = QCheckBox("STEREO-B")
        checkbox_stb.stateChanged.connect(self.update_canvas)
        self.checkboxes.append(checkbox_stb)
        vlayout_stb.addWidget(checkbox_stb)
        self.instr_combobox_stb = QComboBox()
        for i in  dets[1]:
            self.instr_combobox_stb.addItem(i)
        self.instr_combobox_stb.currentIndexChanged.connect(self.update_canvas)
        vlayout_stb.addWidget(self.instr_combobox_stb)
        sidebar_layout.addLayout(vlayout_stb)
        
        vlayout_soho = QHBoxLayout()
        checkbox_soho = QCheckBox("SOHO")
        checkbox_soho.setChecked(True)
        checkbox_soho.stateChanged.connect(self.update_canvas)
        self.checkboxes.append(checkbox_soho)
        vlayout_soho.addWidget(checkbox_soho)
        self.instr_combobox_soho = QComboBox()
        for i in  dets[2]:
            self.instr_combobox_soho.addItem(i)
        self.instr_combobox_stb.currentIndexChanged.connect(self.update_canvas)
        vlayout_soho.addWidget(self.instr_combobox_soho)
        sidebar_layout.addLayout(vlayout_soho)
        
        
        # 3DCORE MODEL
        
        sidebar_layout.addWidget(emptylabel)

        core_label = QLabel()
        core_label.setText('Draw Grid:')
        sidebar_layout.addWidget(core_label)
        corebox = QCheckBox("3DCORE Flux Rope Model")
        corebox.stateChanged.connect(self.plot_mesh)
        sidebar_layout.addWidget(corebox)

        
        sidebar_layout.addStretch(50)
        
        # FONT SIZE
        font_size_label = QLabel("Font Size:")
        self.font_size_combobox = QComboBox()
        for i in font_sizes:
            self.font_size_combobox.addItem(str(i))
        self.font_size_combobox.setCurrentIndex(1)
        self.font_size_combobox.currentIndexChanged.connect(self.update_canvas)
        sidebar_layout.addWidget(font_size_label)
        sidebar_layout.addWidget(self.font_size_combobox)
        
        
        # add the sidebar to the main layout
        main_layout.addLayout(sidebar_layout, 0)
        
        ############################### creating main canvas and time slider ###############################
        
        plotdate = dt.datetime(2020,4,15,6)
        runndiff = False
        
        middle_layout = QVBoxLayout()
        
        self.fig = Figure(figsize=(10, 5), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        middle_layout.addWidget(self.canvas, 4)
        
        staimage = load_image('STA', 'COR2', plotdate, runndiff)
        self.ax1 = self.fig.add_subplot(121, projection = staimage)
        staimage.plot(axes=self.ax1, cmap='Greys_r', norm=colors.Normalize(vmin=-30, vmax=30) if runndiff else None)
        ctype = self.ax1.wcs.wcs.ctype
        self.ax1.set_xlabel(axis_labels_from_ctype(ctype[0],spatial_units[0]), fontsize=12)
        self.ax1.set_ylabel(axis_labels_from_ctype(ctype[1],spatial_units[1]), fontsize=12)

        self.ax1.tick_params(axis='x', labelsize=12)
        self.ax1.tick_params(axis='y', labelsize=12)
        self.ax1.title.set_size(12+2)
        
        sohoimage = load_image('SOHO', 'C2', plotdate, runndiff)
        self.ax2 = self.fig.add_subplot(122, projection = sohoimage)
        sohoimage.plot(axes=self.ax2, cmap='Greys_r', norm=colors.Normalize(vmin=-30, vmax=30) if runndiff else None)
        ctype = self.ax2.wcs.wcs.ctype
        self.ax2.set_xlabel(axis_labels_from_ctype(ctype[0],spatial_units[0]), fontsize=12)
        self.ax2.set_ylabel(axis_labels_from_ctype(ctype[1],spatial_units[1]), fontsize=12)
        self.ax2.tick_params(axis='x', labelsize=12)
        self.ax2.tick_params(axis='y', labelsize=12)
        self.ax2.title.set_size(12+2)
        
        vlayout_dt = QHBoxLayout()
        self.dtslider = QSlider(Qt.Horizontal)
        self.dtslider.setRange(0, 1000)
        self.dtslider.setTickInterval(10)
        
        vlayout_dt.addWidget(self.dtslider)
        self.dtslider.valueChanged.connect(self.dt_changed)
        self.dt_val = QLabel("\u0394 t: {} h".format(0))
        vlayout_dt.addWidget(self.dt_val)
        
        middle_layout.addLayout(vlayout_dt)
        
        main_layout.addLayout(middle_layout, 1)
        
        ############################### creating right sidebar ###############################
        
        
        rightbar_layout = QVBoxLayout()
        rightbar_layout.setAlignment(Qt.AlignTop)
        
        # launchtime
        
        dtlayout = QHBoxLayout()
        
        dtlabel = QLabel("Launch Time:                     ")
        dtlayout.addWidget(dtlabel)
        dtlayout.addStretch(1)
        
        self.dtlabelupdating = QLabel(plotdate.strftime('%Y-%m-%d %H:%M:00'))
        dtlayout.addWidget(self.dtlabelupdating)
        rightbar_layout.addLayout(dtlayout)
        
        # cme longitude 
        
        sidebar_layout.addWidget(emptylabel)

        lonlayout = QHBoxLayout()
        
        lonlabel = QLabel("CME Longitude:                     ")
        lonlayout.addWidget(lonlabel)
        lonlayout.addStretch(1)
        
        self.lonlabelupdating = QLabel("lon: {} °".format(0) )
        lonlayout.addWidget(self.lonlabelupdating)
        rightbar_layout.addLayout(lonlayout)
        
        self.lonslider = QSlider(Qt.Horizontal)
        self.lonslider.setRange(-90, 90)
        rightbar_layout.addWidget(self.lonslider)
        self.lonslider.valueChanged.connect(self.lon_changed)
        
        

        # save / load
        
        sidebar_layout.addStretch(1)
        

        # add the sidebar to the main layout
        main_layout.addLayout(rightbar_layout, 0)
        

        # show the main window
        self.show()
        
    def update_canvas(self):
            
        # clear the existing subfigures
        self.fig.clf()
        runndiff = False
        
        # get the selected font size from the combobox
        font_size = int(self.font_size_combobox.currentText())
        selected_date = self.calendar.selectedDate().toPyDate()
        selected_time = QTime.fromString(self.time_combobox.currentText(), "h:mm AP").toPyTime()
        plotdate = dt.datetime.combine(selected_date, selected_time)
        
        
        #update font size everywhere
        
        # set the font size of all relevant widgets
        font = QFont()
        font.setPointSize(font_size)
    
        for label in self.findChildren(QLabel):
            label.setFont(font)
        for label in self.findChildren(QSlider):
            label.setFont(font)
        for label in self.findChildren(QComboBox):
            label.setFont(font)
        for label in self.findChildren(QCheckBox):
            label.setFont(font)
        self.calendar.setFont(font)

        # create a new subfigure for each checkbox that is checked
        subplots = []
        checkedsc = []
        checkeddet = []
        currentdets = [self.instr_combobox_sta.currentText(),self.instr_combobox_stb.currentText(),self.instr_combobox_soho.currentText()]
        
        for i, checkbox in enumerate(self.checkboxes):
            if checkbox.isChecked():
                checkedsc.append(spacecrafts[i])
                checkeddet.append(currentdets[i])
        
        for i, sc in enumerate(checkedsc):
            image = load_image(sc, checkeddet[i], plotdate, runndiff)
            subplots.append(self.fig.add_subplot( 1, len(checkedsc),i+1, projection=image))
            image.plot(axes=subplots[-1], cmap='Greys_r', norm=colors.Normalize(vmin=-30, vmax=30) if runndiff else None)
            ctype = subplots[-1].wcs.wcs.ctype
            subplots[-1].set_xlabel(axis_labels_from_ctype(ctype[0],spatial_units[0]), fontsize=font_size)
            subplots[-1].set_ylabel(axis_labels_from_ctype(ctype[1],spatial_units[1]), fontsize=font_size)
            subplots[-1].tick_params(axis='x', labelsize=font_size)
            subplots[-1].tick_params(axis='y', labelsize=font_size)
            subplots[-1].title.set_size(font_size+2)
            
        # redraw the canvas
        self.canvas.draw()
    
    def plot_mesh(self):
        runndiff = False
        
    def dt_changed(self):
        self.dt_val.setText("\u0394 t: {} h".format(self.dtslider.value()/10))
        
        selected_date = self.calendar.selectedDate().toPyDate()
        selected_time = QTime.fromString(self.time_combobox.currentText(), "h:mm AP").toPyTime()
        plotdate = dt.datetime.combine(selected_date, selected_time)
        launchtime = plotdate - datetime.timedelta(hours = self.dtslider.value()/10)
        self.dtlabelupdating.setText(launchtime.strftime('%Y-%m-%d %H:%M:00'))
        self.plot_mesh()
        
    def lon_changed(self):
        self.lonlabelupdating.setText("lon: {} °".format(self.lonslider.value()))
        self.plot_mesh()
        
        
    def load_model(self):
        runndiff = False

    def save(self):
        runndiff = False


def main():
    qapp = QtWidgets.QApplication(sys.argv)
    app = py3dcoreGUI()
    app.show()
    sys.exit(qapp.exec_())


if __name__ == '__main__':
    main()