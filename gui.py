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
from PyQt5.QtWidgets import QLabel, QComboBox, QSlider, QHBoxLayout, QVBoxLayout, QCheckBox, QGraphicsScene, QGraphicsView
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QTime, QDateTime, QDate, QSize
from PyQt5.QtGui import QFont

from astropy import units as u
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
#from py3dcore_h4c.gui.utils.widgets import SliderAndTextbox

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

params = ['CME Launch Time','CME Longitude', 'CME Latitude', 'CME Inclination', 'CME Diameter 1 AU', 'CME Aspect Ratio', 'CME Launch Radius', 'CME Launch Velocity', 'CME Expansion Rate', 'Background Drag', 'Background Velocity']

units = ['h','°', '°', '°', 'AU', '', 'rS','km/s', '', '', 'km/s']

variables = ['\u0394 t','lon', 'lat', 'inc', 'd(1AU)', '\u03B4', 'r\u2080', 'v\u2080', 'n', '\u03B3', 'v']

mins = [0, 0, -90, 0, 0.05, 1, 5, 400, 0.3, 0.2, 250]

maxs = [100, 360, 90, 360, 0.35, 6, 100, 1000, 2, 2,700]

inits = [0, 0, 0, 0, 0.2, 3, 20, 800, 1.14, 1, 500]

resolutions = [0.1, 0.1, 0.1, 0.1, 0.01, 1, 1, 1, 0.1, 0.1, 1]

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
        self.meshplots = []
        
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
        self.corebox = QCheckBox("3DCORE Flux Rope Model")
        self.corebox.stateChanged.connect(self.update_canvas)
        sidebar_layout.addWidget(self.corebox)

        
        sidebar_layout.addStretch(1)
        
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
        
        self.subplots = []
        self.images = []
        self.checkedsc = []
        self.iparams_list = inits
        
        staimage = load_image('STA', 'COR2', plotdate, runndiff)
        self.ax1 = self.fig.add_subplot(121, projection = staimage)
        staimage.plot(axes=self.ax1, cmap='Greys_r', norm=colors.Normalize(vmin=-30, vmax=30) if runndiff else None)
        self.images.append(staimage)
        ctype = self.ax1.wcs.wcs.ctype
        self.ax1.set_xlabel(axis_labels_from_ctype(ctype[0],spatial_units[0]), fontsize=12)
        self.ax1.set_ylabel(axis_labels_from_ctype(ctype[1],spatial_units[1]), fontsize=12)

        self.ax1.tick_params(axis='x', labelsize=12)
        self.ax1.tick_params(axis='y', labelsize=12)
        self.ax1.title.set_size(12+2)
        self.subplots.append(self.ax1)
        self.checkedsc.append('STA')
        
        sohoimage = load_image('SOHO', 'C2', plotdate, runndiff)
        self.ax2 = self.fig.add_subplot(122, projection = sohoimage)
        sohoimage.plot(axes=self.ax2, cmap='Greys_r', norm=colors.Normalize(vmin=-30, vmax=30) if runndiff else None)
        self.images.append(sohoimage)
        ctype = self.ax2.wcs.wcs.ctype
        self.font_size=12
        self.ax2.set_xlabel(axis_labels_from_ctype(ctype[0],spatial_units[0]), fontsize=self.font_size)
        self.ax2.set_ylabel(axis_labels_from_ctype(ctype[1],spatial_units[1]), fontsize=self.font_size)
        self.ax2.tick_params(axis='x', labelsize=12)
        self.ax2.tick_params(axis='y', labelsize=12)
        self.ax2.title.set_size(12+2)
        self.subplots.append(self.ax2)
        self.checkedsc.append('SOHO')
        
        vlayout_dt = QHBoxLayout()
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 1000)
        slider.setValue(0)
        slider.setTickInterval(10)
        vlayout_dt.addWidget(slider)
        self.dt_val = QLabel("\u0394 t: {} h".format(0))
        vlayout_dt.addWidget(self.dt_val)
        
        middle_layout.addLayout(vlayout_dt)
        
        main_layout.addLayout(middle_layout,5)
        
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
        
        # parameters
        
        self.paramlabels = []
        self.paramsliders = []
        
        for i in range(11):
            if i == 0:
                self.paramsliders.append(slider)
                self.paramlabels.append(dtlabel)
            else:
                rightbar_layout.addWidget(emptylabel)
                hlayout = QHBoxLayout()
                label = QLabel(params[i])
                hlayout.addWidget(label)
                hlayout.addStretch(1)
                updatelabel = QLabel('{}: {} {}'.format(variables[i], inits[i], units[i]))
                hlayout.addWidget(updatelabel)
                self.paramlabels.append(updatelabel)
                rightbar_layout.addLayout(hlayout)
            
                slider = QSlider(Qt.Horizontal)
                slider.setRange(int(mins[i]/resolutions[i]),int(maxs[i]/resolutions[i]))
                slider.setValue(int(inits[i]/resolutions[i]))
                self.paramsliders.append(slider)
                rightbar_layout.addWidget(slider)
        
        for slider in self.paramsliders:
            slider.valueChanged.connect(self.plot_mesh)


        # save / load
        
        #sidebar_layout.addStretch(1)
        

        # add the sidebar to the main layout
        main_layout.addLayout(rightbar_layout,1)
        

        # show the main window
        self.show()
        
    def update_canvas(self):
        self.runndiff = False
        # get the selected font size from the combobox
        self.font_size = int(self.font_size_combobox.currentText())
        selected_date = self.calendar.selectedDate().toPyDate()
        selected_time = QTime.fromString(self.time_combobox.currentText(), "h:mm AP").toPyTime()
        self.plotdate = dt.datetime.combine(selected_date, selected_time)
        
        
        #update font size everywhere
        
        # set the font size of all relevant widgets
        font = QFont()
        font.setPointSize(self.font_size)
    
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
        self.images = []
        
        self.checkedsc = []
        checkeddet = []
        currentdets = [self.instr_combobox_sta.currentText(),self.instr_combobox_stb.currentText(),self.instr_combobox_soho.currentText()]
        
        for i, checkbox in enumerate(self.checkboxes):
            if checkbox.isChecked():
                self.checkedsc.append(spacecrafts[i])
                checkeddet.append(currentdets[i])
                
        for i, sc in enumerate(self.checkedsc):
            image = load_image(sc, checkeddet[i], self.plotdate, self.runndiff)
            self.images.append(image)
    
        
        ########self._bg = fig.canvas.copy_from_bbox(self.fig.bbox)
        # redraw the canvas
        self.fig.canvas.draw()
    
    def plot_mesh(self):
        # clear the existing subfigures
        self.fig.clf()
        self.subplots =[]
        self.runndiff = False
        sender = self.sender() 
        self.iparams_list = []
        
        for i, label in enumerate(self.paramlabels):
            if i == 0:
                self.dt_val.setText("\u0394 t: {} h".format(self.paramsliders[i].value()/10))
                selected_date = self.calendar.selectedDate().toPyDate()
                selected_time = QTime.fromString(self.time_combobox.currentText(), "h:mm AP").toPyTime()
                plotdate = dt.datetime.combine(selected_date, selected_time)
                launchtime = plotdate - datetime.timedelta(hours = self.paramsliders[i].value()/10)
                self.dtlabelupdating.setText(launchtime.strftime('%Y-%m-%d %H:%M:00'))
            else:
                
                if resolutions[i] == 1:
                    val = int(self.paramsliders[i].value()*resolutions[i])
                elif resolutions[i] == 0.1:
                    val = float("{:.1f}".format(self.paramsliders[i].value()*resolutions[i]))
                elif resolutions[i] == 0.01:
                    val = float("{:.2f}".format(self.paramsliders[i].value()*resolutions[i]))
                self.iparams_list.append(val)
                label.setText('{}: {} {}'.format(variables[i], val, units[i]))

               
        for i, image in enumerate(self.images):
            self.subplots.append(self.fig.add_subplot( 1, len(self.checkedsc),i+1, projection=image))
            image.plot(axes= self.subplots[-1], cmap='Greys_r', norm=colors.Normalize(vmin=-30, vmax=30) if self.runndiff else None)
            ctype = self.subplots[-1].wcs.wcs.ctype
            self.subplots[-1].set_xlabel(axis_labels_from_ctype(ctype[0],spatial_units[0]), fontsize=self.font_size)
            self.subplots[-1].set_ylabel(axis_labels_from_ctype(ctype[1],spatial_units[1]), fontsize=self.font_size)
            self.subplots[-1].tick_params(axis='x', labelsize=self.font_size)
            self.subplots[-1].tick_params(axis='y', labelsize=self.font_size)
            self.subplots[-1].title.set_size(self.font_size+2)
            
        
        iparams = self.get_iparams()
               
        timedelta = self.paramsliders[0].value()/10
        if self.corebox.isChecked():
            t_snap = datetime.datetime(2020,4,15,6)
            dt_0 = t_snap - datetime.timedelta(hours = timedelta)
            mesh = go.py3dcore_mesh_sunpy(dt_0, t_snap,iparams)
            for i, (image, subplot) in enumerate(zip(self.images, self.subplots)):
               # if len(self.meshplots) <= i:
                style = '-'
                params = dict(lw=0.5)
                p = subplot.plot_coord(mesh.T, color='blue', scalex=False, scaley=False, **params)[0]
                p2 = subplot.plot_coord(mesh, color='blue', scalex=False, scaley=False, **params)[0]
                  #  self.meshplots.append(p)
          #      else:
                    # update plot
            #        p = self.meshplots[i]

            #        frame0 = mesh.frame.transform_to(image.coordinate_frame)
             #       xdata = frame0.spherical.lon.to_value(u.deg)
            #        ydata = frame0.spherical.lat.to_value(u.deg)
           #         p.set_xdata(xdata)
           #         p.set_ydata(ydata)
           #         subplot.draw_artist(p)

            self.fig.canvas.draw()

        else:
            return
        
    def load_model(self):
        runndiff = False

    def save(self):
        runndiff = False
        
    def get_iparams(self):
        model_kwargs = {
            "ensemble_size": int(1), #2**17
            "iparams": {
                "cme_longitude": {
                    "distribution": "fixed",
                    "default_value": self.iparams_list[0],
                    "maximum": 360,
                    "minimum": 0
                },
                "cme_latitude": {
                    "distribution": "fixed",
                    "default_value": self.iparams_list[1],
                    "maximum": 90,
                    "minimum": -90
                },
                "cme_inclination": {
                    "distribution": "fixed",
                    "default_value": self.iparams_list[2],
                    "maximum": 360,
                    "minimum": 0
                }, 
                "cme_diameter_1au": {
                    "distribution": "fixed",
                    "default_value": self.iparams_list[3],
                    "maximum": 0.35,
                    "minimum": 0.05
                }, 
                "cme_aspect_ratio": {
                    "distribution": "fixed",
                    "default_value": self.iparams_list[4],
                    "maximum": 6,
                    "minimum": 1
                },
                "cme_launch_radius": {
                    "distribution": "fixed",
                    "default_value": self.iparams_list[5],
                    "maximum": 100,
                    "minimum": 5
                },
                "cme_launch_velocity": {
                    "distribution": "fixed",
                    "default_value": self.iparams_list[6],
                    "maximum": 1000,
                    "minimum": 400
                },
                "t_factor": {
                    "distribution": "fixed",
                    "default_value": 100,
                    "maximum": 250,
                    "minimum": 50
                },
                "magnetic_field_strength_1au": {
                    "distribution": "fixed",
                    "default_value": 25,
                    "maximum": 50,
                    "minimum": 5
                },
                
                "cme_expansion_rate": {
                    "distribution": "fixed",
                    "default_value": self.iparams_list[7],
                    "maximum": 2,
                    "minimum": 0.3
                }, 
                "background_drag": {
                    "distribution": "fixed",
                    "default_value": self.iparams_list[8],
                    "maximum": 2,
                    "minimum": 0.2
                }, 
                "background_velocity": {
                    "distribution": "fixed",
                    "default_value": self.iparams_list[9],
                    "maximum": 700,
                    "minimum": 250
                } 
            }
        }
        return model_kwargs


def main():
    qapp = QtWidgets.QApplication(sys.argv)
    app = py3dcoreGUI()
    app.show()
    sys.exit(qapp.exec_())


if __name__ == '__main__':
    main()
    
    
    
