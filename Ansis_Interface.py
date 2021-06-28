# Ramiro Isa-Jara, ramiro.isaj@gmail.com
# ANSIS_GUI version 1.2 ->
# Graphical Interface for Analysis of cells Images using Segmentation methods for automatic control

import PySimpleGUI as sg
import Ansis_def as Ans
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

# -------------------------------
# Adjust size screen
# -------------------------------
Screen_size = 10
# -------------------------------
sg.theme('LightGrey1')

img = np.ones((350, 250, 1), np.uint8)*255

f1 = np.array([60])
fig, ax = plt.subplots(figsize=(4, 3), dpi=80)
ax.plot(f1, 'o-')
ax.grid()

portsWIN = ['COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9']
portsLIN = ['/dev/pts/2', '/dev/ttyS0', '/dev/ttyS1', '/dev/ttyS2', '/dev/ttyS3']

layout1 = [[sg.Radio('Windows', "RADIO1", enable_events=True, default=True, key='_SYS_')],
           [sg.Radio('Linux', "RADIO1", enable_events=True, key='_LIN_')], [sg.Text('')]]

layout2 = [[sg.Text('Name: ', size=(10, 1)),
            sg.Combo(values=portsWIN, size=(10, 1), enable_events=True, key='_PORT_')],
           [sg.Text('Baudrate:', size=(10, 1)), sg.InputText('9600', key='_RTE_', size=(10, 1))],
           [sg.Text('Status:', size=(9, 1)), sg.Text('NOT CONNECT', size=(13, 1), key='_CON_', text_color='red')]]

layout3 = [[sg.Text('Source : ', size=(10, 1)), sg.InputText(size=(40, 1), key='_ORI_'), sg.FolderBrowse()],
           [sg.Text('Destiny: ', size=(10, 1)), sg.InputText(size=(40, 1), key='_DES_'), sg.FolderBrowse()]]

layout4 = [[sg.Checkbox('*.jpg', default=True, key="_IN1_")], [sg.Checkbox('*.png', default=False, key="_IN2_")],
           [sg.Checkbox('*.tiff', default=False, key="_IN3_")]]

layout5 = [[sg.Text('Buffer size:', size=(12, 1)), sg.InputText('10', key='_BUF_', size=(7, 1)),
            sg.Text('Lambda:', size=(10, 1)), sg.InputText('0.20', key='_LAM_', size=(5, 1))],
           [sg.Text('Time wait (min):', size=(12, 1)), sg.InputText('5', key='_TIM_', size=(7, 1)),
            sg.Text('Threshold:', size=(10, 1)), sg.InputText('25', key='_THE_', size=(5, 1))],
           [sg.Text('Fluid_Max_Min:', size=(12, 1)), sg.InputText('250/200', key='_FVA_', size=(7, 1), enable_events=True),
            sg.Text('%Areas_ON:', size=(10, 1)), sg.InputText('85/50', key='_PAR_', size=(5, 1), enable_events=True)]]

layout5b = [[sg.Radio('Gema', "RADIO2", default=True, key='_ALG_'),
             sg.Radio('On-line', "RADIO3", default=True, key='_MET_')],
            [sg.Radio('Origin', "RADIO2"), sg.Radio('Off-line', "RADIO3")], [sg.Text('')]]

layout6 = [[sg.T("", size=(15, 1)), sg.Text('Image viewer', size=(15, 1), text_color='DarkBlue'),
            sg.T("", size=(17, 1)), sg.Text('PLOT: Computed percentage area', size=(37, 1), text_color='DarkBlue')]]

layout7 = [[sg.Text('Current time: ', size=(13, 1)), sg.Text('', size=(12, 1), key='_TAC_'), sg.T("", size=(4, 1)),
            sg.Text('Start time: ', size=(11, 1)), sg.Text('-- : -- : --', size=(15, 1), key='_TIN_', text_color='blue'),
            sg.Text('Finish time: ', size=(12, 1)), sg.Text('-- : -- : --', size=(12, 1), key='_TFI_', text_color='red')],
           [sg.Text('Experiment:', size=(13, 1)), sg.InputText('', key='_NEX_', size=(15, 1)), sg.Text('', size=(1, 1)),
            sg.Text('Image:', size=(5, 1)), sg.InputText('', key='_NIM_', size=(18, 1)), sg.Text('', size=(1, 1)),
            sg.Text('Mean area: ', size=(13, 1)), sg.InputText('0', key='_MPO_', size=(6, 1)), sg.Text('%', size=(3, 1))],
           [sg.Text('Current image:', size=(13, 1)), sg.InputText('', key='_CIM_', size=(7, 1)), sg.Text('', size=(9, 1)),
            sg.Text('State PUMP: ', size=(12, 1)), sg.Text('OFF', size=(8, 1), text_color='red', key='_PUM_'),
            sg.T('', size=(4, 1)), sg.Text('Current area:', size=(13, 1)), sg.InputText('0', key='_CPO_', size=(6, 1)),
            sg.Text('%', size=(3, 1))],
           [sg.Text('Buffer image:', size=(13, 1)), sg.InputText('', key='_BIM_', size=(7, 1)), sg.Text('', size=(9, 1)),
            sg.Text('Fluid / value:', size=(11, 1)), sg.Text('0', size=(4, 1), text_color='blue', key='_FRE_'),
            sg.Text('ul/min', size=(5, 1)), sg.T('', size=(2, 1)), sg.Text('Previous area:', size=(13, 1)),
            sg.InputText('0', key='_PPO_', size=(6, 1)), sg.Text('%', size=(3, 1))]]

v_image = [sg.Image(filename='', key="_IMA_")]

layout = [[sg.Frame('Operative Syst: ', layout1, title_color='Blue'),
           sg.Frame('Type image: ', layout4, title_color='Blue'),
           sg.Frame('Settings: ', layout5, title_color='Blue'),
           sg.Frame('Methods: ', layout5b, title_color='Blue')],
          [sg.Frame('Directories: ', layout3, title_color='Blue'),
           sg.Frame('Serial port: ', layout2, title_color='Blue')],
          [sg.T(" ", size=(28, 1)), sg.Button('Start', size=(8, 1)),
           sg.Button('Pause', size=(8, 1)), sg.Button('Save & Finish', size=(10, 1))],
          [sg.Frame('', layout6)], [sg.Frame('', [v_image]), sg.Canvas(key="_CANVAS_")],
          [sg.Frame('Results: ', layout7, title_color='Blue')]]

# Create the Window
window = sg.Window('ANSIS Interface', layout, font="Helvetica "+str(Screen_size), icon='icon.ico', finalize=True)
window['_IMA_'].update(data=Ans.bytes_(img))
graph = Ans.draw_figure(window["_CANVAS_"].TKCanvas, fig)
# ------------------------------
control, finish_ = False, False
path_org, path_des, type_i, ide, method, port_name, buffer, exp, c_var, filenames = [], [], [], [], [], [], [], [], [], []
val_fluid, c_port, i, k, current_p, bauds, p_area, id_sys, pause_, per_area = 200, 0, -1, 0, 0, 0, [60], 0, False, 85
adap, adap1, cont, max_v, min_v, data_b, x, y, m1, y_base = 5, 5, 0, 0, 0, [], [], [], 0, None
area_sup, area_inf, flu_sup, flu_inf = 0, 0, 0, 0
# ------------------------------
data = pd.DataFrame(columns=['Image', 'Percentage', 'Back_area', 'Cell_area', 'Time (sec.)'])
events = pd.DataFrame(columns=['Time', 'Mean_area', 'Fluid_value', 'Pump_state'])
# ------------------------------
filters = Ans.build_filters()

# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read(timeout=10)
    window.Refresh()
    now = datetime.now()
    now_time = now.strftime("%H : %M : %S")
    window['_TAC_'].update(now_time)

    if event == sg.WIN_CLOSED:
        break

    if event == 'Save & Finish' or finish_:
        if control:
            now_time = now.strftime("%H : %M : %S")
            window['_IMA_'].update(data=Ans.bytes_(img))
            window['_TFI_'].update(now_time)
            # save results and events
            Ans.save_csv_file(data, path_des, exp, ide, 'Results')
            Ans.save_csv_file(events, path_des, exp, ide, 'Events')
            for i in range(1, 100, 25):
                sg.OneLineProgressMeter('Saving RESULTS in CSV files', i + 25, 100, 'single')
                time.sleep(1)
            Ans.graph_data(path_des, exp, ide)
            i, k, p_area = -1, 0, [60]
            adap, adap1, cont, max_v, min_v, data_b, x, y, m1, y_base = 5, 5, 0, 0, 0, [], [], [], 0, None
            window['_CPO_'].update('')
            window['_PPO_'].update('')
            window['_CIM_'].update('')
            window['_BIM_'].update('')
            window['_FRE_'].update('0')
            window['_PUM_'].update('OFF')
            ax.clear()
            ax.plot(f1, 'o-')
            ax.grid()
            graph.draw()
        data.drop(data.index, inplace=True)
        events.drop(events.index, inplace=True)
        control, finish_, pause_ = False, False, False

    if event == 'Pause':
        control = False

    if event == '_LIN_':
        window.Element('_PORT_').update(values=portsLIN)
    if event == '_SYS_':
        window.Element('_PORT_').update(values=portsWIN)

    if event == '_PORT_':
        port_name = values['_PORT_']
        bauds = int(values['_RTE_'])
        sg.Popup('Serial Port: ', values['_PORT_'])
        c_port = Ans.serial_test(port_name, bauds)
        text = 'CONNECT' if c_port == 1 else 'ERROR'
        window.Element('_CON_').update(text)

    if event == '_FVA_':
        if len(values['_FVA_']) == 6:
            val_fluid = values['_FVA_'].split('/')
            flu_sup = int(val_fluid[0]) if int(val_fluid[0]) > 100 else 0
            flu_inf = int(val_fluid[1]) if int(val_fluid[1]) > 100 else 0

    if event == '_PAR_':
        if len(values['_PAR_']) == 4:
            per_area = values['_PAR_'].split('/')
            area_sup = float(per_area[0]) if float(per_area[0]) > 80 else 0
            area_inf = float(per_area[1]) if float(per_area[1]) < 70 else 0

    if event == 'Start':
        if values['_SYS_'] is True:
            id_sys = 0
            path_org = Ans.update_dir(values['_ORI_']) + "\\"
            path_org = r'{}'.format(path_org)
            path_des = Ans.update_dir(values['_DES_']) + "\\"
            path_des = r'{}'.format(path_des)

        else:
            id_sys = 1
            path_org, path_des = values['_ORI_']+'/', values['_DES_']+'/'

        # ------------------------------
        if values['_IN2_']:
            type_i = "*.png"
        elif values['_IN3_']:
            type_i = "*.tiff"
        else:
            type_i = "*.jpg"
        # ------------------------------------------------------------------
        method = 0 if values['_MET_'] else 1
        ide = 'gema' if values['_ALG_'] else 'original'
        # ------------------------------------------------------------------
        per_area = values['_PAR_'].split('/')
        area_sup, area_inf = float(per_area[0]), float(per_area[1])
        val_fluid = values['_FVA_'].split('/')
        flu_sup, flu_inf = int(val_fluid[0]), int(val_fluid[1])
        # ------------------------------------------------------------------
        if method == 0 and path_org != '' and path_des != '' and c_port == 1:
            now_time = now.strftime("%H : %M : %S")
            window['_TIN_'].update(now_time)
            control = True
        elif method == 1 and path_org != '' and path_des != '':
            now_time = now.strftime("%H : %M : %S")
            window['_TIN_'].update(now_time)
            control = True
        else:
            sg.Popup('Error', ['Information not valid...'])

    if control:
        buffer_size = int(values['_BUF_'])
        time_sleep = int(values['_TIM_']) * 60
        lambda_g, thresh = float(values['_LAM_']), int(values['_THE_'])
        if method == 0:
            for i1 in range(1, time_sleep):
                eventC = sg.OneLineProgressMeter('Waiting new image', i1 + 1, time_sleep, 'single')
                if eventC is False and i1 < time_sleep-1:
                    sg.Popup('Process canceled', ['Analysis was stopped...'])
                    finish_ = True
                    break
                time.sleep(1)
            if finish_:
                continue
            image, total, exp, name = Ans.load_image(path_org, type_i, id_sys)
            if total > 0:
                i += 1
            else:
                sg.Popup('Warning', ['No images in directory...'])
                continue                   
        else:
            i += 1
            filenames, image, exp, name = Ans.load_image_i(path_org, i, type_i, filenames, exp, id_sys)
            if len(image) == 0 and i > 0:
                finish_ = True
                continue
            elif len(image) == 0 and i == 0:
                finish_ = True
                continue

        window['_NEX_'].update(exp)
        window['_NIM_'].update(name)
        window['_CIM_'].update(i)
        window['_BIM_'].update(k)

        if ide == 'gema':
            ima_out, data, percent, k, buffer, mean_area, c_var, x, y, max_v, min_v, y_base, m1, cont, adap \
                = Ans.gema(i, data, image, name, filters, buffer_size, buffer, k, lambda_g, c_var, path_des, x, y,
                           cont, max_v, min_v, adap1, adap, y_base, m1)
        else:
            ima_out, data, percent, k, buffer, mean_area = Ans.original(i, data, image, name, buffer_size, buffer, k,
                                                                        thresh, path_des)
        if i == 0:
            current_p, last_p = percent, percent
        else:
            last_p = current_p
            current_p = percent

        window['_PPO_'].update(last_p)
        window['_CPO_'].update(current_p)
        window['_IMA_'].update(data=Ans.bytes_(ima_out))
        # ---------------------------------------------------------------------------------
        if mean_area > 0:
            window['_MPO_'].update(mean_area)
            print('Mean area from buffer:   ' + str(mean_area))
            # ---------------------------------------------------
            p_area.append(mean_area)
            values_area = np.array(p_area)
            ax.clear()
            ax.plot(values_area, 'o-')
            ax.grid()
            graph.draw()
            # -----------------------------------------------------
            # control pump
            if method == 0:
                if pause_ is False:
                    time_control, act = 1, 'OFF'
                    act, v_fluid = Ans.pump_control(port_name, bauds, mean_area, time_control, act,
                                                    area_sup, area_inf, flu_sup, flu_inf)
                    if act == 'ON':
                        window['_FRE_'].update(v_fluid)
                        now_time1 = now.strftime("%H : %M : %S")
                        print('-------------------------------------------------------------------------')
                        print(now_time1 + ' ::: PUMP state  --> ON ::: Fluid value --> ' + str(v_fluid))
                        sg.Popup('Stage finished', ['Analysis changes to RECOVERY STAGE ... '
                                                    'pulse FINISHED to conclude all experiment.'])
                        pause_ = True
                        # Save principal events
                        events = events.append({'Time': now_time1, 'Mean_area': mean_area, 'Fluid_value': v_fluid,
                                                'Pump_state': act}, ignore_index=True)
                else:
                    act = 'OFF'
                    now_time1 = now.strftime("%H : %M : %S")
                    ptg = 0.10
                    area_inf_rec, area_sup_rec = np.round(area_inf+(area_inf*ptg)), np.round(area_sup-(area_sup*ptg))
                    if mean_area < area_inf_rec or mean_area > area_sup_rec:
                        print('-------------------------------------------------------------------------')
                        print(now_time1 + ' ::: Analysis is waiting to RECOVERY experiment ..... ')
                        continue
                    else:
                        print('-------------------------------------------------------------------------')
                        print(now_time1 + ' ::: Analysis returns to NEW STAGE of experiment .....')
                        sg.Popup('New Stage', ['Analysis returns to NEW STAGE of experiment ....'])
                        # Save principal events
                        events = events.append({'Time': now_time1, 'Mean_area': mean_area, 'Fluid_value': flu_inf,
                                                'Pump_state': act}, ignore_index=True)
                        pause_ = False
                window['_PUM_'].update(act)

window.close()
