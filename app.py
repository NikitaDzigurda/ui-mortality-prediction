import sys
import json
import hashlib
from pathlib import Path
import pandas as pd
from PySide6 import QtCore, QtWidgets as W
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
from analysis import read_data, run, FEATURES, NUMERIC

class Worker(QtCore.QThread):
    completed = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    progress = QtCore.Signal(str)
    def __init__(self, args):
        super().__init__()
        self.args = args
    def run(self):
        try:
            self.completed.emit(run(**self.args, progress=self.progress.emit, cancelled=self.isInterruptionRequested))
        except Exception as exc:
            self.failed.emit(str(exc))

def fill_table(widget, data):
    widget.setRowCount(len(data))
    widget.setColumnCount(len(data.columns))
    widget.setHorizontalHeaderLabels(list(map(str,data.columns)))
    for i,row in enumerate(data.itertuples(index=False,name=None)):
        for j,value in enumerate(row):
            text = f'{value:.5g}' if isinstance(value,float) else str(value)
            widget.setItem(i,j,W.QTableWidgetItem(text))
    widget.resizeColumnsToContents()

class Window(W.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Сосудистые исходы · локальная исследовательская программа')
        self.resize(1350,900)
        self.frame=None
        self.result=None
        self.worker=None
        root=W.QWidget(); self.setCentralWidget(root)
        layout=W.QHBoxLayout(root)
        controls=W.QWidget(); form=W.QFormLayout(controls)
        controls.setMaximumWidth(430)
        load=W.QPushButton('Открыть CSV / Excel');load.clicked.connect(self.load)
        form.addRow(load)
        self.info=W.QLabel('Загрузите подготовленную таблицу.\nВсе вычисления выполняются локально.');self.info.setWordWrap(True)
        form.addRow(self.info)
        self.target=W.QComboBox();self.positive=W.QComboBox();self.positive.addItems(['1','0'])
        self.group=W.QComboBox();self.basin_column=W.QComboBox();self.basin=W.QComboBox()
        for label,widget in [('Столбец исхода',self.target),('Событие кодируется',self.positive),('ID пациента',self.group),('Столбец бассейна',self.basin_column),('Выборка',self.basin)]:form.addRow(label,widget)
        self.basin_column.currentTextChanged.connect(self.update_basins)
        self.target.currentTextChanged.connect(self.exclude_roles)
        self.group.currentTextChanged.connect(self.exclude_roles)
        self.feature_table=W.QTableWidget(0,3)
        self.feature_table.setHorizontalHeaderLabels(['Использовать','Признак','Категория'])
        form.addRow(self.feature_table)
        self.family=W.QComboBox();self.family.addItems(['RF','LR'])
        self.trials=W.QSpinBox();self.trials.setRange(1,200);self.trials.setValue(10)
        self.folds=W.QSpinBox();self.folds.setRange(2,5);self.folds.setValue(3)
        form.addRow('Модель',self.family);form.addRow('Optuna trials / блок',self.trials);form.addRow('Внешних / внутренних блоков',self.folds)
        self.start=W.QPushButton('Рассчитать выбранную конфигурацию');self.start.clicked.connect(self.start_run)
        self.stop=W.QPushButton('Отменить');self.stop.clicked.connect(self.cancel)
        self.export=W.QPushButton('Экспорт результатов');self.export.clicked.connect(self.save)
        self.export.setEnabled(False);self.stop.setEnabled(False)
        form.addRow(self.start);form.addRow(self.stop);form.addRow(self.export)
        layout.addWidget(controls)
        self.tabs=W.QTabWidget();layout.addWidget(self.tabs,1)
        self.preview=W.QTableWidget();self.metrics=W.QTextBrowser();self.stats=W.QTableWidget();self.importance=W.QTableWidget()
        self.figure=Figure(figsize=(10,8));self.canvas=FigureCanvasQTAgg(self.figure)
        for widget,title in [(self.preview,'Данные'),(self.metrics,'Результаты'),(self.stats,'Статистика'),(self.importance,'Важность'),(self.canvas,'Графики')]: self.tabs.addTab(widget,title)
        help_text=W.QTextBrowser()
        help_text.setPlainText('ПОРЯДОК РАБОТЫ\nОткройте подготовленный CSV (включая 01_данные.csv из исследования Госпитальные_исходы) или первый лист Excel. Выберите бинарный исход, код события, ID пациента и бассейн. Отметьте признаки и их категориальный тип.\n\nПРОТОКОЛ\nВложенная стратифицированная групповая кросс-валидация, seed=2026. Optuna TPE максимизирует внутренний AP. Пропуски заполняются медианой/модой только внутри train; числовые признаки стандартизуются, категории one-hot. Порог выбирается по внутренней balanced accuracy. Внешний блок не участвует в подборе.\n\nСвязи: Манна–Уитни для числовых, Фишер для бинарных, χ² для достаточных многокатегориальных таблиц. q — FDR по выбранным признакам текущей выборки, порог 0,05. Это описательные проверки без учёта повторов; для ML используйте ID пациента.\n\nВажность — падение AP при перестановке признака на внешних блоках. Это вклад в предсказание, не p-value и не причинное влияние.\n\nАДАПТАЦИЯ ИССЛЕДОВАНИЯ\nИспользованы 14 признаков, пространства RF/LR и групповой протокол дополнительного исследования с Optuna. Здесь анализируются все загруженные реальные записи без сокращения выживших. Это новый эксперимент, не воспроизведение 6000 trials, CatBoost/XGBoost и 3 повторений оригинала. Старые joblib не загружаются.\n\nМалое число событий может исключать вложенную оценку по отдельному бассейну. Сообщение об этом означает недостаточность данных для выбранного разбиения.\n\nСравнивая много наборов признаков по внешним результатам, вы постепенно подгоняете выбор под эти данные: окончательную конфигурацию следует проверять на независимой выборке. Программа предназначена для исследования, не для клинических решений.')
        self.tabs.addTab(help_text,'Методика')
    def error(self,text):
        W.QMessageBox.warning(self,'Сообщение',text)
    def load(self):
        path,_=W.QFileDialog.getOpenFileName(self,'Открыть таблицу','','Таблицы (*.csv *.xlsx *.xls)')
        if not path:return
        try:frame=read_data(path)
        except Exception as exc:self.error(str(exc));return
        self.frame=frame;self.source=path
        self.feature_table.setRowCount(len(frame.columns))
        for i,c in enumerate(frame.columns):
            use=W.QTableWidgetItem();use.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled|QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            use.setCheckState(QtCore.Qt.CheckState.Checked if c in FEATURES else QtCore.Qt.CheckState.Unchecked)
            cat=W.QTableWidgetItem();cat.setFlags(use.flags());cat.setCheckState(QtCore.Qt.CheckState.Checked if c in FEATURES[9:] or (c not in NUMERIC and not pd.api.types.is_numeric_dtype(frame[c])) else QtCore.Qt.CheckState.Unchecked)
            self.feature_table.setItem(i,0,use);self.feature_table.setItem(i,1,W.QTableWidgetItem(c));self.feature_table.setItem(i,2,cat)
        for box,empty in [(self.target,False),(self.group,True),(self.basin_column,True)]:
            box.clear();box.addItems((['—'] if empty else [])+list(frame.columns))
        if 'hospital_death' in frame:self.target.setCurrentText('hospital_death')
        for c in ['group_id','patient_group_id']:
            if c in frame:self.group.setCurrentText(c);break
        if 'vascular_group' in frame:self.basin_column.setCurrentText('vascular_group')
        self.exclude_roles();self.update_basins()
        fill_table(self.preview,frame.head(100));self.feature_table.resizeColumnsToContents()
        self.info.setText(f'{Path(path).name}\n{len(frame)} строк × {len(frame.columns)} столбцов; показаны первые 100.')
    def exclude_roles(self):
        for i in range(self.feature_table.rowCount()):
            if self.feature_table.item(i,1).text() in [self.target.currentText(),self.group.currentText()]:self.feature_table.item(i,0).setCheckState(QtCore.Qt.CheckState.Unchecked)
    def update_basins(self):
        self.basin.clear();self.basin.addItem('Все')
        c=self.basin_column.currentText()
        if self.frame is not None and c in self.frame:
            self.basin.addItems(sorted(self.frame[c].dropna().astype(str).unique()))
    def start_run(self):
        if self.frame is None:return
        features=[];cats=[]
        for i in range(self.feature_table.rowCount()):
            if self.feature_table.item(i,0).checkState()==QtCore.Qt.CheckState.Checked:
                c=self.feature_table.item(i,1).text();features.append(c)
                if self.feature_table.item(i,2).checkState()==QtCore.Qt.CheckState.Checked:cats.append(c)
        frame=self.frame.copy()
        if self.basin.currentText()!='Все':frame=frame.loc[frame[self.basin_column.currentText()].astype(str).eq(self.basin.currentText())]
        if self.group.currentText()=='—':
            if W.QMessageBox.question(self,'Независимость наблюдений','ID пациента не выбран. Подтверждаете, что каждая строка — отдельный пациент?')!=W.QMessageBox.StandardButton.Yes:return
        self.run_context={'source_name':Path(self.source).name,
                          'source_sha256':hashlib.sha256(Path(self.source).read_bytes()).hexdigest(),
                          'basin_column':self.basin_column.currentText(), 'basin':self.basin.currentText(),
                          'seed':2026,'strategy':'all_real','importance':'external-fold permutation AP'}
        self.worker=Worker(dict(frame=frame,features=features,categorical=cats,target=self.target.currentText(),positive=int(self.positive.currentText()),group_column=None if self.group.currentText()=='—' else self.group.currentText(),family=self.family.currentText(),trials=self.trials.value(),folds=self.folds.value()))
        self.centralWidget().layout().itemAt(0).widget().setEnabled(False)
        self.statusBar().showMessage('Подготовка расчёта…')
        self.cancel_action=self.menuBar().addAction('Отменить расчёт');self.cancel_action.triggered.connect(self.cancel)
        self.worker.progress.connect(self.statusBar().showMessage)
        self.worker.completed.connect(self.show_result);self.worker.failed.connect(self.error);self.worker.finished.connect(self.finished)
        self.worker.start()
    def cancel(self):
        if self.worker:self.worker.requestInterruption();self.statusBar().showMessage('Отмена после текущего обучения…')
    def finished(self):
        self.centralWidget().layout().itemAt(0).widget().setEnabled(True)
        self.menuBar().removeAction(self.cancel_action)
        self.statusBar().showMessage('Расчёт завершён')
    def show_result(self,result):
        result['context']=self.run_context
        self.result=result;self.export.setEnabled(True)
        self.metrics.setPlainText('\n'.join(f'{k}: {v:.5g}' for k,v in result['metrics'].items())+'\n\nПараметры внешних блоков:\n'+json.dumps(result['configs'],ensure_ascii=False,indent=2))
        fill_table(self.stats,result['stats']);fill_table(self.importance,result['importance'].sort_values('ΔAP',ascending=False))
        self.figure.clear();axes=self.figure.subplots(2,2);p=result['predictions'];y=p.outcome
        RocCurveDisplay.from_predictions(y,p.probability,ax=axes[0,0])
        PrecisionRecallDisplay.from_predictions(y,p.probability,ax=axes[0,1]);axes[0,1].axhline(y.mean(),linestyle='--',color='gray')
        ConfusionMatrixDisplay.from_predictions(y,p.prediction,ax=axes[1,0],colorbar=False)
        importance=result['importance'].sort_values('ΔAP');axes[1,1].barh(importance['Признак'],importance['ΔAP']);axes[1,1].set_title('Перестановочная важность: ΔAP')
        self.figure.tight_layout();self.canvas.draw();self.tabs.setCurrentWidget(self.metrics)
    def save(self):
        path=W.QFileDialog.getExistingDirectory(self,'Папка для нового каталога результатов')
        if not path or not self.result:return
        import datetime
        output=Path(path)/datetime.datetime.now().strftime('analysis_%Y%m%d_%H%M%S_%f');output.mkdir()
        meta={k:v for k,v in self.result.items() if not isinstance(v,pd.DataFrame)}
        (output/'protocol.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
        for key in ['stats','importance','predictions']:self.result[key].to_csv(output/f'{key}.csv',index=False)
        self.figure.savefig(output/'results.png',dpi=180)
        self.statusBar().showMessage(f'Сохранено: {output}')
    def closeEvent(self,event):
        if self.worker and self.worker.isRunning():
            self.cancel();event.ignore();self.error('Расчёт отменяется. Закройте окно после завершения текущего обучения.')
        else:event.accept()

if __name__=='__main__':
    app=W.QApplication(sys.argv);app.setStyle('Fusion');window=Window();window.show();sys.exit(app.exec())
