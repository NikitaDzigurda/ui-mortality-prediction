import sys
import json
import hashlib
from html import escape
from pathlib import Path
import pandas as pd
from PySide6 import QtCore, QtWidgets as W
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
from analysis import read_data, run, FEATURES, NUMERIC

DATA_COLUMNS = {
    'row_id': ('Идентификатор записи', 'Служебный номер записи; не предиктор.'),
    'group_id': ('Идентификатор пациента', 'Группирует записи одного пациента при кросс-валидации; не предиктор.'),
    'source_row': ('Индекс строки исходной базы', 'Служебная ссылка для сопоставления с исходной таблицей; не предиктор.'),
    'age_years': ('Возраст', 'Полных лет на дату операции; числовой признак.'),
    'ef_percent': ('Фракция выброса левого желудочка', 'Проценты; числовой признак.'),
    'creatinine_value': ('Креатинин', 'Концентрация из исходного столбца «креатинин»; числовой признак.'),
    'total_cholesterol': ('Общий холестерин', 'Значение из исходного столбца «холестерин»; числовой признак.'),
    'ldl_mmol_l': ('Холестерин липопротеинов низкой плотности (ЛПНП)', 'ммоль/л по заголовку исходного столбца; числовой признак.'),
    'glucose_value': ('Глюкоза', 'Уровень глюкозы из исходной базы; числовой признак.'),
    'crp_value': ('С-реактивный белок (СРБ)', 'Значение из исходного столбца «СРБ»; числовой признак.'),
    'egfr_value': ('Скорость клубочковой фильтрации (СКФ)', 'Значение из исходного столбца «СКФ»; числовой признак.'),
    'bmi': ('Индекс массы тела (ИМТ)', 'кг/м²; числовой признак.'),
    'female': ('Женский пол', '1 — женщина; 0 — мужчина. Категориальный признак, полученный по ФИО при подготовке.'),
    'prior_vascular_surgery': ('Операция на сосудах в анамнезе', '1 — была до индексного события; 0 — не было; категориальный признак.'),
    'mi_history': ('Инфаркт миокарда в анамнезе', '1 — был; 0 — не было; категориальный признак.'),
    'stroke_history': ('ОНМК в анамнезе', 'Острое нарушение мозгового кровообращения: 1 — было; 0 — не было; категориальный признак.'),
    'vascular_group': ('Сосудистый бассейн', '0 — периферия; 1 — АБШ / аневризма аорты; 2 — сонные артерии. Категории без числового порядка.'),
    'hospital_death': ('Летальный исход', '1 — смерть; 0 — жив. Целевая переменная; период наблюдения определяется источником данных.'),
}

class Worker(QtCore.QThread):
    completed = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    progress = QtCore.Signal(str)
    def __init__(self, args):
        super().__init__()
        # LLVM/Numba compilation used by SHAP needs more than macOS's default thread stack.
        self.setStackSize(64 * 1024 * 1024)
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

def explained_table(table, text):
    panel=W.QSplitter(QtCore.Qt.Orientation.Vertical)
    explanation=W.QTextBrowser()
    explanation.setHtml(text)
    explanation.setMinimumHeight(100)
    table.setMinimumHeight(150)
    panel.addWidget(explanation)
    panel.addWidget(table)
    panel.setSizes([240,600])
    panel.setStretchFactor(0,0)
    panel.setStretchFactor(1,1)
    return panel

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
        self.basin.currentTextChanged.connect(self.update_preview)
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
        self.stats_panel=explained_table(self.stats, '''
            <p><b>Статистика: связь отдельных признаков с исходом.</b>
            Каждый признак проверяется отдельно, без поправки на остальные факторы.
            Используются записи с известным исходом и значением признака; пропуски не заполняются.
            Здесь 1 — выбранное событие, 0 — его отсутствие.</p>
            <p><b>Признак</b> — название показателя.
            <b>n₀ / n₁</b> — число доступных записей без события / с событием.
            <b>Метод</b> — Манна–Уитни для числовых показателей, Фишер для двух категорий,
            χ² для нескольких категорий при всех ожидаемых частотах не менее 5.
            Для разреженных таблиц с несколькими категориями p не рассчитывается.</p>
            <p><b>Эффект</b> — для числовых признаков рангово-бисериальная корреляция r_rb
            от −1 до 1: положительный знак означает, что значения в группе события в целом выше,
            отрицательный — ниже, близость к нулю — слабую направленную разницу рангов.
            Для двух категорий выводится отношение шансов (ОШ), зависящее от порядка категорий:
            при кодировке 0/1 ОШ &gt; 1 означает более высокие шансы события при признаке 1,
            ОШ &lt; 1 — более низкие, ОШ = 1 — одинаковые шансы.
            Для χ² общий размер эффекта не рассчитывается.</p>
            <p><b>p</b> — вероятность при нулевой гипотезе получить столь же или более
            экстремальный результат проверки.
            <b>q (FDR)</b> — p после поправки Бенджамини–Хохберга по рассчитанным проверкам
            выбранных признаков текущей выборки. Принятый порог — q &lt; 0,05;
            процедура ограничивает ожидаемую долю ложных открытий среди отклонённых гипотез
            при выполнении её предпосылок. p и q не являются вероятностью истинности гипотезы.</p>
            <p><b>nan</b> — показатель не рассчитан или данных недостаточно.
            Отсутствие значимости не доказывает отсутствие связи. Проверки выполняются по записям
            без учёта повторов одного пациента. Выявленная связь не доказывает причинное влияние.</p>
        ''')
        self.importance_panel=explained_table(self.importance, '''
            <p><b>Важность: вклад признаков в качество прогноза обученной модели.</b>
            Оценка выполняется на внешних проверочных блоках, которые не использовались
            для обучения соответствующей модели и подбора её настроек.</p>
            <p><b>Признак</b> — показатель, значения которого перемешиваются между записями
            проверочного блока. Остальные признаки и исходы сохраняются; модель не переобучается.</p>
            <p><b>AP (Average Precision)</b> — средняя точность, взвешенная по приростам
            чувствительности при изменении порога прогноза: AP = Σ (Rᵢ − Rᵢ₋₁) Pᵢ,
            где P — precision (доля событий среди положительных прогнозов),
            R — recall (доля обнаруженных событий). Больше AP — лучше ранжирование событий.</p>
            <p><b>ΔAP = AP до перемешивания − AP после перемешивания.</b>
            В каждом внешнем блоке перемешивание одного признака повторяется 5 раз;
            падения AP усредняются по повторам, затем поровну по внешним блокам.
            Значение в таблице — абсолютная разность AP, а не процент изменения.</p>
            <p><b>ΔAP &gt; 0</b> — перемешивание ухудшает прогноз: модель использует полезную
            информацию признака. <b>ΔAP около 0</b> — заметного вклада не обнаружено.
            <b>ΔAP &lt; 0</b> — после перемешивания качество повышается; возможны шум,
            переобучение или нестабильность оценки. Отрицательный знак не означает защитного
            действия признака и не задаёт направление его связи с исходом.</p>
            <p>Важность зависит от модели, других признаков и проверочной выборки.
            Коррелирующие признаки могут заменять друг друга и получать низкие оценки.
            ΔAP не является p-value или доказательством причинности; по одной этой таблице
            нельзя заключать о статистической значимости или автоматически исключать признаки.</p>
        ''')
        self.figure=Figure(figsize=(10,8));self.canvas=FigureCanvasQTAgg(self.figure)
        for widget,title in [(self.preview,'Данные'),(self.metrics,'Результаты'),(self.stats_panel,'Статистика'),(self.importance_panel,'Важность'),(self.canvas,'Графики')]: self.tabs.addTab(widget,title)
        help_text=W.QTextBrowser()
        help_text.setPlainText('ПОРЯДОК РАБОТЫ\nОткройте подготовленный CSV (включая 01_данные.csv из исследования Госпитальные_исходы) или первый лист Excel. Выберите бинарный исход, код события, ID пациента и бассейн. Отметьте признаки и их категориальный тип.\n\nПРОТОКОЛ\nВложенная стратифицированная групповая кросс-валидация, seed=2026. Optuna TPE максимизирует внутренний AP. Пропуски заполняются медианой/модой только внутри train; числовые признаки стандартизуются, категории one-hot. Порог выбирается по внутренней balanced accuracy. Внешний блок не участвует в подборе.\n\nСвязи: Манна–Уитни для числовых, Фишер для бинарных, χ² для достаточных многокатегориальных таблиц. q — FDR по выбранным признакам текущей выборки, порог 0,05. Это описательные проверки без учёта повторов; для ML используйте ID пациента.\n\nВажность — падение AP при перестановке признака на внешних блоках. Это вклад в предсказание, не p-value и не причинное влияние.\n\nАДАПТАЦИЯ ИССЛЕДОВАНИЯ\nИспользованы 14 признаков, пространства RF/LR и групповой протокол дополнительного исследования с Optuna. Здесь анализируются все загруженные реальные записи без сокращения выживших. Это новый эксперимент, не воспроизведение 6000 trials, CatBoost/XGBoost и 3 повторений оригинала. Старые joblib не загружаются.\n\nМалое число событий может исключать вложенную оценку по отдельному бассейну. Сообщение об этом означает недостаточность данных для выбранного разбиения.\n\nСравнивая много наборов признаков по внешним результатам, вы постепенно подгоняете выбор под эти данные: окончательную конфигурацию следует проверять на независимой выборке. Программа предназначена для исследования, не для клинических решений.')
        self.shap_panel=W.QWidget();self.shap_layout=W.QVBoxLayout(self.shap_panel)
        self.patient=W.QComboBox();self.patient.currentIndexChanged.connect(self.draw_patient)
        self.shap_layout.addWidget(W.QLabel('Вклад в вероятность события; объяснение внешней модели. Категории кодируются числами.'))
        self.shap_layout.addWidget(self.patient)
        self.patient_canvas=None;self.beeswarm_canvas=None
        self.tabs.addTab(self.shap_panel,'SHAP пациента')
        self.beeswarm_panel=W.QWidget();self.beeswarm_layout=W.QVBoxLayout(self.beeswarm_panel)
        self.tabs.addTab(self.beeswarm_panel,'SHAP Beeswarm')
        self.data_dictionary=W.QTextBrowser()
        self.data_dictionary.setPlainText('Загрузите таблицу, чтобы увидеть расшифровку её столбцов.')
        self.tabs.insertTab(1,self.data_dictionary,'Словарь данных')
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
        self.feature_table.resizeColumnsToContents()
        self.update_dictionary()
    def update_dictionary(self):
        rows=[]
        for column in self.frame.columns:
            label,meaning=DATA_COLUMNS.get(column,('Описание не задано', 'Уточните значение и кодировку в источнике данных.'))
            rows.append('<tr>'+''.join(f'<td>{escape(value)}</td>' for value in (column,label,meaning))+'</tr>')
        source_note=('В файле 02_данные_с_обзвоном.csv столбец hospital_death содержит исход по обзвону, '
                     'включая смерти после выписки. Название сохранено для совместимости с программой.')
        self.data_dictionary.setHtml(
            '<p><b>Словарь столбцов загруженной таблицы</b></p>'
            f'<p>Файл: {escape(Path(self.source).name)}. {source_note}</p>'
            '<p><b>Пропуски:</b> nan или пустая ячейка означают отсутствие известного значения, '
            'а не отсутствие заболевания. Код 0 в бинарном признаке означает записанное отсутствие признака.</p>'
            '<p><b>Единицы измерения:</b> здесь приведены только известные единицы. Для остальных лабораторных '
            'показателей их необходимо сверять с исходной медицинской таблицей; CSV не подтверждает их автоматически. '
            'Глюкоза в этом наборе представлена числовым уровнем, без категории «повышена».</p>'
            '<table border="1" cellpadding="6" cellspacing="0" width="100%">'
            '<tr><th>Столбец</th><th>Расшифровка</th><th>Значение и кодировка</th></tr>'
            +''.join(rows)+'</table>')
    def exclude_roles(self):
        for i in range(self.feature_table.rowCount()):
            if self.feature_table.item(i,1).text() in [self.target.currentText(),self.group.currentText()]:self.feature_table.item(i,0).setCheckState(QtCore.Qt.CheckState.Unchecked)
    def update_basins(self):
        self.basin.blockSignals(True)
        self.basin.clear();self.basin.addItem('Все')
        c=self.basin_column.currentText()
        if self.frame is not None and c in self.frame:
            self.basin.addItems(sorted(self.frame[c].dropna().astype(str).unique()))
        self.basin.blockSignals(False)
        self.update_preview()
    def selected_frame(self):
        if self.frame is None:return None
        c=self.basin_column.currentText()
        if c in self.frame and self.basin.currentText() not in ('','Все'):
            return self.frame.loc[self.frame[c].astype(str).eq(self.basin.currentText())]
        return self.frame
    def update_preview(self):
        frame=self.selected_frame()
        if frame is None:return
        fill_table(self.preview,frame)
        self.preview.setVerticalHeaderLabels([str(i+1) for i in frame.index])
        self.preview.scrollToTop()
        self.info.setText(f'{Path(self.source).name}\n{len(self.frame)} строк × {len(self.frame.columns)} столбцов.\nПоказано: {len(frame)} строк; выборка: {self.basin.currentText()}.')
    def start_run(self):
        if self.frame is None:return
        features=[];cats=[]
        for i in range(self.feature_table.rowCount()):
            if self.feature_table.item(i,0).checkState()==QtCore.Qt.CheckState.Checked:
                c=self.feature_table.item(i,1).text();features.append(c)
                if self.feature_table.item(i,2).checkState()==QtCore.Qt.CheckState.Checked:cats.append(c)
        frame=self.selected_frame().copy()
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
        import shap
        import matplotlib.pyplot as plt
        explanation=shap.Explanation(values=result['shap_values'].to_numpy(),base_values=result['shap_base']['base'].to_numpy(),data=result['shap_data'].to_numpy(float),feature_names=result['features'])
        self.explanation=explanation
        plt.figure(figsize=(10,7))
        ax=shap.plots.beeswarm(explanation,max_display=len(result['features']),show=False)
        fig=ax.figure;fig.tight_layout();plt.close(fig)
        if self.beeswarm_canvas:self.beeswarm_layout.removeWidget(self.beeswarm_canvas);self.beeswarm_canvas.deleteLater()
        self.beeswarm_canvas=FigureCanvasQTAgg(fig);self.beeswarm_layout.addWidget(self.beeswarm_canvas)
        self.patient.blockSignals(True);self.patient.clear()
        self.patient.addItems([f'Строка {idx} · p={prob:.3f}' for idx,prob in zip(p.row_index,p.probability)])
        self.patient.blockSignals(False);self.draw_patient(0)
    def draw_patient(self,index):
        if index<0 or not hasattr(self,'explanation'):return
        import shap
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,7))
        ax=shap.plots.waterfall(self.explanation[index],max_display=len(self.result['features']),show=False)
        fig=ax.figure;fig.tight_layout();plt.close(fig)
        if self.patient_canvas:self.shap_layout.removeWidget(self.patient_canvas);self.patient_canvas.deleteLater()
        self.patient_canvas=FigureCanvasQTAgg(fig);self.shap_layout.addWidget(self.patient_canvas)
    def save(self):
        path=W.QFileDialog.getExistingDirectory(self,'Папка для нового каталога результатов')
        if not path or not self.result:return
        import datetime
        output=Path(path)/datetime.datetime.now().strftime('analysis_%Y%m%d_%H%M%S_%f');output.mkdir()
        meta={k:v for k,v in self.result.items() if not isinstance(v,pd.DataFrame)}
        (output/'protocol.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
        for key in ['stats','importance','predictions']:self.result[key].to_csv(output/f'{key}.csv',index=False)
        self.figure.savefig(output/'results.png',dpi=180)
        for key in ['shap_values','shap_data','shap_base']:self.result[key].to_csv(output/f'{key}.csv',index=False)
        self.beeswarm_canvas.figure.savefig(output/'shap_beeswarm.png',dpi=180)
        self.patient_canvas.figure.savefig(output/f'shap_patient_{self.patient.currentIndex()}.png',dpi=180)
        self.statusBar().showMessage(f'Сохранено: {output}')
    def closeEvent(self,event):
        if self.worker and self.worker.isRunning():
            self.cancel();event.ignore();self.error('Расчёт отменяется. Закройте окно после завершения текущего обучения.')
        else:event.accept()

if __name__=='__main__':
    app=W.QApplication(sys.argv);app.setStyle('Fusion');window=Window();window.show();sys.exit(app.exec())
