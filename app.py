import sys
import json
import hashlib
import textwrap
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

PATIENT_COLUMN = 'group_id'
BASIN_COLUMN = 'vascular_group'
EVENT_VALUE = 1
BASIN_NAMES = {0: 'Периферия', 1: 'АБШ / аневризма аорты', 2: 'Сонные артерии'}

def feature_label(column):
    return DATA_COLUMNS.get(column, (column, ''))[0]

def plot_label(column):
    return textwrap.fill(feature_label(column), width=24)

def basin_label(value):
    try:
        code=float(value)
        return BASIN_NAMES.get(code, str(value))
    except (TypeError, ValueError):
        return str(value)

def labeled_features(data):
    result=data.copy()
    result['Признак']=result['Признак'].map(feature_label)
    return result

def summarize_shap(values):
    absolute=values.abs()
    return pd.DataFrame({'Столбец':values.columns,
                         'Признак':[feature_label(c) for c in values.columns],
                         'Средний |SHAP|':absolute.mean().to_numpy(),
                         'Сумма |SHAP|':absolute.sum().to_numpy()})

class CenteredFigureCanvas(FigureCanvasQTAgg):
    def draw(self):
        super().draw()
        bounds=self.figure.get_tightbbox(self.get_renderer())
        if bounds is None:return
        width=self.figure.get_figwidth()
        shift=(width/2-(bounds.x0+bounds.x1)/2)/width
        margins=self.figure.subplotpars
        shift=max(.02-margins.left,min(.98-margins.right,shift))
        if abs(shift)>.002:
            self.figure.subplots_adjust(left=margins.left+shift,right=margins.right+shift)
            super().draw()

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
        self.target=W.QComboBox();self.basin=W.QComboBox()
        form.addRow('Исход',self.target);form.addRow('Выборка',self.basin)
        self.basin.currentTextChanged.connect(self.update_preview)
        self.target.currentTextChanged.connect(self.exclude_roles)
        self.feature_table=W.QTableWidget(0,3)
        self.feature_table.setHorizontalHeaderLabels(['Использовать','Признак','Категория'])
        header=self.feature_table.horizontalHeader()
        header.setSectionResizeMode(0,W.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1,W.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2,W.QHeaderView.ResizeMode.ResizeToContents)
        form.addRow(self.feature_table)
        self.family=W.QComboBox()
        self.family.addItem('Случайный лес','RF')
        self.family.addItem('Логистическая регрессия','LR')
        self.trials=W.QSpinBox();self.trials.setRange(1,200);self.trials.setValue(10)
        self.folds=W.QSpinBox();self.folds.setRange(2,5);self.folds.setValue(3)
        self.trials.setToolTip('Сколько вариантов настроек Optuna проверит в каждой внешней обучающей части. Больше вариантов — дольше расчёт.')
        self.folds.setToolTip('Число частей для внешней проверки и внутреннего подбора настроек. Одно значение применяется на обоих уровнях кросс-валидации.')
        form.addRow('Модель',self.family);form.addRow('Вариантов настроек модели',self.trials);form.addRow('Блоков кросс-валидации',self.folds)
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
        self.figure=Figure(figsize=(10,8),layout='constrained');self.canvas=FigureCanvasQTAgg(self.figure)
        for widget,title in [(self.preview,'Данные'),(self.metrics,'Результаты'),(self.stats_panel,'Статистика'),(self.importance_panel,'Важность'),(self.canvas,'Графики')]: self.tabs.addTab(widget,title)
        help_text=W.QTextBrowser()
        help_text.setHtml('''
            <h3>Как работать с программой</h3>
            <ol>
            <li><b>Загрузите данные.</b> Нажмите «Открыть CSV / Excel» и выберите подготовленную таблицу.
            В Excel читается первый лист. Для текущего исследования используйте 02_данные_с_обзвоном.csv.
            Проверьте число строк на вкладке «Данные». Пропуски отображаются как nan.</li>
            <li><b>Выберите исход и бассейн.</b> Для анализа смерти оставьте «Летальный исход».
            В поле «Выборка» выберите все записи или нужный сосудистый бассейн: предпросмотр сразу обновится.
            Смерть автоматически кодируется значением 1, повторные записи группируются по group_id,
            бассейн определяется по vascular_group. Эти настройки не требуется задавать вручную.</li>
            <li><b>Укажите факторы.</b> Галочка «Использовать» включает показатель в статистику и модель.
            Галочка «Категория» задаёт категориальный тип, в том числе для бинарных признаков.
            Возраст и лабораторные показатели оставьте числовыми. Расшифровки приведены в «Словаре данных».
            Исход и служебные идентификаторы не включайте в предикторы.</li>
            <li><b>Настройте обучение.</b> Выберите «Случайный лес» или «Логистическая регрессия».
            «Вариантов настроек модели» — число комбинаций, которые проверит Optuna в каждом внешнем блоке.
            «Блоков кросс-валидации» — число частей на обоих уровнях: внутренний подбирает настройки,
            внешний проверяет их на других пациентах. Для первого запуска оставьте 10 вариантов и 3 блока.
            Больше значений — дольше расчёт; при редких смертях разбиение может оказаться невозможным.</li>
            <li><b>Запустите расчёт.</b> Нажмите «Рассчитать выбранную конфигурацию».
            Следите за этапами внизу окна; для остановки используйте «Отменить расчёт» в верхнем меню.
            Текущее обучение может завершиться перед остановкой.</li>
            <li><b>Оцените результат.</b> «Результаты» показывает метрики и настройки внешних моделей.
            Recall — доля найденных смертей, специфичность — доля правильно определённых выживших,
            precision — доля истинных смертей среди прогнозов смерти. AP и ROC AUC оценивают ранжирование,
            Brier — ошибку вероятностей (меньше лучше). На «Графиках» доступны ROC, Precision–Recall,
            матрица ошибок и важность факторов. Высокий recall нужно оценивать вместе с ложными тревогами.</li>
            <li><b>Изучите связи и вклады.</b> «Статистика» сравнивает каждый фактор с исходом;
            основной порог — q &lt; 0,05 после поправки FDR. «Важность» показывает падение AP при
            перемешивании признака. Это разные оценки; пояснения расположены над таблицами.
            В SHAP пациента выберите запись: по умолчанию показаны только умершие. Снимите галочку,
            чтобы увидеть все записи. В поле «Представление» можно выбрать «По всем пациентам выборки»:
            появится общий рейтинг факторов по среднему или суммарному абсолютному SHAP.
            В этом режиме учитываются все рассчитанные записи, включая выживших; фильтр умерших скрыт.
            Красные индивидуальные вклады повышают вероятность события, синие снижают.
            SHAP Beeswarm показывает распределение вкладов по всей рассчитанной выборке;
            фильтр умерших на индивидуальной вкладке его не меняет.</li>
            <li><b>Сравните конфигурации и сохраните результат.</b> После смены модели, факторов или
            бассейна запустите расчёт заново; до этого результаты относятся к предыдущему запуску.
            Нажмите «Экспорт результатов» и выберите папку. Сохраняются таблицы, протокол, графики,
            общий SHAP и индивидуальный SHAP выбранной записи.</li>
            </ol>
            <h3>Как устроена проверка</h3>
            <p>Используется вложенная стратифицированная групповая кросс-валидация, seed 2026.
            Optuna выбирает настройки по внутреннему AP, порог — по внутренней balanced accuracy.
            Заполнение пропусков медианой/модой, масштабирование и кодирование обучаются внутри
            обучающих частей. Внешние пациенты не участвуют в подборе своей модели.
            Это отдельный расчёт на реальных записях, без синтетики и сокращения выживших.</p>
            <p>Статистика использует известные значения: Манна–Уитни для числовых признаков,
            Фишер для двух категорий, χ² для достаточных многокатегориальных таблиц.
            FDR относится к выбранным факторам текущего запуска. Отдельные тесты зависимость
            повторных записей пациента не учитывают. SHAP и важность объясняют модель,
            но не доказывают причинное влияние. Для клинических решений нужна независимая проверка.</p>
        ''')
        self.shap_panel=W.QWidget();self.shap_layout=W.QVBoxLayout(self.shap_panel)
        self.shap_mode=W.QComboBox()
        self.shap_mode.addItem('Отдельный пациент','patient')
        self.shap_mode.addItem('По всем пациентам выборки','summary')
        mode_form=W.QFormLayout();mode_form.addRow('Представление',self.shap_mode)
        self.shap_layout.addLayout(mode_form)
        self.shap_aggregation=W.QComboBox()
        self.shap_aggregation.addItem('Средний абсолютный SHAP','Средний |SHAP|')
        self.shap_aggregation.addItem('Суммарный абсолютный SHAP','Сумма |SHAP|')
        self.shap_aggregation.setToolTip('Среднее или сумма модулей вкладов по всем записям рассчитанной выборки. Это важность модели, а не p-value. Сумма не является вероятностью.')
        self.shap_aggregation.hide()
        self.shap_layout.addWidget(self.shap_aggregation)
        self.patient=W.QComboBox();self.patient.currentIndexChanged.connect(self.draw_patient)
        self.patient.setPlaceholderText('Выполните расчёт, чтобы выбрать запись.')
        self.deaths_only=W.QCheckBox('Только пациенты с летальным исходом')
        self.deaths_only.setChecked(True)
        self.deaths_only.toggled.connect(self.update_patients)
        self.shap_layout.addWidget(self.deaths_only)
        self.shap_layout.addWidget(self.patient)
        self.patient_canvas=None;self.beeswarm_canvas=None
        self.shap_mode.currentIndexChanged.connect(self.update_shap_view)
        self.shap_aggregation.currentIndexChanged.connect(self.update_shap_view)
        self.tabs.addTab(self.shap_panel,'SHAP пациента')
        self.beeswarm_panel=W.QWidget();self.beeswarm_layout=W.QVBoxLayout(self.beeswarm_panel)
        self.tabs.addTab(self.beeswarm_panel,'SHAP Beeswarm')
        self.data_dictionary=W.QTextBrowser()
        self.data_dictionary.setPlainText('Загрузите таблицу, чтобы увидеть расшифровку её столбцов.')
        self.tabs.insertTab(1,self.data_dictionary,'Словарь данных')
        self.tabs.insertTab(0,help_text,'Инструкция')
        self.tabs.setCurrentWidget(help_text)
    def error(self,text):
        W.QMessageBox.warning(self,'Сообщение',text)
    def load(self):
        path,_=W.QFileDialog.getOpenFileName(self,'Открыть таблицу','','Таблицы (*.csv *.xlsx *.xls)')
        if not path:return
        try:frame=read_data(path)
        except Exception as exc:self.error(str(exc));return
        missing=[c for c in (PATIENT_COLUMN,BASIN_COLUMN) if c not in frame]
        if missing:
            self.error('В подготовленной таблице отсутствуют обязательные столбцы: '+', '.join(missing))
            return
        self.frame=frame;self.source=path
        self.feature_table.setRowCount(len(frame.columns))
        for i,c in enumerate(frame.columns):
            use=W.QTableWidgetItem();use.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled|QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            use.setCheckState(QtCore.Qt.CheckState.Checked if c in FEATURES else QtCore.Qt.CheckState.Unchecked)
            cat=W.QTableWidgetItem();cat.setFlags(use.flags());cat.setCheckState(QtCore.Qt.CheckState.Checked if c in FEATURES[9:] or (c not in NUMERIC and not pd.api.types.is_numeric_dtype(frame[c])) else QtCore.Qt.CheckState.Unchecked)
            name=W.QTableWidgetItem(feature_label(c));name.setData(QtCore.Qt.ItemDataRole.UserRole,c);name.setToolTip(c)
            self.feature_table.setItem(i,0,use);self.feature_table.setItem(i,1,name);self.feature_table.setItem(i,2,cat)
        self.target.blockSignals(True);self.target.clear()
        for c in frame.columns:self.target.addItem(feature_label(c),c)
        if 'hospital_death' in frame:self.target.setCurrentIndex(self.target.findData('hospital_death'))
        self.target.blockSignals(False)
        self.exclude_roles();self.update_basins()
        self.feature_table.resizeRowsToContents()
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
            if self.feature_table.item(i,1).data(QtCore.Qt.ItemDataRole.UserRole) in [self.target.currentData(),PATIENT_COLUMN,'row_id','source_row']:
                self.feature_table.item(i,0).setCheckState(QtCore.Qt.CheckState.Unchecked)
    def update_basins(self):
        self.basin.blockSignals(True)
        self.basin.clear();self.basin.addItem('Все',None)
        c=BASIN_COLUMN
        if self.frame is not None and c in self.frame:
            for value in sorted(self.frame[c].dropna().astype(str).unique()):
                self.basin.addItem(basin_label(value),value)
        self.basin.blockSignals(False)
        self.update_preview()
    def selected_frame(self):
        if self.frame is None:return None
        c=BASIN_COLUMN
        if c in self.frame and self.basin.currentData() is not None:
            return self.frame.loc[self.frame[c].astype(str).eq(self.basin.currentData())]
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
                c=self.feature_table.item(i,1).data(QtCore.Qt.ItemDataRole.UserRole);features.append(c)
                if self.feature_table.item(i,2).checkState()==QtCore.Qt.CheckState.Checked:cats.append(c)
        frame=self.selected_frame().copy()
        self.run_context={'source_name':Path(self.source).name,
                          'source_sha256':hashlib.sha256(Path(self.source).read_bytes()).hexdigest(),
                          'basin_column':BASIN_COLUMN, 'basin':self.basin.currentData(), 'basin_label':self.basin.currentText(),
                          'seed':2026,'strategy':'all_real','importance':'external-fold permutation AP'}
        self.worker=Worker(dict(frame=frame,features=features,categorical=cats,target=self.target.currentData(),positive=EVENT_VALUE,group_column=PATIENT_COLUMN,family=self.family.currentData(),trials=self.trials.value(),folds=self.folds.value()))
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
        result['shap_summary']=summarize_shap(result['shap_values'])
        self.result=result;self.export.setEnabled(True)
        self.metrics.setPlainText('\n'.join(f'{k}: {v:.5g}' for k,v in result['metrics'].items())+'\n\nПараметры внешних блоков:\n'+json.dumps(result['configs'],ensure_ascii=False,indent=2))
        fill_table(self.stats,labeled_features(result['stats']))
        fill_table(self.importance,labeled_features(result['importance'].sort_values('ΔAP',ascending=False)))
        self.figure.clear();axes=self.figure.subplots(2,2);p=result['predictions'];y=p.outcome
        RocCurveDisplay.from_predictions(y,p.probability,ax=axes[0,0],name='Модель')
        axes[0,0].set(title='ROC-кривая',xlabel='Доля ложных тревог',ylabel='Чувствительность')
        PrecisionRecallDisplay.from_predictions(y,p.probability,ax=axes[0,1],name='Модель');axes[0,1].axhline(y.mean(),linestyle='--',color='gray')
        axes[0,1].set(title='Точность и чувствительность',xlabel='Чувствительность',ylabel='Точность положительных прогнозов')
        death_target=result['target']=='hospital_death'
        ConfusionMatrixDisplay.from_predictions(y,p.prediction,ax=axes[1,0],colorbar=False,display_labels=['Жив','Умер'] if death_target else ['Нет события','Событие'])
        axes[1,0].set(title='Матрица ошибок',xlabel='Прогноз',ylabel='Фактический исход')
        importance=result['importance'].sort_values('ΔAP');axes[1,1].barh(importance['Признак'].map(plot_label),importance['ΔAP']);axes[1,1].set_title('Перестановочная важность: ΔAP');axes[1,1].tick_params(axis='y',labelsize=8)
        self.canvas.draw();self.tabs.setCurrentWidget(self.metrics)
        import shap
        import matplotlib.pyplot as plt
        explanation=shap.Explanation(values=result['shap_values'].to_numpy(),base_values=result['shap_base']['base'].to_numpy(),data=result['shap_data'].to_numpy(float),feature_names=[plot_label(c) for c in result['features']])
        self.explanation=explanation
        plt.figure(figsize=(12,10))
        ax=shap.plots.beeswarm(explanation,max_display=len(result['features']),show=False,plot_size=None)
        ax.set_xlabel('Вклад SHAP в вероятность события');ax.tick_params(axis='y',labelsize=9)
        for color_ax in ax.figure.axes[1:]:
            color_ax.set_ylabel('Значение признака');color_ax.set_yticks([0,1],labels=['Низкое','Высокое'])
        fig=ax.figure;fig.subplots_adjust(left=.37,right=.91,bottom=.09,top=.97);plt.close(fig)
        if self.beeswarm_canvas:self.beeswarm_layout.removeWidget(self.beeswarm_canvas);self.beeswarm_canvas.hide();self.beeswarm_canvas.deleteLater()
        self.beeswarm_canvas=FigureCanvasQTAgg(fig);self.beeswarm_layout.addWidget(self.beeswarm_canvas)
        self.deaths_only.setText('Только пациенты с летальным исходом' if death_target else 'Только записи с событием')
        self.update_patients()
    def update_patients(self):
        if self.result is None or not hasattr(self,'explanation'):return
        selected=self.patient.currentData()
        self.patient.blockSignals(True);self.patient.clear()
        predictions=self.result['predictions']
        death_target=self.result['target']=='hospital_death'
        for position,row in enumerate(predictions.itertuples(index=False)):
            if self.deaths_only.isChecked() and row.outcome!=1:continue
            status=('умер' if row.outcome==1 else 'жив') if death_target else ('есть событие' if row.outcome==1 else 'нет события')
            self.patient.addItem(f'Строка {row.row_index} · {status} · p={row.probability:.3f}',position)
        index=self.patient.findData(selected) if selected is not None else -1
        self.patient.setCurrentIndex(index if index>=0 else (0 if self.patient.count() else -1))
        self.patient.setEnabled(self.patient.count()>0)
        self.patient.setPlaceholderText('Записей с выбранным исходом нет.')
        self.patient.blockSignals(False);self.draw_patient(self.patient.currentIndex())
    def update_shap_view(self):
        summary=self.shap_mode.currentData()=='summary'
        self.deaths_only.setVisible(not summary)
        self.patient.setVisible(not summary)
        self.shap_aggregation.setVisible(summary)
        self.draw_patient(self.patient.currentIndex())
    def make_summary_figure(self):
        metric=self.shap_aggregation.currentData()
        table=self.result['shap_summary'].sort_values(metric)
        fig=Figure(figsize=(12,10))
        ax=fig.subplots()
        bars=ax.barh(table['Столбец'].map(plot_label),table[metric],color='#527A95')
        ax.bar_label(bars,labels=[f'{v:.4g}' for v in table[metric]],padding=4,fontsize=9)
        ax.set_title(f"Важность факторов по всем записям (n={len(self.result['predictions'])})")
        ax.set_xlabel('Средний |SHAP| = Σ |вклад| / n' if metric=='Средний |SHAP|' else 'Сумма |SHAP| = Σ |вклад| по записям')
        ax.tick_params(axis='y',labelsize=9);ax.margins(x=.2)
        fig.subplots_adjust(left=.42,right=.93,bottom=.12,top=.9)
        return fig
    def draw_patient(self,index):
        if self.shap_mode.currentData()=='summary' and self.result is not None and 'shap_summary' in self.result:
            fig=self.make_summary_figure()
            if self.patient_canvas:self.shap_layout.removeWidget(self.patient_canvas);self.patient_canvas.hide();self.patient_canvas.deleteLater()
            self.patient_canvas=CenteredFigureCanvas(fig);self.shap_layout.addWidget(self.patient_canvas)
            return
        if index<0 or not hasattr(self,'explanation'):
            if self.patient_canvas:
                self.shap_layout.removeWidget(self.patient_canvas);self.patient_canvas.hide();self.patient_canvas.deleteLater();self.patient_canvas=None
            return
        position=self.patient.itemData(index)
        if position is None:return
        import shap
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,10))
        ax=shap.plots.waterfall(self.explanation[position],max_display=len(self.result['features']),show=False)
        fig=ax.figure;fig.set_size_inches(12,10,forward=False)
        fig.axes[0].set_xlabel('Вероятность события',labelpad=22);fig.axes[0].tick_params(axis='y',labelsize=9)
        fig.subplots_adjust(left=.5,right=.97,bottom=.12,top=.9);plt.close(fig)
        if self.patient_canvas:self.shap_layout.removeWidget(self.patient_canvas);self.patient_canvas.hide();self.patient_canvas.deleteLater()
        self.patient_canvas=CenteredFigureCanvas(fig);self.shap_layout.addWidget(self.patient_canvas)
    def save(self):
        path=W.QFileDialog.getExistingDirectory(self,'Папка для нового каталога результатов')
        if not path or not self.result:return
        import datetime
        output=Path(path)/datetime.datetime.now().strftime('analysis_%Y%m%d_%H%M%S_%f');output.mkdir()
        meta={k:v for k,v in self.result.items() if not isinstance(v,pd.DataFrame)}
        position=self.patient.currentData()
        summary_view=self.shap_mode.currentData()=='summary'
        meta['shap_view']=self.shap_mode.currentData()
        meta['shap_summary_metric']=self.shap_aggregation.currentData()
        meta['shap_summary_n']=len(self.result['predictions'])
        meta['selected_shap_patient']=(dict(position=position,row_index=int(self.result['predictions'].iloc[position].row_index),only_events=self.deaths_only.isChecked()) if position is not None and not summary_view else None)
        (output/'protocol.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
        for key in ['stats','importance','predictions']:self.result[key].to_csv(output/f'{key}.csv',index=False)
        self.figure.savefig(output/'results.png',dpi=180)
        for key in ['shap_values','shap_data','shap_base','shap_summary']:self.result[key].to_csv(output/f'{key}.csv',index=False)
        self.beeswarm_canvas.figure.savefig(output/'shap_beeswarm.png',dpi=180)
        summary_figure=self.patient_canvas.figure if summary_view and self.patient_canvas else self.make_summary_figure()
        summary_figure.savefig(output/'shap_summary.png',dpi=180)
        if self.patient_canvas and position is not None and not summary_view:self.patient_canvas.figure.savefig(output/f'shap_patient_{position}.png',dpi=180)
        self.statusBar().showMessage(f'Сохранено: {output}')
    def closeEvent(self,event):
        if self.worker and self.worker.isRunning():
            self.cancel();event.ignore();self.error('Расчёт отменяется. Закройте окно после завершения текущего обучения.')
        else:event.accept()

if __name__=='__main__':
    app=W.QApplication(sys.argv);app.setStyle('Fusion');window=Window();window.show();sys.exit(app.exec())
